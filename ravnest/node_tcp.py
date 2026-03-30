import multiprocessing
from threading import Thread
import psutil
import pickle
import shutil
import time
import torch
import torch.distributed as dist
from datetime import timedelta

from .communication import Communication_Torch, Communication_GRPC
from .compute import Compute
from .pipeline_split.split_spec import get_split_spec
from .utils import *
from .strings import *
from .globals import g

from .protos.server_pb2_grpc import CommServerStub
from .protos.server_pb2 import CheckBufferStatus

mp = multiprocessing.get_context('spawn')

class Node():
    """
    Responsible for managing the computational and communication aspects of a distributed machine learning model, including model initialization, parameter synchronization, forward and backward passes, loss computation, and communication between different nodes in the system.
    
    :param name: The name of the node. Strictly in the format: 'node_0', 'node_17' etc.
    :type name: str
    :param model: The PyTorch model associated with the node.
    :type model: torch.nn.Module
    :param optimizer: The optimizer used for training the model.
    :type optimizer: torch.optim.Optimizer
    :param optimizer_params: Parameters for the optimizer.
    :type optimizer_params: dict
    :param lr_scheduler: The learning rate scheduler.
    :type lr_scheduler: torch.optim.lr_scheduler
    :param lr_scheduler_params: Parameters for the learning rate scheduler.
    :type lr_scheduler_params: dict
    :param lr_step_on_epoch_change: Whether to step the learning rate scheduler on epoch change.
    :type lr_step_on_epoch_change: bool
    :param criterion: The loss function.
    :type criterion: callable
    :param update_frequency: Frequency of model parameter updates.
    :type update_frequency: int
    :param reduce_factor: Frequency at which all-reduce will be triggered i.e. trigger all-reduce every time these many updates are done.
    :type reduce_factor: int
    :param labels: Dataloader containing labels.
    :type labels: torch.utils.data.DataLoader
    :param test_labels: Test labels for validation.
    :type test_labels: torch.utils.data.DataLoader
    :param device: The device on which the model will be run (CPU or GPU).
    :type device: torch.device
    :param loss_filename: The filename to save loss values.
    :type loss_filename: str
    :param compression: Whether to use compression.
    :type compression: bool
    :param kwargs: Additional arguments.
    :type kwargs: dict
    """

    def __init__(self, model=None, optimizer=None, optimizer_params={}, update_frequency = 1, batch_size=None, seq_length=None, cluster_length=None,
                 dist_timeout=10, reduce_factor=None, labels=None, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), dtype='float16',
                 mode=NodeModes.TRAIN, loss_filename='losses.txt', recompute=False, backend = 'grpc', compression=False, average_optim=False, **kwargs):
        
        self.backend = backend
        self.loss_filename = loss_filename

        self.reset()

        if model is None:
            self.model = torch.jit.load(kwargs['template_path']+'submod.pt')
        else:
            self.model = model
        
        self.device = device
        self.compression = compression
        self.dist_timeout = timedelta(minutes = dist_timeout)

        self.recompute = recompute

        self.output_tensors = {}
        self.input_tensors = {}
        self.n_backwards = 0
        self.n_forwards = 0
        self.forward_pass_id = 0
        self.backward_pass_id = 0
        self.latest_backward_id = 0
        self.update_frequency = update_frequency
        self.mode = mode
        self.dtype = get_torch_dtype(dtype)

        self.steady_state = False
        self.forward_done = False
        
        if not reduce_factor:
            reduce_factor = len(labels)

        self.reduce_threshold = self.update_frequency * reduce_factor

        self.node_status = NodeStatus.IDLE

        self.averaged_params_buffer = {}
        self.average_no = 0
        self.average_optim = average_optim
        self.send_threads = []

        self.cluster_length = cluster_length #kwargs['cluster_length']
        self.batch_size = batch_size
        self.forward_input_shapes=[[batch_size, seq_length, self.model.config.hidden_size]]
        self.backward_input_shapes=[[batch_size, seq_length, self.model.config.hidden_size]]
        self.feedback_shape = [batch_size] #,1]

        self.comm_session = self.init_comm_session()
        self.node_type = self.comm_session.node_type

        self.configure_model()

        self.compute_session = Compute(model = self.model, compression=self.compression, rank=self.comm_session.rank,
                                        input_tensors = self.input_tensors, node_type=self.node_type, backend=self.backend, recompute=self.recompute,
                                        layer_start_idx = self.layer_start_idx, layer_end_idx = self.layer_end_idx, loss_filename=self.loss_filename, device = self.device) 

    def init_comm_session(self):
        assert self.backend in ['grpc', 'gloo', 'nccl'], 'Backend must be set to one of grpc, gloo or nccl'
        if self.backend == 'grpc':

            self.comm_session = Communication_GRPC(name=self.name,
                                            model=self.model,
                                            optimizer=self.optimizer,
                                            node_type=self.node_type,
                                            local_address = self.local_address,
                                            rank=self.rank,
                                            ring_size=self.ring_size,
                                            ring_param_keys=self.ring_param_keys,
                                            ring_ids = self.ring_ids,
                                            param_address_mapping=self.param_address_mapping, 
                                            device=self.device,
                                            compression=self.compression,
                                            forward_target_host=self.forward_target_host, 
                                            forward_target_port=self.forward_target_port, 
                                            backward_target_host=self.backward_target_host, 
                                            backward_target_port=self.backward_target_port, 
                                            retrieve_latest_params_data=self.retrieve_latest_params_data,
                                            output_tensors=self.output_tensors, 
                                            input_tensors=self.input_tensors,
                                            submod_file=self.submod_file, 
                                            tensor_id=self.tensor_id, 
                                            averaged_params_buffer=self.averaged_params_buffer,
                                            average_no=self.average_no,
                                            average_optim = self.average_optim,
                                            output_template=self.output_template,
                                            model_inputs_template=self.model_inputs_template
                                            )
        else:
            
            

            self.comm_session = Communication_Torch(input_tensors=self.input_tensors,
                                                    dist_timeout = self.dist_timeout,
                                                    backend=self.backend, mode=self.mode,
                                                    forward_input_shapes=self.forward_input_shapes,
                                                    backward_input_shapes=self.backward_input_shapes,
                                                    feedback_shape=self.feedback_shape,
                                                    dtype=self.dtype, device=self.device)
        
        return self.comm_session
    
    def configure_model(self):
        split_spec_class = get_split_spec(self.model)
        print('Split Spec Class: ', split_spec_class)
        split_spec = split_spec_class(stage=self.comm_session.rank, node_type=self.node_type, model=self.model, num_stages=self.comm_session.world_size)
        split_spec.configure_stage_model()
        self.model.to(self.dtype)
        self.model.to(self.device)
        self.layer_start_idx, self.layer_end_idx = split_spec.start_idx, split_spec.end_idx
        print('self.layer_start_idx: ', self.layer_start_idx)
        print('self.layer_end_idx: ', self.layer_end_idx)
        print('Attention mechanism: ', self.model.config._attn_implementation)

    def check_forward_buffer(self, tensors=None, no_grad=False, **kwargs):
        monitor_flag_break = False
        outputs = None

        if not self.comm_session.is_receiving_fwd:
            self.comm_session.start_forward_recv()
            self.comm_session.is_receiving_fwd = True

        if self.comm_session.forward_recv_works_done():
            values = self.comm_session.forward_ips[0]

            if self.backend == 'grpc':
                action = values['action']
                if action == ActionTypes.FORWARD:
                    if self.node_type == NodeTypes.LEAF:
                        action = 'leaf_forward'
                    elif self.node_type == NodeTypes.STEM:
                        action = 'stem_forward'
                    g.forward_done = True
                    
                if action == ActionTypes.NO_GRAD_FORWARD:
                    if self.node_type == NodeTypes.LEAF:
                        action = 'leaf_no_grad_forward'
                    elif self.node_type == NodeTypes.STEM:
                        action = 'stem_no_grad_forward'
            else:
                if self.node_type == NodeTypes.LEAF:
                    action = 'leaf_forward'
                elif self.node_type == NodeTypes.STEM:
                    action = 'stem_forward'
                
                if no_grad:
                    action = 'no_grad_' + action
                else:
                    g.forward_done = True
            
            self.comm_session.is_receiving_fwd = False
            inputs = self.create_intermediate_input_args(values, **kwargs)
            outputs = getattr(self, action)(**inputs)
            monitor_flag_break = True

        self.node_status = NodeStatus.IDLE
        return monitor_flag_break, outputs

    def monitor_forward_buffer(self, no_grad=False, **kwargs):
        while True:
            monitor_flag_break, outputs = self.check_forward_buffer(no_grad=no_grad, **kwargs)
            if monitor_flag_break:
                break
            time.sleep(0)

        return outputs

    def check_backward_buffer(self):
        monitor_flag_break = False

        if self.comm_session.backward_recv_works_done():
            values = self.comm_session.backward_ips
            self.comm_session.start_backward_recv()
            action = 'stem_backward'

            getattr(self, action)(values)
            monitor_flag_break = True

            if not self.steady_state:
                self.steady_state = True

            self.node_status = NodeStatus.IDLE
        return monitor_flag_break
    
    def monitor_backward_buffer(self):
        while True:
            monitor_flag_break = self.check_backward_buffer()
            if monitor_flag_break:
                break
            time.sleep(0)

    def join_send_threads(self):
        if len(self.send_threads)>0:
            for send_threads in self.send_threads:
                send_threads.join()

    def forward(self, tensors=None, **kwargs):
        outputs = None
        if self.node_type == NodeTypes.ROOT:
            if self.forward_pass_id - self.latest_backward_id <= self.cluster_length:
                self.forward_compute(tensors, **kwargs)
        elif self.node_type == NodeTypes.STEM:
            self.check_forward_buffer(tensors, **kwargs)
        else:
            outputs = self.monitor_forward_buffer(**kwargs)
        return outputs
    
    def backward(self, loss=None):
        if self.node_type == NodeTypes.ROOT or self.node_type == NodeTypes.STEM:
            self.backward_monitor_flag = self.check_backward_buffer()
        elif self.node_type == NodeTypes.LEAF:
            self.leaf_backward_compute(loss)
            self.backward_monitor_flag = True
        self.join_send_threads()

    def no_grad_forward(self, tensors=None, **kwargs):
        
        outputs = None
        if self.node_type == NodeTypes.ROOT:
            self.no_grad_forward_compute(tensors, **kwargs)
        else:
            outputs = self.monitor_forward_buffer(no_grad=True, **kwargs)
        self.join_send_threads()
        return outputs

    # @torch.inference_mode()
    @torch.no_grad()
    def generate(self, input_ids=None, tokenizer=None, max_seq_length=50, top_k=1, temperature=1.0, **kwargs):
        self.num_generated_tokens = 0
        while self.num_generated_tokens < max_seq_length:
            self.comm_session.forward_input_shapes[0][1] = input_ids.shape[1]
            outputs = self.no_grad_forward(input_ids=input_ids, **kwargs)
            if self.node_type == NodeTypes.LEAF:
                last_token_logits = outputs.logits[:, -1, :]
                next_token_ids = sample_token_from_logits(last_token_logits, top_k, temperature)
                self.comm_session.trigger_feedback_send(next_token_ids)
                input_ids = torch.cat((input_ids, next_token_ids), dim=-1)
            else:
                self.comm_session.start_feedback_recv()
                while not self.comm_session.feedback_recv_work_done():
                    time.sleep(0)
                next_token_ids = self.comm_session.feedback_ip
                input_ids = torch.cat((input_ids, next_token_ids), dim=-1)
            self.num_generated_tokens += 1
        
        return tokenizer_decode_batch(input_ids, tokenizer)

    def forward_compute(self, tensors=None, **kwargs):
        """Initiate a forward computation request.

        Adds the forward computation request to the load forward buffer,
        ensuring synchronization and handling of computational resources.

        :param tensors: Input tensors for the forward computation, defaults to None
        :type tensors: torch.Tensor, optional
        :param kwargs: Additional keyword arguments for the computation, defaults to {}
        :type kwargs: dict, optional
        """
        if tensors is not None:
            tensors = tensors.to(self.device)

        modified_kwargs = {}
        for kwarg_key, kwarg_val in kwargs.items():
            if isinstance(kwarg_val, torch.Tensor):
                modified_kwargs['l_'+kwarg_key+'_'] = kwarg_val.to(self.device)
            else:
                modified_kwargs['l_'+kwarg_key+'_'] = kwarg_val

        self.node_status = NodeStatus.FORWARD
        outputs = self.compute_session.root_forward_compute(tensors, self.forward_pass_id, **modified_kwargs)

        self.n_forwards += 1

        if self.backend == 'grpc':

            if self.n_forwards - self.latest_backward_id > (self.cluster_length - 1):
                self.steady_state = True

            sent_data = self.comm_session.create_forward_payload(outputs, tensors, steady_state=self.steady_state)
            self.trigger_send(sent_data, type=ActionTypes.FORWARD)
        else:
            if isinstance(outputs, tuple):
                for output in outputs:
                    self.comm_session.trigger_forward_send(output.detach().clone())
            else:
                self.comm_session.trigger_forward_send(outputs.detach().clone())
        self.forward_pass_id += 1
        g.forward_done = True
        self.root_compute = True
        self.node_status = NodeStatus.IDLE

    def leaf_forward(self, values):
        if self.backend == 'grpc':
            data = values['data']
        else:
            data = values
        model_args, outputs = self.compute_session.leaf_forward(data)
        self.leaf_model_args = model_args
        return outputs

    def leaf_backward_compute(self, loss):
        self.node_status = NodeStatus.BACKWARD
        self.compute_session.leaf_backward(loss)

        sent_data = self.comm_session.create_backward_payload(forward_pass_id=self.forward_pass_id, model_args=self.leaf_model_args)
        
        if self.backend == 'grpc':
            t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.BACKWARD,))
            self.send_threads.append(t)
            t.start()
        else:
            if isinstance(sent_data, torch.Tensor):
                self.comm_session.trigger_backward_send(sent_data)
            else:
                for grad in sent_data:
                    self.comm_session.trigger_backward_send(grad)

        self.forward_pass_id += 1
        self.n_backwards += 1   

    def no_grad_leaf_forward(self, **kwargs):
        output = self.compute_session.leaf_no_grad_forward(**kwargs)
        return output

    def no_grad_forward_compute(self, tensors=None, **kwargs):
        """Perform a forward pass without computing gradients.

        Executes a forward pass without gradient computation and sends
        the output to the designated target host and port.

        :param tensors: Input tensors for the forward pass, defaults to None
        :type tensors: torch.Tensor, optional
        :param output_type: Type of output computation (e.g., validation accuracy), defaults to None
        :type output_type: str, optional
        """
        if tensors is not None:
            tensors = tensors.to(self.device)
        self.node_status = NodeStatus.FORWARD
        
        outputs = self.compute_session.root_no_grad_forward_compute(tensors=tensors, **kwargs)
        if self.backend == 'grpc':

            sent_data = self.comm_session.create_no_grad_forward_payload(outputs, tensors=tensors)
            self.trigger_send(sent_data, type=ActionTypes.FORWARD)
        else:
            if isinstance(outputs, tuple):
                for output in outputs:
                    self.comm_session.trigger_forward_send(output)
            elif isinstance(outputs, dict):
                self.comm_session.trigger_forward_send(outputs['hidden_states'])
            else:
                self.comm_session.trigger_forward_send(outputs)
        self.node_status = NodeStatus.IDLE

    def stem_forward(self, values):
        self.node_status = NodeStatus.FORWARD
        if self.backend == 'grpc':
            data = values['data']
            self.steady_state = values['steady_state']
        else:
            data = values
        
        outputs = self.compute_session.middle_forward_compute(data, forward_pass_id=self.forward_pass_id)

        if self.backend == 'grpc':
            sent_data = self.comm_session.create_forward_payload(outputs, data=data)
            t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.FORWARD,))
            self.send_threads.append(t)
            t.start()
        else:
            if isinstance(outputs, tuple):
                for output in outputs:
                    self.comm_session.trigger_forward_send(output.detach().clone())
            else:
                self.comm_session.trigger_forward_send(outputs.detach().clone())
        
        self.forward_pass_id += 1
        self.n_forwards += 1

    def no_grad_stem_forward(self, **kwargs):
        self.node_status = NodeStatus.FORWARD
        outputs = self.compute_session.middle_no_grad_forward_compute(**kwargs)
        if isinstance(outputs, tuple):
            for output in outputs:
                self.comm_session.trigger_forward_send(output)
        elif isinstance(outputs, dict):
            self.comm_session.trigger_forward_send(outputs['hidden_states'])
        else:
            self.comm_session.trigger_forward_send(outputs)

    def stem_backward(self, values):
        self.node_status = NodeStatus.BACKWARD
        if self.backend == 'grpc':
            gradient_data = values['data']
            forward_pass_id = values['forward_pass_id']
        else:
            forward_pass_id = self.backward_pass_id
            gradient_data = values
        
        self.latest_backward_id = forward_pass_id

        pass_grad_keys = self.compute_session.middle_backward_compute(gradient_data, forward_pass_id)

        if self.node_type != NodeTypes.ROOT:
            if self.backend == 'grpc':
                sent_data = self.comm_session.create_backward_payload(forward_pass_id=forward_pass_id, pass_grad_keys=pass_grad_keys, gradient_dict=gradient_data)
                t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.BACKWARD,))
                self.send_threads.append(t)
                t.start()
            else:
                sent_data = self.comm_session.create_backward_payload(forward_pass_id=forward_pass_id)
                if isinstance(sent_data, torch.Tensor):
                    self.comm_session.trigger_backward_send(sent_data)
                else:
                    for grad in sent_data:
                        self.comm_session.trigger_backward_send(grad)
            
        if self.input_tensors.get(forward_pass_id, None) is not None:
            del self.input_tensors[forward_pass_id]

        self.backward_pass_id += 1
        self.n_backwards += 1

    def optimizer_step(self):
        if self.backward_monitor_flag:
            # print('Step called')
            self.compute_session.optimizer_step()
        if self.n_backwards % self.reduce_threshold == 0:
            # print('\nPre AVeraged params: ', self.compute_session.model.state_dict()['L__self___bert_encoder_layer_9_output_dense.weight'])#list(self.compute_session.model.state_dict().keys())[0]])
            if self.ring_size > 1 and self.node_type != NodeTypes.LEAF:
                self.comm_session.parallel_ring_reduce()
                if self.version_to_fpid.get(self.current_version, None) is None:
                    if self.compute_session.current_version in self.compute_session.version_to_param:
                        del self.compute_session.version_to_param[self.current_version]

                self.compute_session.current_version += 1
                self.compute_session.update_model_version()


    def dist_func(self, fn, args=(), kwargs={}):
        if self.node_type == NodeTypes.LEAF:
            return fn(*args, **kwargs)
        return None
    
    def create_intermediate_input_args(self, received_inputs=None, **dataloader_kwargs):
        input_kwargs = {}
        for k,v in dataloader_kwargs.items():
            input_kwargs[k] = v.to(self.device)
        input_kwargs['hidden_states'] = received_inputs
        return input_kwargs
    
    def trigger_send(self, data, type=None):
        t1 = time.time()
        with self.comm_session.comm_channel_context(type=type) as channel:
            stub = CommServerStub(channel)

            send_flag = False
            while not send_flag:
                buffer_status = stub.buffer_status(CheckBufferStatus(name=self.name, type=type))
                
                if buffer_status.status == BufferStatus.SEND_BUFFER:
                    send_flag = True
                else:
                    if self.node_type == NodeTypes.ROOT:
                        self.check_backward_buffer()
                
            response = stub.send_buffer(generate_stream(data, type=type))

    def save_submodel(self, value):
        script = torch.jit.script(self.model)
        script.save('{}/{}.pt'.format(self.template_path, self.submod_file))
        os.remove('{}/submod.pt'.format(self.template_path))
        if self.node_type != NodeTypes.LEAF:
            # t = Thread(target=self.comm_session.trigger_send, args=({'action': ActionTypes.SAVE_SUBMODEL}, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
            t = Thread(target=self.trigger_send, args=({'action': ActionTypes.SAVE_SUBMODEL}, ActionTypes.FORWARD,))
            self.send_threads.append(t)
            t.start()                        
        print('SAVE done')

    def wait_for_backwards(self):
        """Wait until all backward passes are completed.

        Checks and waits until all initiated backward computations are finished
        before proceeding with further operations.

        """
        while self.n_backwards < self.n_forwards:
            self.check_backward_buffer()
            time.sleep(0)

    def trigger_save_submodel(self):
        """Trigger saving of the current submodel state.

        Saves the current state of the model to disk and optionally sends
        the updated model state to the designated target host and port.

        """
        script = torch.jit.script(self.model)
        os.makedirs(self.template_path, exist_ok=True)
        script.save('{}/{}.pt'.format(self.template_path,self.submod_file))
        os.remove('{}/submod.pt'.format(self.template_path))
        # self.comm_session.trigger_send({'action': ActionTypes.SAVE_SUBMODEL}, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
        self.trigger_send({'action': ActionTypes.SAVE_SUBMODEL}, type=ActionTypes.FORWARD)
        print('SAVE done')

    def update_with_latest_weights(self):
        latest_sd = self.comm_session.get_latest_weights()
        load_state_dict_conserve_versions(self.compute_session.model, latest_sd)
        self.compute_session.update_model_version()
        print('Model latest weights loaded!')

    def reset(self):
        """Reset the node's auxiliary and stateful data.

        Cleans up temporary directories and files associated with the node,
        preparing it for a fresh start.

        """
        # if os.path.exists('{}_aux'.format(self.name)):
        #     shutil.rmtree('{}_aux'.format(self.name))
        if os.path.exists('trained'):
            shutil.rmtree('trained')
        if os.path.exists(self.loss_filename):
            os.remove(self.loss_filename)
        if os.path.exists('val_accuracies.txt'):
            os.remove('val_accuracies.txt')
