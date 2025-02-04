import multiprocessing
from threading import Thread
import psutil
import pickle
import shutil
import time
import torch.distributed as dist

from .communication import Communication_Torch, Communication_GRPC
from .compute import Compute
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

    def __init__(self, name=None, model=None, optimizer=None, optimizer_params={}, update_frequency = 1, 
                 reduce_factor=None, labels=None, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                 loss_filename='losses.txt', recompute=False, backend = 'grpc', compression=False, average_optim=False, **kwargs):

        self.backend = backend

        node_metadata = load_node_json_configs(node_name=name)
        kwargs.update(node_metadata)

        self.node_type = kwargs.get('node_type', None)
        self.template_path = kwargs.get('template_path', None)[:-1]
        self.local_address = '{}:{}'.format(kwargs.get('local_host', None), kwargs.get('local_port', None))
        self.name = name
        self.loss_filename = loss_filename

        self.reset()

        if model is None:
            self.model = torch.jit.load(kwargs['template_path']+'submod.pt')
        else:
            self.model = model
        
        self.device = device
        self.compression = compression

        # if not next(self.model.parameters()).is_cuda:
        #     self.model.to(device)
        self.model.to(self.device)

        # self.load_forward_buffer = self.manager.list()
        # self.load_backward_buffer = self.manager.list()

        if kwargs.get('ring_ids', None) is not None:
            self.ring_ids = kwargs.get('ring_ids', None)

        self.rank = kwargs.get('rank', None)
        # print('\n Rank: ', self.rank)
        self.ring_size = kwargs.get('ring_size', None)
        self.recompute = recompute
        
        self.ring_param_keys = {}
        data_dict_keys = get_trainable_param_names(model=self.model)
        for i, ring in enumerate(self.ring_ids.items()):
            if i < len(self.ring_ids) - 1:
                keys = data_dict_keys[data_dict_keys.index(ring[1]):data_dict_keys.index(self.ring_ids[ring[0]+1])]
            else:
                keys = data_dict_keys[data_dict_keys.index(ring[1]):]
            
            self.ring_param_keys[ring[0]] = keys

        self.param_address_mapping = {}
        param_addresses = kwargs.get('param_addresses', None)
        self.retrieve_latest_params_data = {}
        # print(param_addresses)
        for i, address_to_param in enumerate(param_addresses.items()):
            if i < len(param_addresses) - 1:
                keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):data_dict_keys.index(param_addresses[list(param_addresses.keys())[i+1]])]
            else:
                keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):]
            
            self.retrieve_latest_params_data[address_to_param[0]] = (keys[0], keys[-1])

            for param_name in keys:
                self.param_address_mapping[param_name] = address_to_param[0]

        # print('Ring param keys: ', self.ring_param_keys.keys())
        # print('Param address mapping: ', self.param_address_mapping)
        # print('State dict: ', self.model.state_dict().keys())

        self.forward_target_host = kwargs.get('forward_target_host', None)
        self.forward_target_port = kwargs.get('forward_target_port', None)
        self.backward_target_host = kwargs.get('backward_target_host', None)
        self.backward_target_port = kwargs.get('backward_target_port', None)

        self.output_tensors = {}
        self.input_tensors = {}
        self.n_backwards = 0
        self.n_forwards = 0
        self.forward_pass_id = 0
        self.backward_pass_id = 0
        self.latest_backward_id = 0
        self.update_frequency = update_frequency

        self.steady_state = False
        self.forward_done = False
        
        if not reduce_factor:
            reduce_factor = len(labels)

        self.reduce_threshold = self.update_frequency * reduce_factor

        self.submod_file = kwargs.get('submod_file', None)
        self.node_status = NodeStatus.IDLE
        self.tensor_id = '0_{}'.format(self.submod_file)#0

        self.averaged_params_buffer = {}
        self.average_no = 0
        self.average_optim = average_optim
        self.send_threads = []

        self.cluster_length = kwargs['cluster_length']
        self.world_size = kwargs.get('cluster_length', None)
        self.rank_ = kwargs.get('node_id', None)
        self.forward_input_shapes=kwargs.get('input_shape', None)
        self.backward_input_shapes=kwargs.get('output_shape', None)

        if kwargs.get('submod_file', None) is not None:
            with open('{}{}_input.pkl'.format(kwargs.get('template_path', None), kwargs.get('submod_file', None)), 'rb') as fout:
                self.input_template = pickle.load(fout)
            with open('{}{}_output.pkl'.format(kwargs.get('template_path', None), kwargs.get('submod_file', None)), 'rb') as fout:
                self.output_template = pickle.load(fout)
            # print(self.input_template)
            self.model_inputs_template = None
            if self.node_type == NodeTypes.ROOT:
                with open('{}model_inputs.pkl'.format(kwargs.get('template_path', None)), 'rb') as fout:
                    self.model_inputs_template = pickle.load(fout)
                self.optimizer = optimizer(current_model_params_clone(self.model), **optimizer_params)
            elif self.node_type == NodeTypes.LEAF:
                self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
            elif self.node_type == NodeTypes.STEM:
                self.optimizer = optimizer(current_model_params_clone(self.model), **optimizer_params)

        self.comm_session = self.init_comm_session()

        self.compute_session = Compute(model = self.model, optimizer = self.optimizer, compression=self.compression,
                                        input_tensors = self.input_tensors, tensor_id = self.tensor_id, output_template = self.output_template, 
                                        input_template = self.input_template, node_type=self.node_type, backend=self.backend, recompute=self.recompute,
                                        submod_file=self.submod_file, loss_filename=self.loss_filename, device = self.device) 
                                        #latest_weights_buffer = self.latest_weights_buffer, latest_weights_lock=self.latest_weights_lock, 


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

            self.comm_session = Communication_Torch(node_type=self.node_type,
                                                input_tensors=self.input_tensors,
                                                backend=self.backend,
                                                rank=self.rank_, #int(os.environ["RANK"])
                                                world_size=self.world_size, #int(os.environ["WORLD_SIZE"])
                                                forward_input_shapes=self.forward_input_shapes,
                                                backward_input_shapes=self.backward_input_shapes,
                                                device=self.device)
        
        return self.comm_session

    def check_forward_buffer(self, no_grad=False):
        monitor_flag_break = False
        outputs = None
        
        if self.comm_session.forward_recv_works_done():
            #not self.forward_work.is_alive(): #self.forward_work.is_completed(): #
            # print('Forward thread over')
            # print('Self forward ip: ', self.forward_ip)
            values = self.comm_session.forward_ips
            # self.forward_ip = None
            # self.forward_work = None
            self.comm_session.start_forward_recv()

            if self.backend == 'grpc':
                action = values['action']
                # print('\n', action, ' Popped from forward buffer')
                if action == ActionTypes.FORWARD:
                    if self.node_type == NodeTypes.LEAF:
                        action = 'leaf_forward'#ActionTypes.FIND_LOSS
                    elif self.node_type == NodeTypes.STEM:
                        action = 'stem_forward'
                    g.forward_done = True
                    
                if action == ActionTypes.NO_GRAD_FORWARD:
                    if self.node_type == NodeTypes.LEAF:
                        # action = ActionTypes.VAL_ACCURACY
                        action = 'leaf_no_grad_forward'
                    elif self.node_type == NodeTypes.STEM:
                        action = 'stem_no_grad_forward'
            else:
                if self.node_type == NodeTypes.LEAF:
                    action = 'leaf_forward'#ActionTypes.FIND_LOSS
                    # outputs = self.leaf_forward(value)
                elif self.node_type == NodeTypes.STEM:
                    action = 'stem_forward'
                    # outputs = self.stem_forward(value)
                
                if no_grad:
                    action = 'no_grad_' + action
                else:
                    g.forward_done = True
            outputs = getattr(self, action)(values) #, self.send_threads)
            monitor_flag_break = True

        self.node_status = NodeStatus.IDLE
        return monitor_flag_break, outputs

    def monitor_forward_buffer(self, no_grad=False):
        while True:
            monitor_flag_break, outputs = self.check_forward_buffer(no_grad=no_grad)
            if monitor_flag_break:
                break
            time.sleep(0)

        return outputs

    def check_backward_buffer(self):
        monitor_flag_break = False

        # if self.backward_work is not None:
        if self.comm_session.backward_recv_works_done():
            #self.backward_work.is_completed():
            # if not self.backward_work.is_alive():
            # print('Stem Backward recieved for: ', self.backward_pass_id)
            # print('Backward thread over')
            values = self.comm_session.backward_ips #(fp_id, self.backward_ip)
            # self.backward_ip = None
            # self.backward_work = None
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
            self.check_forward_buffer()
        else:
            outputs = self.monitor_forward_buffer()
        return outputs
    
    def backward(self, loss=None):
        if self.node_type == NodeTypes.ROOT or self.node_type == NodeTypes.STEM:
            self.backward_monitor_flag = self.check_backward_buffer()
        elif self.node_type == NodeTypes.LEAF:
            self.leaf_backward_compute(loss)
            self.backward_monitor_flag = True
        # print('Version to param: ', print(self.compute_session.version_to_param.keys()))
        self.join_send_threads()

    def no_grad_forward(self, tensors=None, **kwargs):
        
        outputs = None
        if self.node_type == NodeTypes.ROOT:
            self.no_grad_forward_compute(tensors, **kwargs)
        else:
            outputs = self.monitor_forward_buffer(no_grad=True)
        self.join_send_threads()
        return outputs

    def forward_compute(self, tensors=None, **kwargs):
        """Initiate a forward computation request.

        Adds the forward computation request to the load forward buffer,
        ensuring synchronization and handling of computational resources.

        :param tensors: Input tensors for the forward computation, defaults to None
        :type tensors: torch.Tensor, optional
        :param kwargs: Additional keyword arguments for the computation, defaults to {}
        :type kwargs: dict, optional
        """
        # t = time.time()
        if tensors is not None:
            tensors = tensors.to(self.device)

        modified_kwargs = {}
        for kwarg_key, kwarg_val in kwargs.items():
            if isinstance(kwarg_val, torch.Tensor):
                modified_kwargs['l_'+kwarg_key+'_'] = kwarg_val.to(self.device)
            else:
                modified_kwargs['l_'+kwarg_key+'_'] = kwarg_val

        self.node_status = NodeStatus.FORWARD

        # print('Before Root Forward: ')
        # check_gpu_usage()
        outputs = self.compute_session.root_forward_compute(tensors, self.forward_pass_id, **modified_kwargs)
        # print('Total root forward compute time: ', time.time() - t)
        # print('After Root Forward: ')
        # check_gpu_usage()

        self.n_forwards += 1

        if self.backend == 'grpc':
            # self.n_forwards += 1

            if self.n_forwards - self.latest_backward_id > (self.cluster_length - 1):
                self.steady_state = True

            sent_data = self.comm_session.create_forward_payload(outputs, tensors, steady_state=self.steady_state) #, forward_pass_id=self.forward_pass_id)

            
            # self.comm_session.trigger_send(sent_data, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
            self.trigger_send(sent_data, type=ActionTypes.FORWARD)
        else:
            # t = time.time()
            # self.forward_to_comm_pipe.send(output.detach().clone())
            # print('Forward sending')
            # self.forward_send_work = dist.isend(output.detach().clone(), self.rank_ + 1)
            # self.n_forwards += 1
            # work = dist.isend(output.detach().clone(), self.rank_ + 1)
            # self.forward_send_work = self.check_work_thread(work, type='send_fwd')
            # print('Forward sent for: ', self.forward_pass_id)
            if isinstance(outputs, tuple):
                for output in outputs:
                    self.comm_session.trigger_forward_send(output.detach().clone())
            else:
                self.comm_session.trigger_forward_send(outputs.detach().clone())
            # print('Forward Snt')
            # self.forward_comm_buffer.append(output.detach().clone())
            # print('Time taken to send to forward comm pipe: ', time.time() - t)
            # self.forward_send_buffer.append(output.detach().clone())
            # self.comm_session.send_forward_tensors(output.detach().clone())
        self.forward_pass_id += 1
        g.forward_done = True
        self.root_compute = True
        self.node_status = NodeStatus.IDLE

    def leaf_forward(self, values):
        if self.backend == 'grpc':
            data = values['data']
        else:
            data = values
        # print('leaf forward ip: ', data[0])
        # self.forward_pass_id = value['forward_pass_id']
        model_args, outputs = self.compute_session.leaf_forward(data)
        # print('Leaf forward done for: ', self.forward_pass_id)
        self.leaf_model_args = model_args
        return outputs

    def leaf_backward_compute(self, loss):
        self.node_status = NodeStatus.BACKWARD
        self.compute_session.leaf_backward(loss)

        sent_data = self.comm_session.create_backward_payload(forward_pass_id=self.forward_pass_id, model_args=self.leaf_model_args)
        
        # t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.BACKWARD, self.backward_target_host, self.backward_target_port,))
        if self.backend == 'grpc':
            t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.BACKWARD,))
            self.send_threads.append(t)
            t.start()
        else:
            # t = time.time()
            # self.backward_to_comm_pipe.send(sent_data)

            # dist.send(torch.tensor(sent_data[0], dtype=torch.int16), self.rank_-1)
            # print('sending backward')
            # dist.send(sent_data, self.rank_-1)
            # print('sent backward')

            # work = dist.isend(sent_data, self.rank_-1)#dist.isend(output.detach().clone(), self.rank_ + 1)
            # self.backward_send_work = self.check_work_thread(work, type='send_backward')
            # print('Backward sent for: ', self.forward_pass_id)
            if isinstance(sent_data, torch.Tensor):
                self.comm_session.trigger_backward_send(sent_data)
            else:
                for grad in sent_data:
                    self.comm_session.trigger_backward_send(grad)
            # self.backward_comm_buffer.append(sent_data)
            # print('Time taken to send to backward comm pipe: ', time.time() - t)
            # self.comm_session.send_grad_tensors(*sent_data)
            # self.backward_send_buffer.append(sent_data)

        # print('find_loss done. Used RAM %: ', psutil.virtual_memory().percent)
        self.forward_pass_id += 1
        self.n_backwards += 1
        # print('N_backwards: ', self.n_backwards)

        # if self.n_backwards % self.reduce_threshold == 0:
        #     # print('\nPre AVeraged params: ', self.compute_session.model.state_dict()['L__self___bert_encoder_layer_9_output_dense.weight'])#list(self.compute_session.model.state_dict().keys())[0]])

        #     self.comm_session.parallel_ring_reduce()
        #     # print('\nAVeraged params: ', self.compute_session.model.state_dict()['L__self___bert_encoder_layer_9_output_dense.weight'])#[list(self.compute_session.model.state_dict().keys())[0]])

        #     if self.version_to_fpid.get(self.current_version, None) is None:
        #         if self.current_version in self.version_to_param:
        #             del self.version_to_param[self.current_version]

        #     self.current_version += 1
        #     self.update_model_version()

        # if self.device.type == 'cuda':
        #     # print('Sync')
        #     torch.cuda.synchronize()    

    def no_grad_leaf_forward(self, value):  
        if self.backend == 'grpc':  
            data = value['data']
        else:
            data = value
            # print('Received input no grad leaf: ', data[0])
        output = self.compute_session.leaf_no_grad_forward(data)
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
        tensors = tensors.to(self.device)
        # self.comm_session.parallel_ring_reduce()
        self.node_status = NodeStatus.FORWARD
        
        outputs = self.compute_session.root_no_grad_forward_compute(tensors=tensors, **kwargs)

        if self.backend == 'grpc':

            sent_data = self.comm_session.create_no_grad_forward_payload(outputs, tensors=tensors)

            # self.comm_session.trigger_send(sent_data, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
            self.trigger_send(sent_data, type=ActionTypes.FORWARD)
        else:
            # work = dist.isend(output, self.rank_ + 1)
            # self.no_grad_forward_send_work = self.check_work_thread(work)
            # dist.send(output, self.rank_ + 1)
            # self.trigger_forward_send(output)

            if isinstance(outputs, tuple):
                for output in outputs:
                    self.comm_session.trigger_forward_send(output)
            else:
                self.comm_session.trigger_forward_send(outputs)
            # print('sent no grad forward: ', output[0])
        # print('No Grad forward compute done')
        self.node_status = NodeStatus.IDLE

    def stem_forward(self, values):
        # print('n_backwards in FORWARD: ', self.n_backwards)
        self.node_status = NodeStatus.FORWARD
        if self.backend == 'grpc':
            data = values['data']
            # forward_pass_id = value['forward_pass_id']
            self.steady_state = values['steady_state']
            # print('Start of forward: ', forward_pass_id)
        else:
            data = values
        
        outputs = self.compute_session.middle_forward_compute(data, forward_pass_id=self.forward_pass_id)

        if self.backend == 'grpc':

            sent_data = self.comm_session.create_forward_payload(outputs, data=data) #, forward_pass_id=self.forward_pass_id)

            # t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
            t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.FORWARD,))
            self.send_threads.append(t)
            t.start()
        else:
            # self.forward_to_comm_pipe.send(output.detach().clone())
            # self.forward_comm_buffer.append(output.detach().clone())
            # self.comm_session.send_forward_tensors(output.detach().clone())
            # self.forward_send_buffer.append(output.detach().clone())
            if isinstance(outputs, tuple):
                for output in outputs:
                    self.comm_session.trigger_forward_send(output.detach().clone())
            else:
                self.comm_session.trigger_forward_send(outputs.detach().clone())

            # self.trigger_forward_send([output.detach().clone() for output in outputs])
        
        self.forward_pass_id += 1
        self.n_forwards += 1
        # print('Forward Done Used RAM %: ', psutil.virtual_memory().percent)

    def no_grad_stem_forward(self, value):
        # self.comm_session.parallel_ring_reduce()
        self.node_status = NodeStatus.FORWARD
        # print('No grad forward')
        if self.backend == 'grpc':
            data = value['data']
        else:
            data = value
        
        outputs = self.compute_session.middle_no_grad_forward_compute(data)
        # self.trigger_forward_send(output)

        if isinstance(outputs, tuple):
            for output in outputs:
                self.comm_session.trigger_forward_send(output)
        else:
            self.comm_session.trigger_forward_send(outputs)
        
        # sent_data = self.comm_session.create_no_grad_forward_payload(output, data=data)

        # # t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
        # t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.FORWARD,))
        # self.send_threads.append(t)
        # t.start()

    def stem_backward(self, values):
        self.node_status = NodeStatus.BACKWARD
        if self.backend == 'grpc':
            gradient_data = values['data']
            forward_pass_id = values['forward_pass_id']
        else:
            forward_pass_id = self.backward_pass_id#value[0].item()
            gradient_data = values#[1]
        
        self.latest_backward_id = forward_pass_id
        # print('Start of backward: ', forward_pass_id)

        pass_grad_keys = self.compute_session.middle_backward_compute(gradient_data, forward_pass_id)

        if self.node_type != NodeTypes.ROOT:
            if self.backend == 'grpc':
                sent_data = self.comm_session.create_backward_payload(forward_pass_id=forward_pass_id, pass_grad_keys=pass_grad_keys, gradient_dict=gradient_data)

                # t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.BACKWARD, self.backward_target_host, self.backward_target_port,))
                t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.BACKWARD,))
                self.send_threads.append(t)
                t.start()
            else:
                sent_data = self.comm_session.create_backward_payload(forward_pass_id=forward_pass_id)
                # self.backward_to_comm_pipe.send(sent_data)
                # dist.send(sent_data, self.rank_ - 1)
                # self.backward_comm_buffer.append(sent_data)
                # self.comm_session.send_grad_tensors(*sent_data)
                # self.backward_send_buffer.append(sent_data)
                # self.trigger_backward_send(sent_data)
                if isinstance(sent_data, torch.Tensor):
                    self.comm_session.trigger_backward_send(sent_data)
                else:
                    for grad in sent_data:
                        self.comm_session.trigger_backward_send(grad)
            
        if self.input_tensors.get(forward_pass_id, None) is not None:
            del self.input_tensors[forward_pass_id]

        # print('Backward done, Used RAM %: ', psutil.virtual_memory().percent)
        self.backward_pass_id += 1
        self.n_backwards += 1
        # print('n-backward: ', self.n_backwards)

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
    
    def trigger_send(self, data, type=None):
        # with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
        # print('Forward and backward buffer lengths: ', len(self.load_forward_buffer), len(self.load_backward_buffer))
        t1 = time.time()
        with self.comm_session.comm_channel_context(type=type) as channel:
            stub = CommServerStub(channel)

            send_flag = False
            # print('Send trigger started', type)
            while not send_flag:
                buffer_status = stub.buffer_status(CheckBufferStatus(name=self.name, type=type))
                
                if buffer_status.status == BufferStatus.SEND_BUFFER:
                    send_flag = True
                else:
                    if self.node_type == NodeTypes.ROOT:
                        self.check_backward_buffer()
                        # print('Root check backward')
                
            response = stub.send_buffer(generate_stream(data, type=type))
            # print('Send trigger finished', type)
        print('Trigger send time for: ', type, time.time() - t1)

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
            # time.sleep(1)
            self.check_backward_buffer()

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
        if os.path.exists('{}_aux'.format(self.name)):
            shutil.rmtree('{}_aux'.format(self.name))
        if os.path.exists('trained'):
            shutil.rmtree('trained')
        if os.path.exists(self.loss_filename):
            os.remove(self.loss_filename)
        if os.path.exists('val_accuracies.txt'):
            os.remove('val_accuracies.txt')
