from concurrent import futures
import asyncio
import grpc
import threading
import multiprocessing
import threading
from threading import Thread
import numpy as np
import psutil
import pickle
import shutil
import time
from .communication import Communication
from .compute import Compute
from .utils import *
from .strings import *
from .endpoints import GrpcService

from .protos.server_pb2_grpc import add_CommServerServicer_to_server

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

    def __init__(self, name=None, model=None, optimizer=None, optimizer_params={}, lr_scheduler=None, lr_scheduler_params={}, lr_step_on_epoch_change=True, criterion=None, 
                 update_frequency = 1, reduce_factor=None, labels=None, test_labels=None, device = torch.device('cpu'), loss_filename='losses.txt', compression=False, average_optim=False, **kwargs):
        self.manager = mp.Manager()
        self.forward_lock = mp.Lock()
        self.backward_lock = mp.Lock()
        self.latest_weights_lock = mp.Lock()
        self.reduce_lock = mp.Lock()
        self.gather_lock = mp.Lock()

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

        if not next(self.model.parameters()).is_cuda:
            self.model.to(device)

        self.load_forward_buffer = self.manager.list()
        self.load_backward_buffer = self.manager.list()
        self.reduce_ring_buffers = self.manager.dict()
        self.gather_ring_buffers = self.manager.dict()
        self.latest_weights_buffer = self.manager.dict()
        self.reduce_iteration = self.manager.dict()
        self.gather_iteration = self.manager.dict()


        self.start_server_flag = self.manager.Value(bool, False)

        if kwargs.get('ring_ids', None) is not None:
            self.ring_ids = kwargs.get('ring_ids', None)

            for ring_id, _ in self.ring_ids.items():
                self.reduce_iteration[ring_id] = 0
                self.gather_iteration[ring_id] = 0
            print('ring ids: ', self.ring_ids)

        self.rank = kwargs.get('rank', None)
        print('\n Rank: ', self.rank)
        self.ring_size = kwargs.get('ring_size', None)
        
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
        print(param_addresses)
        for i, address_to_param in enumerate(param_addresses.items()):
            if i < len(param_addresses) - 1:
                keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):data_dict_keys.index(param_addresses[list(param_addresses.keys())[i+1]])]
            else:
                keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):]
            
            self.retrieve_latest_params_data[address_to_param[0]] = (keys[0], keys[-1])

            for param_name in keys:
                self.param_address_mapping[param_name] = address_to_param[0]

        print('Ring param keys: ', self.ring_param_keys.keys())
        # print('Param address mapping: ', self.param_address_mapping)
        # print('State dict: ', self.model.state_dict().keys())

        if self.node_type == NodeTypes.LEAF:
            self.criterion = criterion

            if test_labels is not None:
                self.test_labels = test_labels
                self.test_labels_iterator = None
            
            if labels is not None:
                self.labels = labels
                if isinstance(labels, torch.Tensor):
                    self.labels_iterator = labels
                else:
                    self.labels_iterator = iter(labels)

        else:
            self.criterion = None
            self.test_labels = None
            self.test_labels_iterator = None
            self.labels = None
            self.labels_iterator = None

        self.net_val_accuracy = []

        self.forward_target_host = kwargs.get('forward_target_host', None)
        self.forward_target_port = kwargs.get('forward_target_port', None)
        self.backward_target_host = kwargs.get('backward_target_host', None)
        self.backward_target_port = kwargs.get('backward_target_port', None)

        self.output_tensors = {}
        self.input_tensors = {}
        self.n_backwards = 0
        self.n_forwards = 0
        self.forward_pass_id = 0
        self.latest_backward_id = 0
        self.update_frequency = update_frequency
        
        if not reduce_factor:
            reduce_factor = len(labels)

        self.reduce_threshold = self.update_frequency * reduce_factor

        self.submod_file = kwargs.get('submod_file', None)
        self.node_status = NodeStatus.IDLE
        self.tensor_id = '0_{}'.format(self.submod_file)#0

        self.averaged_params_buffer = {}
        self.average_no = 0
        self.average_optim = average_optim

        self.cluster_length = kwargs['cluster_length']

        self.lr_step_on_epoch_change = lr_step_on_epoch_change

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

            self.lr_scheduler = None
            if lr_scheduler is not None:
                self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_params)

        self.compute_session = Compute(model = self.model, optimizer = self.optimizer, 
                                        criterion=self.criterion, compression=self.compression,
                                        input_tensors = self.input_tensors, latest_weights_buffer = self.latest_weights_buffer,
                                        latest_weights_lock=self.latest_weights_lock, tensor_id = self.tensor_id, output_template = self.output_template, 
                                        input_template = self.input_template, node_type=self.node_type,
                                        submod_file=self.submod_file, loss_filename=self.loss_filename, device = self.device)

        self.comm_session = Communication(name=self.name,
                                          model=self.model,
                                          optimizer=self.optimizer,
                                          node_type=self.node_type,
                                          rank=self.rank,
                                          ring_size=self.ring_size,
                                          ring_param_keys=self.ring_param_keys,
                                          ring_ids = self.ring_ids,
                                          param_address_mapping=self.param_address_mapping, 
                                          reduce_lock=self.reduce_lock, 
                                          gather_lock=self.gather_lock,
                                          device=self.device,
                                          compression=self.compression,
                                          forward_target_host=self.forward_target_host, 
                                          forward_target_port=self.forward_target_port, 
                                          backward_target_host=self.backward_target_host, 
                                          backward_target_port=self.backward_target_port, 
                                          retrieve_latest_params_data=self.retrieve_latest_params_data,
                                          output_tensors=self.output_tensors, 
                                          input_tensors=self.input_tensors,
                                          reduce_ring_buffers=self.reduce_ring_buffers, 
                                          gather_ring_buffers=self.gather_ring_buffers,
                                          reduce_iteration=self.reduce_iteration, 
                                          gather_iteration=self.gather_iteration,
                                          submod_file=self.submod_file, 
                                          tensor_id=self.tensor_id, 
                                          averaged_params_buffer=self.averaged_params_buffer,
                                          average_no=self.average_no,
                                          average_optim = self.average_optim,
                                          output_template=self.output_template,
                                          model_inputs_template=self.model_inputs_template
                                          )

        self.start()

    def init_server(self, load_forward_buffer=None, load_backward_buffer=None, 
                    reduce_ring_buffers = None, gather_ring_buffers = None, latest_weights_buffer=None, 
                    forward_lock=None, backward_lock=None, reduce_lock=None, gather_lock=None,
                    latest_weights_lock=None, reduce_iteration = None, gather_iteration = None):
        """Initialize the gRPC server for handling communication with other nodes.

        :param load_forward_buffer: Shared buffer for incoming forward pass data, defaults to None
        :type load_forward_buffer: multiprocessing.Manager.list, optional
        :param load_backward_buffer: Shared buffer for incoming backward pass data, defaults to None
        :type load_backward_buffer: multiprocessing.Manager.list, optional
        :param reduce_ring_buffers: Shared dictionary for reduce operation buffers, defaults to None
        :type reduce_ring_buffers: multiprocessing.Manager.dict, optional
        :param gather_ring_buffers: Shared dictionary for gather operation buffers, defaults to None
        :type gather_ring_buffers: multiprocessing.Manager.dict, optional
        :param forward_lock: Lock for synchronizing access to forward buffers, defaults to None
        :type forward_lock: multiprocessing.Lock, optional
        :param backward_lock: Lock for synchronizing access to backward buffers, defaults to None
        :type backward_lock: multiprocessing.Lock, optional
        :param reduce_lock: Lock for synchronizing reduce operations, defaults to None
        :type reduce_lock: multiprocessing.Lock, optional
        :param gather_lock: Lock for synchronizing gather operations, defaults to None
        :type gather_lock: multiprocessing.Lock, optional
        :param reduce_iteration: Shared dictionary for reduce iteration counts, defaults to None
        :type reduce_iteration: multiprocessing.Manager.dict, optional
        :param gather_iteration: Shared dictionary for gather iteration counts, defaults to None
        :type gather_iteration: multiprocessing.Manager.dict, optional
        """
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        add_CommServerServicer_to_server(GrpcService(
            load_forward_buffer=load_forward_buffer, load_backward_buffer=load_backward_buffer, 
            reduce_ring_buffers=reduce_ring_buffers, gather_ring_buffers=gather_ring_buffers, 
            latest_weights_buffer=latest_weights_buffer,
            forward_lock=forward_lock, backward_lock=backward_lock, reduce_lock=reduce_lock, 
            gather_lock=gather_lock,latest_weights_lock=latest_weights_lock,
            reduce_iteration = reduce_iteration, gather_iteration = gather_iteration), self.server)
        print('Length of forward buffer: ', len(load_backward_buffer), os.getpid())


    def grpc_server_serve(self):
        """Starts the gRPC server and listens for incoming connections.
        """
        self.server.add_insecure_port(self.local_address)
        self.server.start()
        print('Listening on : ', self.local_address)
        self.start_server_flag.value = True
        self.server.wait_for_termination()


    def start_grpc_server(self):
        """Start the gRPC server asynchronously.

        Uses asyncio to start the gRPC server in an asynchronous manner.
        """
        asyncio.get_event_loop().run_until_complete(self.grpc_server_serve())

    def start(self):
        """Start the gRPC server and buffer checking threads.

        Spawns a process for serving gRPC requests and starts a thread
        for checking and processing incoming data buffers.
        """
        print('Main process: ', os.getpid())
        serve_process = mp.Process(target=self.grpc_server_serve, daemon=True)
        serve_process.start()

        while not self.start_server_flag.value:
            time.sleep(0.5)
        
        buffer_thread = threading.Thread(target=self.check_load_forward_buffer, daemon=True)
        buffer_thread.start()

    def check_load_forward_buffer(self):
        """Check and process the load forward buffer for incoming data.

        Continuously monitors the load forward buffer and processes incoming
        data for forward pass computations.
        """
        while True:
            send_trigger_threads = []
            if len(self.load_backward_buffer) != 0:
                self.backward_lock.acquire(block=True)
                value = self.load_backward_buffer[0]
                del self.load_backward_buffer[0]
                self.backward_lock.release()
                action = value['action']

                if action == ActionTypes.BACKWARD:
                    self.node_status = NodeStatus.BACKWARD
                    gradient_dict = value['data']
                    forward_pass_id = value['forward_pass_id']
                    epoch_change = value['epoch_change']
                    if epoch_change and self.lr_step_on_epoch_change:
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()

                    self.latest_backward_id = forward_pass_id
                    print('Start of backward: ', forward_pass_id)

                    update_flag = False
                    if (self.n_backwards + 1) % self.update_frequency == 0:
                        update_flag = True

                    pass_grad_keys = self.compute_session.middle_backward_compute(gradient_dict, forward_pass_id, update_flag=update_flag)
                    
                    if update_flag and not self.lr_step_on_epoch_change:
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()

                    if self.node_type != NodeTypes.ROOT:
                        gradients = self.comm_session.create_backward_payload(forward_pass_id=forward_pass_id)
                        for pass_key in pass_grad_keys:
                            if pass_key in gradients.keys():
                                # if isinstance(gradient_dict[pass_key], list):
                                #     gradient_dict[pass_key].append(gradients[pass_key])
                                #     gradients[pass_key] = gradient_dict[pass_key]
                                # else:
                                #     gradients[pass_key] = [gradient_dict[pass_key], gradients[pass_key]]
                                assert gradient_dict[pass_key]['dtype'] == gradients[pass_key]['dtype']
                                gradients[pass_key] = {'dtype': gradients[pass_key]['dtype'], 'data': gradient_dict[pass_key]['data'].add_(gradients[pass_key]['data'])}
                            else:
                                gradients[pass_key] = gradient_dict[pass_key]

                        sent_data = {'action':ActionTypes.BACKWARD,
                                    'forward_pass_id':forward_pass_id, 
                                    'data':gradients, 
                                    'epoch_change':epoch_change,
                                    }
                        t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.BACKWARD, self.backward_target_host, self.backward_target_port,))
                        send_trigger_threads.append(t)
                        t.start()
                        
                    if self.input_tensors.get(forward_pass_id, None) is not None:
                        del self.input_tensors[forward_pass_id]

                    print('Backward done, Used RAM %: ', psutil.virtual_memory().percent)
                    self.n_backwards += 1

                    if self.n_backwards % self.reduce_threshold == 0:
                        # print('\nPre AVeraged params: ', self.compute_session.model.state_dict()[list(self.compute_session.model.state_dict().keys())[0]])

                        self.comm_session.parallel_ring_reduce()
                        # self.compute_session.current_version += 1
                        # self.compute_session.version_to_param[self.compute_session.current_version] = self.compute_session.get_params_clone()

                        # self.latest_weights_lock.acquire(block=True)
                        # self.latest_weights_buffer['state_dict'] = self.compute_session.version_to_param[self.compute_session.current_version]
                        # self.latest_weights_lock.release()

                        self.compute_session.update_model_version()

                        # print('\nAVeraged params: ', self.compute_session.model.state_dict()[list(self.compute_session.model.state_dict().keys())[0]])

                    # if self.device.type == 'cuda':
                    #     torch.cuda.synchronize()

            self.node_status = NodeStatus.IDLE

            if len(self.load_forward_buffer) != 0:
                self.forward_lock.acquire(block=True)
                value = self.load_forward_buffer[0]
                del self.load_forward_buffer[0]
                self.forward_lock.release()
                action = value['action']

                if action == ActionTypes.FORWARD and self.node_type == NodeTypes.LEAF:
                    action = ActionTypes.FIND_LOSS
                if action == ActionTypes.NO_GRAD_FORWARD and self.node_type == NodeTypes.LEAF:
                    action = ActionTypes.VAL_ACCURACY

                if action == ActionTypes.ROOT_FORWARD:                    
                    tensors = value['data']
                    kwargs = value['kwargs']

                    if tensors is not None:
                        tensors = tensors.to(self.device)

                    modified_kwargs = {}
                    for kwarg_key, kwarg_val in kwargs.items():
                        if isinstance(kwarg_val, torch.Tensor):
                            modified_kwargs['l_'+kwarg_key+'_'] = kwarg_val.to(self.device)
                        else:
                            modified_kwargs['l_'+kwarg_key+'_'] = kwarg_val

                    self.node_status = NodeStatus.FORWARD

                    print('Before Root Forward: ')
                    check_gpu_usage()
                    output = self.compute_session.root_forward_compute(tensors, self.forward_pass_id, **modified_kwargs)
                    print('After Root Forward: ')
                    check_gpu_usage()

                    payload = self.comm_session.create_forward_payload(output, tensors=tensors)

                    final_payload = {}
                    final_payload[self.submod_file] = payload

                    sent_data = {'forward_pass_id':self.forward_pass_id,
                                'data': final_payload,
                                'action': ActionTypes.FORWARD}
                    
                    print('Forward compute done for: ', self.forward_pass_id)
                    self.forward_pass_id += 1
                    self.comm_session.trigger_send(sent_data, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
                    self.n_forwards += 1
                    # print('Forward compute done for: ', self.tensor_id)
                    self.root_compute = True
                    self.node_status = NodeStatus.IDLE
                
                elif action == ActionTypes.FORWARD:
                    print('n_backwards in FORWARD: ', self.n_backwards)
                    self.node_status = NodeStatus.FORWARD
                    data = value['data']
                    forward_pass_id = value['forward_pass_id']
                    print('Start of forward: ', forward_pass_id)
                    
                    output = self.compute_session.middle_forward_compute(data, forward_pass_id=forward_pass_id)

                    payload = self.comm_session.create_forward_payload(output)

                    final_payload = data
                    final_payload[self.submod_file] = payload

                    sent_data = {'forward_pass_id':forward_pass_id,
                                'data': final_payload,
                                'action': ActionTypes.FIND_LOSS}
                    t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
                    send_trigger_threads.append(t)
                    t.start()
                    self.n_forwards += 1
                    print('Forward Done Used RAM %: ', psutil.virtual_memory().percent)

                elif action == ActionTypes.FIND_LOSS:
                    self.node_status = NodeStatus.FORWARD
                    epoch_change = False
                    
                    targets = next(self.labels_iterator, None)
                    if targets is None:
                        epoch_change = self.lr_step_on_epoch_change
                        self.labels_iterator = iter(self.labels)
                        targets = next(self.labels_iterator)
                        if epoch_change:
                            if self.lr_scheduler is not None:
                                self.lr_scheduler.step()
                        print('\n ---------------------- Reset Data Iterator ------------------------')

                    # print('For: ', value['forward_pass_id'])
                    # print('X_train: ', targets[0][0][0])
                    # print('y_train: ', targets[1])

                    # targets = targets[1].to(self.device)
                    # targets = targets.to(self.device)    # For BERT

                    update_flag = False
                    if (self.n_backwards + 1) % self.update_frequency == 0:
                        update_flag = True
                    
                    data = value['data']
                    model_args = self.compute_session.leaf_find_loss(data, targets=targets, update_flag=update_flag)
                    if update_flag and not self.lr_step_on_epoch_change:
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                    
                    gradients = self.comm_session.create_backward_payload(model_args=model_args)

                    sent_data = {'action':ActionTypes.BACKWARD, 
                                'data':gradients, 
                                'forward_pass_id':value['forward_pass_id'],
                                'epoch_change':epoch_change
                                }
                    t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.BACKWARD, self.backward_target_host, self.backward_target_port,))
                    send_trigger_threads.append(t)
                    t.start()

                    # print('find_loss done. Used RAM %: ', psutil.virtual_memory().percent)
                    self.n_backwards += 1
                    # print('N_backwards: ', self.n_backwards)

                    if self.n_backwards % self.reduce_threshold == 0:
                        # print('\nPre AVeraged params: ', self.compute_session.model.state_dict()['L__self___bert_encoder_layer_9_output_dense.weight'])#list(self.compute_session.model.state_dict().keys())[0]])

                        self.comm_session.parallel_ring_reduce()
                        # print('\nAVeraged params: ', self.compute_session.model.state_dict()['L__self___bert_encoder_layer_9_output_dense.weight'])#[list(self.compute_session.model.state_dict().keys())[0]])

                    # if self.device.type == 'cuda':
                    #     # print('Sync')
                    #     torch.cuda.synchronize()

                elif action == ActionTypes.NO_GRAD_FORWARD:
                    # self.comm_session.parallel_ring_reduce()
                    self.node_status = NodeStatus.FORWARD
                    print('No grad forward')
                    data = value['data']
                    output = self.compute_session.middle_no_grad_forward_compute(data)
                    payload = self.comm_session.create_no_grad_forward_payload(output)

                    final_payload = data
                    final_payload[self.submod_file] = payload

                    sent_data = {
                                    'data': final_payload,
                                    'action': value['output_type']                                            
                                }
                    t = Thread(target=self.comm_session.trigger_send, args=(sent_data, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
                    send_trigger_threads.append(t)
                    t.start()

                elif action == ActionTypes.VAL_ACCURACY:
                    data = value['data']
                    model_args = self.compute_session.create_no_grad_model_args(data)

                    if self.test_labels_iterator is None:
                        if isinstance(self.test_labels, torch.Tensor):
                            self.test_labels_iterator = self.test_labels
                        else:
                            self.test_labels_iterator = iter(self.test_labels)
                    
                    self.model.eval()
                    with torch.no_grad():
                        y_pred = self.model(*model_args)
                        _, y_pred_tags = torch.max(y_pred, dim=1)
                                                
                        y_test = next(self.test_labels_iterator, None)
                        if y_test is None:
                            self.test_labels_iterator = iter(self.test_labels)
                            y_test = next(self.test_labels_iterator)

                        y_test = y_test[1].to(self.device)

                        #for cnn
                        y_test = torch.argmax(y_test, dim=1)

                        correct_pred = (y_pred_tags == y_test).float()
                        val_acc = correct_pred.sum() / len(y_test)
                        val_acc = torch.round(val_acc * 100)

                    self.net_val_accuracy.append(val_acc.item())
                    if len(self.net_val_accuracy) == len(self.test_labels_iterator):
                        validation_accuracy = round(sum(self.net_val_accuracy) / len(self.net_val_accuracy), 2)
                        print('Validation Accuracy: ', validation_accuracy)
                        f = open("val_accuracies.txt", "a")
                        f.write(str(validation_accuracy) + '\n')
                        f.close() 
                        self.net_val_accuracy = []


                elif action == ActionTypes.ACCURACY:
                    # self.comm_session.parallel_ring_reduce()
                    print('Finding accuracy')
                    data = value['data']
                    model_args = self.compute_session.create_no_grad_model_args(data)
        
                    self.model.eval()
                    with torch.no_grad():
                        y_pred = self.model(*model_args)
                        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=-1)
                        y_test = np.argmax(self.test_labels, axis=-1)
                        accuracy = np.sum(y_pred == y_test, axis=0)/len(y_test)
                        print('\nTest Accuracy: ', accuracy)

                elif action == ActionTypes.PREDICTION:
                    data = value['data']
                    print('Prediction: ', data)
                    model_args = self.create_no_grad_model_args(data)
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(*model_args)
                    print('Predicted: ', pred)

                elif action == ActionTypes.SAVE_SUBMODEL:
                    script = torch.jit.script(self.model)
                    script.save('{}/{}.pt'.format(self.template_path, self.submod_file))
                    os.remove('{}/submod.pt'.format(self.template_path))
                    if self.node_type != NodeTypes.LEAF:
                        t = Thread(target=self.comm_session.trigger_send, args=({'action': ActionTypes.SAVE_SUBMODEL}, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
                        send_trigger_threads.append(t)
                        t.start()                        
                    print('SAVE done')

            if len(send_trigger_threads)>0:
                for send_threads in send_trigger_threads:
                    send_threads.join()
            self.node_status = NodeStatus.IDLE
                        

    def forward_compute(self, tensors=None, **kwargs):
        """Initiate a forward computation request.

        Adds the forward computation request to the load forward buffer,
        ensuring synchronization and handling of computational resources.

        :param tensors: Input tensors for the forward computation, defaults to None
        :type tensors: torch.Tensor, optional
        :param kwargs: Additional keyword arguments for the computation, defaults to {}
        :type kwargs: dict, optional
        """
        data = {'data':tensors, 'kwargs':kwargs, 'action': ActionTypes.ROOT_FORWARD}


        while self.forward_pass_id - self.latest_backward_id > self.cluster_length:
            time.sleep(0)
        
        if self.n_forwards % self.reduce_threshold == 0:
            self.wait_for_backwards()

        self.forward_lock.acquire(block=True)
        self.load_forward_buffer.append(data)
        self.forward_lock.release()

        self.root_compute = False

        while not self.root_compute:
            time.sleep(0)

    def no_grad_forward_compute(self, tensors=None, output_type=None):
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
        
        output = self.compute_session.root_no_grad_forward_compute(tensors=tensors)

        payload = self.comm_session.create_no_grad_forward_payload(output, tensors=tensors)

        final_payload = {}
        final_payload[self.submod_file] = payload

        sent_data = {
                        'data': final_payload,
                        'action': ActionTypes.NO_GRAD_FORWARD,
                        'output_type': output_type
                    }
        self.comm_session.trigger_send(sent_data, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('No Grad forward compute done')
        self.node_status = NodeStatus.IDLE

    def wait_for_backwards(self):
        """Wait until all backward passes are completed.

        Checks and waits until all initiated backward computations are finished
        before proceeding with further operations.

        """
        while self.n_backwards < self.n_forwards:
            time.sleep(1)

    def trigger_save_submodel(self):
        """Trigger saving of the current submodel state.

        Saves the current state of the model to disk and optionally sends
        the updated model state to the designated target host and port.

        """
        script = torch.jit.script(self.model)
        os.makedirs(self.template_path, exist_ok=True)
        script.save('{}/{}.pt'.format(self.template_path,self.submod_file))
        os.remove('{}/submod.pt'.format(self.template_path))
        self.comm_session.trigger_send({'action': ActionTypes.SAVE_SUBMODEL}, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
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


    def __getstate__(self):
        return dict(
            forward_lock = self.forward_lock,
            backward_lock = self.backward_lock,
            reduce_lock = self.reduce_lock,
            gather_lock = self.gather_lock,
            latest_weights_lock = self.latest_weights_lock,
            local_address = self.local_address,
            load_forward_buffer = self.load_forward_buffer,
            load_backward_buffer = self.load_backward_buffer,
            latest_weights_buffer = self.latest_weights_buffer,
            reduce_ring_buffers = self.reduce_ring_buffers,
            gather_ring_buffers = self.gather_ring_buffers,
            reduce_iteration = self.reduce_iteration,
            gather_iteration = self.gather_iteration,
            start_server_flag = self.start_server_flag
        )

    def __setstate__(self, state):
        self.local_address = state['local_address']
        self.start_server_flag = state['start_server_flag']
        self.init_server(load_forward_buffer=state['load_forward_buffer'], 
                         load_backward_buffer=state['load_backward_buffer'], 
                         reduce_ring_buffers= state['reduce_ring_buffers'],
                         gather_ring_buffers= state['gather_ring_buffers'],
                         latest_weights_buffer=state['latest_weights_buffer'],
                         forward_lock=state['forward_lock'], 
                         backward_lock=state['backward_lock'],
                         reduce_lock=state['reduce_lock'],
                         gather_lock=state['gather_lock'],
                         latest_weights_lock = state['latest_weights_lock'],
                         reduce_iteration = state['reduce_iteration'],
                         gather_iteration = state['gather_iteration']
                         )
