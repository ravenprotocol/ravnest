import grpc
from threading import Thread
import psutil
import torch
from contextlib import contextmanager
from .utils import *
from .strings import *
from .protos.server_pb2_grpc import CommServerStub
from .protos.server_pb2 import CheckBufferStatus, CheckReduceIteration, CheckGatherIteration, SendLatestWeights

class Communication():
    def __init__(self, name=None, model=None, optimizer=None, node_type=None, rank=None, ring_size=None, ring_param_keys=None,
                 ring_ids = None, param_address_mapping=None, reduce_lock=None, gather_lock=None, device = torch.device('cpu'),
                 compression=False, forward_target_host=None, forward_target_port=None,
                 backward_target_host=None, backward_target_port=None, retrieve_latest_params_data = None,
                 output_tensors=None, input_tensors=None,
                 reduce_ring_buffers=None, gather_ring_buffers=None,
                 reduce_iteration=None, gather_iteration=None,
                 submod_file=None, tensor_id=None, averaged_params_buffer=None,
                 average_no=None, average_optim=False, output_template=None, model_inputs_template=None):
        
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.node_type = node_type
        self.rank = rank
        self.ring_size = ring_size
        self.ring_param_keys = ring_param_keys
        self.ring_ids = ring_ids
        self.param_address_mapping = param_address_mapping
        self.retrieve_latest_params_data = retrieve_latest_params_data

        self.reduce_lock = reduce_lock
        self.gather_lock = gather_lock

        self.device = device
        self.compression = compression

        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

        self.forward_target_host = forward_target_host
        self.forward_target_port = forward_target_port
        self.backward_target_host = backward_target_host
        self.backward_target_port = backward_target_port

        self.reduce_ring_buffers = reduce_ring_buffers
        self.gather_ring_buffers = gather_ring_buffers

        self.reduce_iteration = reduce_iteration
        self.gather_iteration = gather_iteration

        self.submod_file = submod_file
        self.tensor_id = tensor_id

        self.averaged_params_buffer = averaged_params_buffer
        self.average_no = average_no
        self.average_optim = average_optim

        if self.average_optim:
            self.averaged_optim_params_buffer = {}

        self.output_template = output_template
        
        self.model_inputs_template = model_inputs_template


    # def trigger_send(self, data, type=None, target_host=None, target_port=None):
    #     with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
    #         stub = CommServerStub(channel)

    #         send_flag = False
    #         while not send_flag:
    #             buffer_status = stub.buffer_status(CheckBufferStatus(name=self.name, type=type))
                
    #             if buffer_status.status == BufferStatus.SEND_BUFFER:
    #                 send_flag = True
    #         response = stub.send_buffer(generate_stream(data, type=type))

    @contextmanager
    def comm_channel_context(self, type=None, host=None, port=None):
        if type == ActionTypes.FORWARD:
            with grpc.insecure_channel('{}:{}'.format(self.forward_target_host, self.forward_target_port)) as channel:
                yield channel
        elif type == ActionTypes.BACKWARD:
            with grpc.insecure_channel('{}:{}'.format(self.backward_target_host, self.backward_target_port)) as channel:
                yield channel
        else:
            with grpc.insecure_channel('{}:{}'.format(host, port)) as channel:
                yield channel
    
    def create_backward_payload(self, forward_pass_id=None, model_args=None):        
        grad_payload = {}
        if self.node_type == NodeTypes.LEAF:
            for key, value in model_args.items():
                if value.requires_grad:
                    original_dtype = value.dtype
                    grad_payload[key] = {'dtype': original_dtype, 'data': value.grad.detach().clone().to(torch.device('cpu'))} #value.grad.to(torch.device('cpu'))
                    if self.compression:
                        grad_payload[key]['data'] = compress_tensor_float16(grad_payload[key]['data'])
        else:
            for key, value in self.input_tensors[forward_pass_id].items():
                if value.requires_grad:
                    # grad_payload[key] = value.grad.to(torch.device('cpu'))
                    original_dtype = value.dtype
                    grad_payload[key] = {'dtype': original_dtype, 'data': value.grad.detach().clone().to(torch.device('cpu'))} #value.grad.to(torch.device('cpu'))
                    if self.compression:
                        grad_payload[key]['data'] = compress_tensor_float16(grad_payload[key]['data'])
        return grad_payload

    def create_forward_payload(self, output, tensors=None):
        payload = self.output_template.copy()
        for k, v in payload.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            
            # payload[k]['data'] = out.to(torch.device('cpu'))
            payload[k]['dtype'] = out.dtype
            payload[k]['data'] = out.detach().clone().to(torch.device('cpu'))

            if self.compression:
                payload[k]['data'] = compress_tensor_float16(payload[k]['data'])

            payload[k]['tensor_id'] = self.tensor_id
            # if self.node_type != NodeTypes.ROOT:
            #     self.output_tensors[self.tensor_id] = out
            self.tensor_id = str(int(self.tensor_id.split('_')[0]) + 1) + '_{}'.format(self.submod_file)

        if self.node_type == NodeTypes.ROOT:
            payload['model_inputs'] = self.model_inputs_template.copy()
            for k, v in self.model_inputs_template.items():
                if payload['model_inputs'][k].get('target', None) is not None:
                    payload['model_inputs'][k]['data'] = tensors[k]
        return payload

    def parallel_ring_reduce(self):
        if self.ring_size > 1:
            # print('\nBegining Parameter Averaging')
            threads = []
            for ring_id, _ in self.ring_ids.items():
                # ring_data = {k:self.model.state_dict()[k] for k in self.ring_param_keys[ring_id]}
                if self.average_optim:
                    ring_data = {}
                    optim_state = dict(self.optimizer.state)
                    optim_ring_data = {}
                    for (param_name, model_param), optim_param in zip(self.model.named_parameters(), optimizer_params(self.optimizer)):
                        if param_name in self.ring_param_keys[ring_id]:
                            ring_data[param_name] = model_param
                            optim_ring_data[param_name] = optim_state[optim_param]
                else:
                    ring_data = ring_data = {k:self.model.state_dict()[k] for k in self.ring_param_keys[ring_id]}
                    optim_ring_data = {}

                t = Thread(target=self.single_ring_reduce, args=(ring_data, optim_ring_data, ring_id,))
                threads.append(t)
                t.start()

            for thread in threads:
                thread.join()

            load_state_dict_conserve_versions(self.model, self.averaged_params_buffer)
            self.averaged_params_buffer = {}
            load_model_weights_into_optim(self.model, self.optimizer)
            if self.average_optim:
                load_optim_state(self.optimizer, self.averaged_optim_params_buffer, self.model)
                self.averaged_optim_params_buffer = {}
            self.average_no += 1 
            # print('\nParameter Averaging Complete: ', self.average_no, ' Used RAM %: ', psutil.virtual_memory().percent)
            # print('Averaged state_dict: ', self.model.state_dict())

    @torch.no_grad()
    def single_ring_reduce(self, ring_data, optim_ring_data, ring_id):
        chunked_data = create_chunks(data=ring_data, size=self.ring_size)
        if self.average_optim:
            chunked_optim_data = create_chunks_optim(data=optim_ring_data, size=self.ring_size)
        iterations = self.ring_size - 1
        recv_pos = ((self.rank-1)+self.ring_size) % self.ring_size 
        send_pos = (self.rank)%self.ring_size

        for i in range(iterations):
            # print('Send pos: ', send_pos)
            # print(' RIng size: ', self.ring_size)
            address_send_data_mapping = {}
            for id, chunks in chunked_data.items():
                dest = self.param_address_mapping[id]

                optim_chunk = {}
                if self.average_optim:
                    for state_param, state in chunked_optim_data[id].items():
                        optim_chunk[state_param] = state['data'][send_pos]

                if dest in address_send_data_mapping:
                    address_send_data_mapping[dest][id] = {'pos':send_pos, 'chunk':chunks['data'][send_pos], 'optim_chunk':optim_chunk}
                else:
                    address_send_data_mapping[dest] = {id:{'pos':send_pos, 'chunk': chunks['data'][send_pos], 'optim_chunk':optim_chunk}}

            send_threads = []
            for address, data_dict in address_send_data_mapping.items():
                t = Thread(target=self.send_reduce_chunk, args=(address.split(':')[0], address.split(':')[1], data_dict, ring_id,))
                send_threads.append(t)
                t.start()

            for t in send_threads:
                t.join()
            send_threads = []
            keys_received = 0
            while keys_received < len(chunked_data):
                with self.reduce_lock:
                    received_data = self.reduce_ring_buffers.get(ring_id, None)
                    if received_data is not None and len(received_data)>0:
                        recv_chunk = received_data.pop(0)
                        self.reduce_ring_buffers[ring_id] = received_data
                        for param_index, chunk_dict in recv_chunk.items():
                            chunked_data[param_index]['data'][chunk_dict['pos']] = chunked_data[param_index]['data'][chunk_dict['pos']].add(chunk_dict['chunk'][:])
                            if self.average_optim:
                                for optim_state_name, optim_state_param in chunk_dict['optim_chunk'].items():
                                    chunked_optim_data[param_index][optim_state_name]['data'][chunk_dict['pos']] = chunked_optim_data[param_index][optim_state_name]['data'][chunk_dict['pos']].add(optim_state_param[:])
                            keys_received += 1
            
            recv_pos = ((recv_pos - 1)+self.ring_size)%self.ring_size
            send_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
            self.reduce_iteration[ring_id] += 1

        self.reduce_iteration[ring_id] = 0
        # print('Reduced Ring id: ', ring_id)

        send_pos = (recv_pos+1)%self.ring_size
        recv_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
        for i in range(iterations):
            address_send_data_mapping = {}
            for id, chunks in chunked_data.items():
                dest = self.param_address_mapping[id]

                optim_chunk = {}
                if self.average_optim:
                    for state_param, state in chunked_optim_data[id].items():
                        optim_chunk[state_param] = state['data'][send_pos]

                if dest in address_send_data_mapping:
                    address_send_data_mapping[dest][id] = {'pos':send_pos, 'chunk':chunks['data'][send_pos], 'optim_chunk':optim_chunk}
                else:
                    address_send_data_mapping[dest] = {id:{'pos':send_pos, 'chunk': chunks['data'][send_pos], 'optim_chunk':optim_chunk}}

            send_threads = []
            for address, data_dict in address_send_data_mapping.items():
                t = Thread(target=self.send_gather_chunk, args=(address.split(':')[0], address.split(':')[1], data_dict, ring_id,))
                send_threads.append(t)
                t.start()

            for t in send_threads:
                t.join()
                
            keys_received = 0
            while keys_received < len(chunked_data):
                with self.gather_lock:
                    received_data = self.gather_ring_buffers.get(ring_id, None)
                    if received_data is not None and len(received_data)>0:
                        recv_chunk = received_data.pop(0)
                        self.gather_ring_buffers[ring_id] = received_data
                        
                        for param_index, chunk_dict in recv_chunk.items():
                            chunked_data[param_index]['data'][chunk_dict['pos']].data = chunk_dict['chunk'].data[:]

                            if self.average_optim:
                                for optim_state_name, optim_state_param in chunk_dict['optim_chunk'].items():
                                    chunked_optim_data[param_index][optim_state_name]['data'][chunk_dict['pos']].data = optim_state_param.data[:]

                            keys_received += 1

            recv_pos = ((recv_pos - 1)+self.ring_size)%self.ring_size
            send_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
            self.gather_iteration[ring_id] += 1

        self.gather_iteration[ring_id] = 0

        for param, chunk in chunked_data.items():
            chunked_data[param] = torch.cat(chunk['data'], dim=chunk['split_axis']).div(self.ring_size).to(self.device)
            if self.average_optim:
                for state_param, state in chunked_optim_data[param].items():
                    reduced_tensor = torch.cat(state['data'], dim=state['split_axis']).div(self.ring_size)
                    if state['reshape']:
                        reduced_tensor = reduced_tensor.reshape(())
                    chunked_optim_data[param][state_param] = reduced_tensor.to(self.device)#torch.cat(state['data'], dim=state['split_axis']).div(self.ring_size).to(self.device)

        # print('Gathered Ring id: ', ring_id)
        self.averaged_params_buffer.update(chunked_data)
        if self.average_optim:
            self.averaged_optim_params_buffer.update(chunked_optim_data)

    def get_latest_weights(self):
        total_state_dict = {}
        threads = []
        for address, param_names in self.retrieve_latest_params_data.items():
            t = Thread(target=self.send_latest_weights_request, args=(address.split(':')[0], address.split(':')[1], param_names, total_state_dict))
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()

        return total_state_dict

    def send_reduce_chunk(self, target_host, target_port, data_dict, ring_id):
        # with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
        with self.comm_channel_context(host=target_host, port=target_port) as channel:
            stub = CommServerStub(channel)
            while True:
                iteration_resp = stub.reduce_iteration(CheckReduceIteration(ring_id=ring_id))
                if iteration_resp.iteration == self.reduce_iteration[ring_id]:
                    break
            response = stub.reduce_chunk(generate_data_stream(data_dict, ring_id=ring_id))

    def send_gather_chunk(self, target_host, target_port, data_dict, ring_id):
        # with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
        with self.comm_channel_context(host=target_host, port=target_port) as channel:
            stub = CommServerStub(channel)
            while True:
                iteration_resp = stub.gather_iteration(CheckGatherIteration(ring_id=ring_id))
                if iteration_resp.iteration == self.gather_iteration[ring_id]:
                    break
            response = stub.gather_chunk(generate_data_stream(data_dict, ring_id=ring_id))

    def send_latest_weights_request(self, target_host, target_port, param_names, total_state_dict):
        # with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
        with self.comm_channel_context(host=target_host, port=target_port) as channel:
            stub = CommServerStub(channel)
            param_names = cPickle.dumps(param_names)
            response = stub.get_latest_weights(SendLatestWeights(param_names=param_names))

            size_accumulated_tensor_buffer = 0
            accumulated_tensor_buffer = b''
        
            for data in response:
                data = data.tensor_chunk
                tensor_size = data.tensor_size
                buffer = data.buffer

                if size_accumulated_tensor_buffer < tensor_size:
                    accumulated_tensor_buffer += buffer
                    size_accumulated_tensor_buffer = len(accumulated_tensor_buffer)

        data = cPickle.loads(accumulated_tensor_buffer)
        total_state_dict.update(data)
        # return data


    def create_no_grad_forward_payload(self, output, tensors=None):
        payload = self.output_template.copy()
        for k, v in payload.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            payload[k]['data'] = out.to(torch.device('cpu'))
            
        if self.node_type == NodeTypes.ROOT:
            payload['model_inputs'] = self.model_inputs_template.copy() #Changed to copy
            for k, v in self.model_inputs_template.items():
                if payload['model_inputs'][k].get('target', None) is not None:
                    payload['model_inputs'][k]['data'] = tensors[k]

        return payload