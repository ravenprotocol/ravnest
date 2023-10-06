from server_pb2_grpc import CommServer
from tensor_pb2 import SendTensor, SendTensorReply
from server_pb2 import CheckBufferStatus, BufferStatusReply
import multiprocessing as mp
from typing import AsyncIterator
from utils import aiter_with_timeout
import _pickle as cPickle

class GrpcService(CommServer):
    def __init__(self, load_forward_buffer=None, load_backward_buffer=None, forward_lock=None, backward_lock=None):
        super().__init__()
        self.load_forward_buffer = load_forward_buffer
        self.load_backward_buffer = load_backward_buffer
        self.forward_lock = forward_lock
        self.backward_lock = backward_lock
        self.forward_id_queue = []
        self.backward_id_queue = []
        
    def send_buffer(self, request: SendTensor, context) -> SendTensorReply:
       
        size_accumulated_tensor_buffer = 0
        accumulated_tensor_buffer = b''
       
        buffer_type = None
        for data in request:
            buffer_type = data.type
            data = data.tensor_chunk
            buffer = data.buffer
            data_type = data.type
            tensor_size = data.tensor_size

            if size_accumulated_tensor_buffer < tensor_size:
                accumulated_tensor_buffer += buffer
                size_accumulated_tensor_buffer = len(accumulated_tensor_buffer)

        data = cPickle.loads(accumulated_tensor_buffer)

        if buffer_type == 'forward':

            self.forward_lock.acquire(block=True)
            
            self.load_forward_buffer.append(data)
            del self.forward_id_queue[0]
            self.forward_lock.release()

        elif buffer_type == 'backward':

            self.backward_lock.acquire(block=True)
            self.load_backward_buffer.append(data)
            del self.backward_id_queue[0]
            self.backward_lock.release()
            
        return SendTensorReply(reply=True)
        

    def buffer_status(self, request: CheckBufferStatus, context) -> BufferStatusReply:
       
        incoming_node = request.name
        node_type = request.type

        if node_type == 'forward':

            if incoming_node not in self.forward_id_queue:
                self.forward_id_queue.append(incoming_node)
            if len(self.load_forward_buffer) == 0 and self.forward_id_queue[0] == incoming_node:
                return BufferStatusReply(status='send_buffer')

        elif node_type == 'backward':

            if incoming_node not in self.backward_id_queue:
                self.backward_id_queue.append(incoming_node)
            if len(self.load_backward_buffer) == 0 and self.backward_id_queue[0] == incoming_node:
                return BufferStatusReply(status='send_buffer')

        return BufferStatusReply(status='wait')

