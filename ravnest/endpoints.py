from .protos.server_pb2_grpc import CommServer
from .protos.server_pb2 import CheckBufferStatus, BufferStatusReply, ReceivedChunk, ReduceChunk, GatherChunk, CheckReduceIteration, CheckGatherIteration, ReduceIterationReply, GatherIterationReply
from .protos.tensor_pb2 import SendTensor, SendTensorReply

import _pickle as cPickle

class GrpcService(CommServer):
    def __init__(self, load_forward_buffer=None, 
                 load_backward_buffer=None, 
                 forward_lock=None, backward_lock=None, 
                 reduce_ring_buffers = None, gather_ring_buffers=None,
                 reduce_lock=None, gather_lock=None, 
                 reduce_iteration = None, gather_iteration = None
                 ):
        super().__init__()
        self.load_forward_buffer = load_forward_buffer
        self.load_backward_buffer = load_backward_buffer
        self.forward_lock = forward_lock
        self.backward_lock = backward_lock
        self.reduce_lock = reduce_lock
        self.gather_lock = gather_lock
        self.forward_id_queue = []
        self.backward_id_queue = []
        self.reduce_ring_buffers = reduce_ring_buffers
        self.gather_ring_buffers = gather_ring_buffers
        self.reduce_iteration_manager = reduce_iteration
        self.gather_iteration_manager = gather_iteration
        
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
    
    def reduce_iteration(self, request:CheckReduceIteration, context) -> ReduceIterationReply:
        return ReduceIterationReply(iteration=self.reduce_iteration_manager[request.ring_id])
    
    def gather_iteration(self, request:CheckGatherIteration, context) -> GatherIterationReply:
        return GatherIterationReply(iteration=self.gather_iteration_manager[request.ring_id])

    def reduce_chunk(self, request:ReduceChunk, context) -> ReceivedChunk:
        size_accumulated_data_buffer = 0
        accumulated_data_buffer = b''
        for data in request:
            ring_id = data.ring_id
            data = data.data_chunk
            buffer = data.buffer
            data_size = data.data_size

            if size_accumulated_data_buffer < data_size:
                accumulated_data_buffer += buffer
                size_accumulated_data_buffer = len(accumulated_data_buffer)
            
        data = cPickle.loads(accumulated_data_buffer)
        with self.reduce_lock:
            if ring_id in self.reduce_ring_buffers:
                ring_buffer_data = self.reduce_ring_buffers[ring_id]
                ring_buffer_data.append(data)
                self.reduce_ring_buffers[ring_id] = ring_buffer_data #.append(data)
            else:
                self.reduce_ring_buffers[ring_id] = [data]

        return ReceivedChunk(reply=True)
    
    def gather_chunk(self, request:GatherChunk, context) -> ReceivedChunk:
        size_accumulated_data_buffer = 0
        accumulated_data_buffer = b''
        for data in request:
            ring_id = data.ring_id
            data = data.data_chunk
            buffer = data.buffer
            data_size = data.data_size

            if size_accumulated_data_buffer < data_size:
                accumulated_data_buffer += buffer
                size_accumulated_data_buffer = len(accumulated_data_buffer)
            
        data = cPickle.loads(accumulated_data_buffer)
        with self.gather_lock:
            if ring_id in self.gather_ring_buffers:
                ring_buffer_data = self.gather_ring_buffers[ring_id]
                ring_buffer_data.append(data)
                self.gather_ring_buffers[ring_id] = ring_buffer_data
            else:
                self.gather_ring_buffers[ring_id] = [data]

        return ReceivedChunk(reply=True)