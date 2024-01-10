import multiprocessing as mp
import os
import threading
import uuid
import concurrent.futures._base as base
from concurrent.futures._base import Future
import weakref
import torch
import asyncio
import _pickle as cPickle
from typing import TypeVar, AsyncIterable, Optional, AsyncIterator
from protos.tensor_pb2 import TensorChunk, SendTensor
from protos.server_pb2 import ReduceChunk, DataChunk

T = TypeVar("T")

async def aiter_with_timeout(iterable: AsyncIterable[T], timeout: Optional[float]) -> AsyncIterator[T]:
    """Iterate over an async iterable, raise TimeoutError if another portion of data does not arrive within timeout"""
    # based on https://stackoverflow.com/a/50245879
    iterator = iterable.aiter()
    while True:
        try:
            yield await asyncio.wait_for(iterator.anext(), timeout=timeout)
        except StopAsyncIteration:
            break


def generate_stream(data, type=None):
    if not isinstance(data, bytes):
        obj = cPickle.dumps(data)
    else:
        obj = data
    
    file_size = len(obj)
    blocksize = 2*1024*1024
    if file_size > blocksize:
        for i in range(0, file_size, blocksize):
            data = obj[i:i+blocksize]
            tensor_chunk = TensorChunk(buffer=data, type='$tensor', tensor_size=file_size)
            yield SendTensor(tensor_chunk=tensor_chunk, type=type)
        
    else:
        tensor_chunk = TensorChunk(buffer=obj, type='$tensor', tensor_size=file_size)
        yield SendTensor(tensor_chunk=tensor_chunk, type=type)
        
def generate_data_stream(data, ring_id=None, type=None):
    if not isinstance(data, bytes):
        obj = cPickle.dumps(data)
    else:
        obj = data
    
    file_size = len(obj)
    blocksize = 2*1024*1024
    if file_size > blocksize:
        for i in range(0, file_size, blocksize):
            data = obj[i:i+blocksize]
            data_chunk = DataChunk(buffer=data, data_size=file_size)
            yield ReduceChunk(ring_id=ring_id, data_chunk=data_chunk)
        
    else:
        data_chunk = DataChunk(buffer=obj, data_size=file_size)
        yield ReduceChunk(ring_id=ring_id, data_chunk=data_chunk)

@torch.no_grad()
def current_model_params_clone(model):
    for param in model.parameters():
        yield param.clone()

@torch.no_grad()
def optimizer_params(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            yield param

@torch.no_grad()
def load_grads_into_optimizer(model, optimizer):
    for model_param, optimizer_param in zip(model.parameters(), optimizer_params(optimizer)):
        optimizer_param.grad = model_param.grad.clone()

@torch.no_grad()
def load_optim_weights_into_model(model, optimizer):
    for model_param, optimizer_param in zip(model.parameters(), optimizer_params(optimizer)):
        model_param.data = optimizer_param.data

class TestFuture(Future):
    '''Based on hivemind future'''
    active_pid = None
    global_send_pipe = None
    future_comm_thread = None
    current_futures = None
    init_lock = mp.Lock()
    update_lock = mp.Lock()

    def __init__(self) -> None:
        self.init_future()
        self.uid, self.origin_pid = uuid.uuid4().int, os.getpid()
        Future.__init__(self)
        self._result, self._exception = None, None
        TestFuture.current_futures[self.uid] = weakref.ref(self)
        self.send_pipe = TestFuture.global_send_pipe
        # self._aio_event = asyncio.Event()
        # self._aio_event.set()

        try:
            self._loop = asyncio.get_event_loop()
            self._aio_event = asyncio.Event()
        except RuntimeError:
            self._loop, self._aio_event = None, None

        self._set_event_threadsafe()
        
    def _set_event_threadsafe(self):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        async def _event_setter():
            self._aio_event.set()

        if self._loop.is_closed():
            return  # do nothing, the loop is already closed
        elif self._loop.is_running() and running_loop == self._loop:
            asyncio.create_task(_event_setter())
        elif self._loop.is_running() and running_loop != self._loop:
            asyncio.run_coroutine_threadsafe(_event_setter(), self._loop)
        else:
            self._loop.run_until_complete(_event_setter())

    @classmethod
    def init_future(cls):
        pid = os.getpid()
        if pid != TestFuture.active_pid:
            with TestFuture.init_lock:
                if pid != TestFuture.active_pid:
                    receiver_pipe, cls.global_send_pipe = mp.Pipe(duplex=False)
                    cls.active_pid, cls.current_futures = pid, {}
                    cls.future_comm_thread = threading.Thread(target=cls.future_comm_fn, args=(receiver_pipe,), daemon=True)
                    cls.future_comm_thread.start()

    @classmethod
    def future_comm_fn(cls, receiver_pipe):
        while True:
            uid, action, payload = receiver_pipe.recv()
            if action == 'result':
                current_future = cls.current_futures.pop(uid)()
                current_future.set_result(payload)

    def send_comm(self, action, payload):
        with TestFuture.update_lock:
            self.send_pipe.send([self.uid, action, payload])

    def set_result(self, result):
        if os.getpid() == self.origin_pid:
            super().set_result(result)
            TestFuture.current_futures.pop(self.uid, None)
        else:
            self.send_comm('result', result)

    def result(self, timeout=None):
        return super().result(timeout)
    
    def __await__(self):
        if not self._aio_event:
            raise RuntimeError("Can't await: MPFuture was created with no event loop")
        yield from self._aio_event.wait().__await__()
        try:
            return super().result()
        except base.CancelledError:
            raise asyncio.CancelledError()
        
    def __getstate__(self):
        return dict(
            send_pipe=self.send_pipe,
            origin_pid=self.origin_pid,
            uid=self.uid,
            _result=self._result,
            _exception=self._exception,
        )

    def __setstate__(self, state):
        self.send_pipe = state["send_pipe"]
        self.uid = state["uid"]
        self.origin_pid = state['origin_pid']
        self._result, self._exception = state["_result"], state["_exception"]

        self._waiters, self._done_callbacks = [], []
        self._condition = threading.Condition()
        self._aio_event = None