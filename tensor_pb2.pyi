from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SendTensor(_message.Message):
    __slots__ = ["tensor_chunk", "type"]
    TENSOR_CHUNK_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    tensor_chunk: TensorChunk
    type: str
    def __init__(self, tensor_chunk: _Optional[_Union[TensorChunk, _Mapping]] = ..., type: _Optional[str] = ...) -> None: ...

class SendTensorReply(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: bool
    def __init__(self, reply: bool = ...) -> None: ...

class TensorChunk(_message.Message):
    __slots__ = ["buffer", "tensor_size", "type"]
    BUFFER_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SIZE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    buffer: bytes
    tensor_size: int
    type: str
    def __init__(self, buffer: _Optional[bytes] = ..., type: _Optional[str] = ..., tensor_size: _Optional[int] = ...) -> None: ...
