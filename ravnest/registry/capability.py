import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class NodeType(str, Enum):
    """Top-level role of a node in the decentralized network."""
    PIPELINE_COMPUTE   = "pipeline_compute"    # ravnest ROOT/STEM/LEAF stage
    STANDALONE_COMPUTE = "standalone_compute"   # full-model server: vllm, sglang, ollama, …
    AGENT              = "agent"                # LLM-wrapped agent with tool use
    DATA_SOURCE        = "data_source"          # text corpus, image store, vector/graph DB
    ORCHESTRATOR       = "orchestrator"         # request router / API gateway (Phase 6)


class ComputeSubtype(str, Enum):
    RAVNEST       = "ravnest"
    VLLM          = "vllm"
    SGLANG        = "sglang"
    OLLAMA        = "ollama"
    OPENAI_COMPAT = "openai_compat"


class AgentSubtype(str, Enum):
    RESEARCH = "research"
    SQL      = "sql"
    RAG      = "rag"
    GENERIC  = "generic"


class DataSubtype(str, Enum):
    TEXT      = "text"
    IMAGE     = "image"
    VECTOR_DB = "vector_db"
    GRAPH_DB  = "graph_db"


@dataclass
class ResourceSpec:
    ram_mb:         int   = 0
    vram_mb:        int   = 0
    bandwidth_mbps: float = 0.0
    disk_gb:        int   = 0
    cpu_cores:      int   = 0

    @classmethod
    def from_system(cls) -> "ResourceSpec":
        """Auto-detect the local machine's resources."""
        import psutil
        vm   = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        ram_mb    = int(vm.total / 1024 / 1024)
        disk_gb   = int(disk.total / 1024 / 1024 / 1024)
        cpu_cores = psutil.cpu_count(logical=False) or 1

        vram_mb = 0
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle  = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info    = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            vram_mb = int(info.total / 1024 / 1024)
            nvidia_smi.nvmlShutdown()
        except Exception:
            pass

        return cls(ram_mb=ram_mb, vram_mb=vram_mb, disk_gb=disk_gb, cpu_cores=cpu_cores)


@dataclass
class NodeCapability:
    """Complete description of a node's identity, resources, and live state."""
    node_id:        str
    node_type:      NodeType
    subtype:        str           # value from one of the *Subtype enums above
    address:        str           # host:port of this node's own gRPC server
    resources:      ResourceSpec  = field(default_factory=ResourceSpec)
    models:         List[str]     = field(default_factory=list)   # for compute/agent nodes
    metadata:       Dict          = field(default_factory=dict)
    registered_at:  float         = field(default_factory=time.time)
    last_heartbeat: float         = field(default_factory=time.time)
    current_load:   Dict          = field(default_factory=dict)   # keys: cpu/ram/gpu_percent

    def is_alive(self, ttl_seconds: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < ttl_seconds
