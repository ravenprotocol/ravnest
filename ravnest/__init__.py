# Heavy framework imports require PyTorch + gRPC.  Guard them so that the
# lightweight sub-packages (registry, compute) can be imported standalone
# without a full Ravnest + PyTorch environment.
try:
    from .operations import *
    from .communication import *
    from .endpoints import *
    from .node_tcp import *
    from .inference_engine import InferenceEngine
    from .strings import *
    from .trainer import *
    from .utils import *
except (ImportError, ModuleNotFoundError):
    # PyTorch / gRPC / other heavy deps not installed in this environment.
    # Registry and compute backends remain fully usable.
    pass

# ── Registry ─────────────────────────────────────────────────────────────────
# Requires grpcio; skipped gracefully if not installed.
try:
    from .registry import (
        NodeType,
        ComputeSubtype,
        AgentSubtype,
        DataSubtype,
        ResourceSpec,
        NodeCapability,
        RegistryClient,
        HeartbeatSender,
        NodeRegistry,
        serve_registry,
    )
except (ImportError, ModuleNotFoundError):
    pass

# ── Compute backends & router ─────────────────────────────────────────────────
from .compute import (
    # data classes
    Message,
    GenerateRequest,
    GenerateResponse,
    EmbedRequest,
    EmbedResponse,
    HealthStatus,
    ComputeCapability,
    # abstract base
    ComputeBackend,
    # concrete backends
    RavnestBackend,
    VLLMBackend,
    SGLangBackend,
    OllamaBackend,
    OpenAICompatBackend,
    # router
    ComputeRouter,
    RoutingStrategy,
    LoadBasedStrategy,
    RoundRobinStrategy,
    ModelMatchStrategy,
)