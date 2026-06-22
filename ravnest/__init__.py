# Heavy framework imports require PyTorch + gRPC.  Guard them so that the
# lightweight sub-packages (registry, compute) can be imported standalone
# without a full Ravnest + PyTorch environment.
try:
    from .operations import *
    from .communication import *
    from .endpoints import *
    from .node_tcp import *
    from .inference import InferenceEngine
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

# ── Agent nodes & router ──────────────────────────────────────────────────────
from .agents import (
    # data classes
    AgentRequest,
    AgentResponse,
    AgentHealthStatus,
    AgentCapability,
    ToolCall,
    ToolResult,
    # abstract base
    AgentBackend,
    # concrete agents
    LiteLLMAgent,
    ResearchAgent,
    SQLAgent,
    # router
    AgentRouter,
    AgentRoutingStrategy,
    LoadBasedAgentStrategy,
    RoundRobinAgentStrategy,
    AgentTypeStrategy,
)

# ── Mesh — inter-node communication & pipeline ───────────────────────────────
from .mesh import (
    NodeMessage,
    NodeResponse,
    PipelineStep,
    PipelineResult,
    NodeServer,
    NodeClient,
    Pipeline,
)

# ── Data source nodes & router ────────────────────────────────────────────────
from .data_sources import (
    # data classes
    DataChunk,
    DataRequest,
    DataResponse,
    DataSourceCapability,
    DataSourceHealthStatus,
    # abstract base
    DataSourceBackend,
    # concrete backends
    TextSource,
    ImageSource,
    VectorDBSource,
    GraphDBSource,
    # router
    DataRouter,
    DataRoutingStrategy,
    LoadBasedDataStrategy,
    RoundRobinDataStrategy,
    SourceTypeStrategy,
)