"""
Background heartbeat sender.

HeartbeatSender runs as a daemon thread, periodically calling the registry
to confirm the node is alive and reporting its current resource utilisation.
"""

import logging
import threading

import psutil

logger = logging.getLogger(__name__)


def _sample_load() -> dict:
    """Collect current CPU, RAM, and (if available) GPU utilisation."""
    load = {
        "cpu_percent":      psutil.cpu_percent(interval=None),
        "ram_percent":      psutil.virtual_memory().percent,
        "gpu_percent":      0.0,
        "gpu_vram_percent": 0.0,
    }
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        mem    = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        util   = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        load["gpu_percent"]      = float(util.gpu)
        load["gpu_vram_percent"] = 100.0 * mem.used / mem.total
        nvidia_smi.nvmlShutdown()
    except Exception:
        pass
    return load


class HeartbeatSender:
    """
    Sends periodic heartbeats from a node to the registry server.

    Args:
        registry_client: A RegistryClient connected to the registry.
        node_id:         ID of the node sending heartbeats.
        interval:        Seconds between heartbeats (default 10 s).
    """

    def __init__(self, registry_client, node_id: str, interval: float = 10.0):
        self._client   = registry_client
        self._node_id  = node_id
        self._interval = interval
        self._stop     = threading.Event()
        self._thread: threading.Thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"heartbeat-{self._node_id}",
        )
        self._thread.start()
        logger.info("HeartbeatSender started for %s (interval=%ss)", self._node_id, self._interval)

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                load = _sample_load()
                ok   = self._client.heartbeat(self._node_id, load)
                if not ok:
                    logger.warning(
                        "Heartbeat rejected by registry for %s — node may not be registered",
                        self._node_id,
                    )
            except Exception as exc:
                logger.warning("Heartbeat send failed for %s: %s", self._node_id, exc)
