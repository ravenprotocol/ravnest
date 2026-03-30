# TODOS

## P2: Add Gloo warmup handshakes for Docker networking validation

**What:** Add isend/irecv warmup exchange at startup for Gloo backend (similar to NCCL warmup at communication_torch.py:163-165).

**Why:** Currently NCCL gets warmup handshakes that validate networking works, but Gloo skips this entirely. With Docker containers, Gloo networking failures only surface on the first inference request instead of at startup. A warmup would give early "Gloo warmup successful" confirmation in container logs or crash immediately if networking is broken.

**Context:** The NCCL warmup path calls `create_cuda_streams()` and `nccl_warmup_groups()` which do isend/irecv/broadcast handshakes. For Gloo, add a simpler version: one isend/irecv exchange between adjacent ranks after `prepare_process_groups()`.

**Effort:** S (human: ~1hr / CC: ~10min)
**Depends on:** Nothing
**Added by:** /plan-ceo-review + /plan-eng-review on 2026-03-30
