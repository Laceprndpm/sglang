# NCCL EP regression fixtures

These fixtures support the registered tests for NCCL EP LL, CUDA Graph
ownership, Triton expert compute, zero-token ranks and serial shared experts.

Run from the repository root with SGLang and pytest installed:

```bash
PYTHONPATH=python:test python -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_*.py
```

The CUDA tests use one GPU; SM89 is sufficient. They exercise production
Graphs, dispatchers, runner input buffers and expert kernels with a narrow
one-rank replacement for the external EP bindings. They do not validate native
NCCL EP communication. The oracle and prefill fallback tests also run on CPU.

The fixtures retain independent routing/output oracles, temporary tensor
lifetime checks, capture/replay ordering, shutdown/recapture failures, empty
ranks and persistent resource ownership. Each registered test file exposes
the repository CI entrypoint.

The full-model benchmarks, native multi-rank experiments, Nsight analysis,
rental-session drivers and their own unit tests are maintained separately.
Their [original versions](https://github.com/Laceprndpm/sglang/tree/abec649bf47333da7d343d38ddca29b51eab9635/test/nccl_ep_test)
remain available with their historical validation results. Those results name
the tested revision; they do not establish a new native pass for this checkout.
