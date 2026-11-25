---
paths:
  - 'packages/inference/src/board_reader_inference/recognition/**'
  - 'packages/inference/tests/recognition/test_onnxrt.py'
  - 'packages/inference/tests/recognition/test_chessq.py'
  - 'Makefile'
  - 'tools/stop.sh'
  - 'tools/wtest.sh'
---

# The GPU (one 4 GB card that also drives the display)

- Build every onnxruntime session's providers through `onnxrt.cuda_providers()`, never by hand and never with `LD_LIBRARY_PATH` at the launcher, because it calls `ort.preload_dlls()` (the PyPI onnxruntime-gpu wheels are built for CUDA 12, and on a CUDA 13 system the CUDA provider otherwise falls back to CPU without a word) and caps each session with `gpu_mem_limit` (`BR_GPU_MEM_LIMIT_MB`): the card also holds the display's scanout surface, and when X cannot reallocate it the monitor goes blank.
- Feed each session one input shape for its whole life: pad a partial batch up to the batch size by repeating the last row (not zeros) and slice the outputs back, because the CUDA BFC arena never frees and reserves a new chunk set per distinct shape, so ragged batches grow it until every request fails at the cap.
- Read an onnxruntime "BFCArena failed to allocate" from a test, `eval` or a live node as a full card (a running dev server, another agent, a browser), not a code regression: rerun with `CUDA_VISIBLE_DEVICES=` and check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before you debug the code.
- When the live position stream goes quiet mid-take, check GPU memory first, because under memory pressure every read fails while the worker loop survives, so nothing crashes. Never run a browser (Playwright, Chromium) on this machine during a live take.
- Do not take a freed port as proof that a server exited: uvicorn can wedge behind an in-flight ONNX call, keep its CUDA context and ignore SIGTERM. `make stop` (`tools/stop.sh`) checks that the process is really gone; check `nvidia-smi`, not `ss -ltn`.
- Keep `CUDA_VISIBLE_DEVICES=""` in `make test` and `tools/wtest.sh` (`BR_TEST_GPU=1` opts back in), because `-m "not gpu"` deselects only the tests that build a session on purpose; others build one incidentally, and a dozen xdist workers each opening a CUDA context hang inside the driver instead of failing.
- Keep `OPENCV_OPENCL_RUNTIME=disabled` in `make dev`, because OpenCV's OpenCL backend turns a full card into an uncatchable C++ abort (exit 134), while onnxruntime's OOM is a catchable exception. Do not mirror it in compose.yaml, which declares no GPU device: env parity between the entry points does not mean the same hardware.
- Serve fp32 on CUDA and int8 on CPU (one release, two files, each as its `-attn` copy, chosen per entry point; `tests/test_deploy_parity.py` pins GPU to fp32), because the int8 file's `MatMulInteger` has no CUDA kernels (600+ ms vs ~205 ms fp32 on the GTX 1650), while on CPU int8 is the faster one. Use `make dev-cpu` when the machine running the node also drives your screens.
