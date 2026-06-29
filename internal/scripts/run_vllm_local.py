import os
import sys

def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_path = "/home/mohamad/LLM-end-to-end-Service-main/Qwen2.5-0.5B-Instruct"
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    log_path = "/tmp/vllm.log"
    log_f = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = log_f
    sys.stderr = log_f

    argv = [
        "vllm",
        "serve",
        model_path,
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.80",
        "--max-model-len", "2048",
        "--host", "0.0.0.0",
        "--port", "8001",
    ]

    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = argv
    return vllm_main()

if __name__ == "__main__":
    raise SystemExit(main())

