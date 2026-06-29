import os
import torch

def main():
    # Optional: target the first GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("free/total GiB (before):",
              tuple(round(x / (1024**3), 2) for x in torch.cuda.mem_get_info()))

    from vllm import LLM, SamplingParams

    model = "Qwen/Qwen2.5-0.5B-Instruct"

    llm = LLM(
        model=model,
        dtype="float16",                #
        gpu_memory_utilization=0.90,    # lower than 0.90 
        max_model_len=2048,             # keep KV cache small-ish for 8GB
    )

    prompts = [
        "Write one sentence explaining what vLLM is.",
        "Give 3 bullet points about KV cache."
    ]

    params = SamplingParams(temperature=0.2, max_tokens=128)

    outputs = llm.generate(prompts, params)

    for o in outputs:
        print("\nPROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)

    if torch.cuda.is_available():
        print("\nfree/total GiB (after):",
              tuple(round(x / (1024**3), 2) for x in torch.cuda.mem_get_info()))

if __name__ == "__main__":
    main()
