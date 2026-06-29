from vllm import LLM, SamplingParams

MODEL_PATH = "/home/mohamad/LLM-end-to-end-Service-main/Qwen2.5-0.5B-Instruct"

llm = LLM(MODEL_PATH, dtype="float16", gpu_memory_utilization=0.80, max_model_len=2048)
params = SamplingParams(max_tokens=128, temperature=0.7)

out = llm.generate(["Tell me in 2 sentences what Huawei does."], params)
print(out[0].outputs[0].text.strip())
