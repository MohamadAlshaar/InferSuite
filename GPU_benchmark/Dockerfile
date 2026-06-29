# Minimal, self-contained image for the GPU sweep load-generator.
#
# The sweep only sends HTTP to vLLM and polls Prometheus endpoints, so it needs
# nothing heavy — just python + requests. (transformers is optional, for exact
# prompt tokenisation; without it prompts use a char heuristic and we bin on the
# token counts vLLM actually reports, so it is not required.)
#
# Build & push (from repo root):
#   ACCOUNT=524558748675; REGION=eu-west-2
#   REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/llm-service/gpu-benchmark
#   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
#   docker build -t $REPO:latest GPU_benchmark
#   docker push $REPO:latest
FROM python:3.12-slim

WORKDIR /app/GPU_benchmark
RUN pip install --no-cache-dir requests

# Only the code + config + pre-generated ragbench prompts — never results/.
COPY *.py config.json ./
COPY prompts ./prompts

# The pod overrides this to `sleep infinity` so you can exec in and run sweeps,
# but the default makes the image runnable standalone too.
ENTRYPOINT ["python3"]
CMD ["run_sweeps.py"]
