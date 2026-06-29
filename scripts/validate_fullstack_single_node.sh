#!/usr/bin/env bash
set -euo pipefail

LLMD_NAMESPACE="${LLMD_NAMESPACE:-llm-d-local}"
FASTAPI_NAMESPACE="${FASTAPI_NAMESPACE:-llm-service}"

LLMD_SERVICE="${LLMD_SERVICE:-infra-local-inference-gateway-istio}"
FASTAPI_SERVICE="${FASTAPI_SERVICE:-llm-service-kernel}"

LLMD_LOCAL_PORT="${LLMD_LOCAL_PORT:-18080}"
FASTAPI_LOCAL_PORT="${FASTAPI_LOCAL_PORT:-18081}"

LLMD_BASE_URL="${LLMD_BASE_URL:-http://127.0.0.1:${LLMD_LOCAL_PORT}}"
FASTAPI_BASE_URL="${FASTAPI_BASE_URL:-http://127.0.0.1:${FASTAPI_LOCAL_PORT}}"

MODEL_NAME="${MODEL_NAME:-qwen2.5-0.5b}"
LLMD_API_MODE="${LLMD_API_MODE:-completions}"

AUTO_PORT_FORWARD="${AUTO_PORT_FORWARD:-1}"
HTTP_TIMEOUT_S="${HTTP_TIMEOUT_S:-30}"
RAG_TEST_PROMPT="${RAG_TEST_PROMPT:-What does tenantA say about StarNUMA?}"

_MANAGED_PIDS=()

log() {
  printf '[validate_fullstack_single_node] %s\n' "$*"
}

die() {
  printf '[validate_fullstack_single_node] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

cleanup() {
  local pid
  for pid in "${_MANAGED_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

is_local_url() {
  case "$1" in
    http://127.0.0.1:*|http://localhost:*|https://127.0.0.1:*|https://localhost:*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

start_port_forward() {
  local namespace="$1"
  local service="$2"
  local local_port="$3"
  local remote_port="$4"

  log "Starting port-forward svc/${service} ${local_port}:${remote_port} in namespace ${namespace}"
  kubectl port-forward -n "${namespace}" "svc/${service}" "${local_port}:${remote_port}" >/tmp/"${service}"-portforward.log 2>&1 &
  local pid=$!
  _MANAGED_PIDS+=("${pid}")
  sleep 2

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    die "port-forward failed for svc/${service}; check /tmp/${service}-portforward.log"
  fi
}

if [ "${AUTO_PORT_FORWARD}" = "1" ]; then
  if is_local_url "${LLMD_BASE_URL}"; then
    start_port_forward "${LLMD_NAMESPACE}" "${LLMD_SERVICE}" "${LLMD_LOCAL_PORT}" 80
  fi

  if is_local_url "${FASTAPI_BASE_URL}"; then
    start_port_forward "${FASTAPI_NAMESPACE}" "${FASTAPI_SERVICE}" "${FASTAPI_LOCAL_PORT}" 8080
  fi
fi

require_cmd python3

export VALIDATE_LLMD_BASE_URL="${LLMD_BASE_URL}"
export VALIDATE_FASTAPI_BASE_URL="${FASTAPI_BASE_URL}"
export VALIDATE_MODEL_NAME="${MODEL_NAME}"
export VALIDATE_LLMD_API_MODE="${LLMD_API_MODE}"
export VALIDATE_HTTP_TIMEOUT_S="${HTTP_TIMEOUT_S}"
export VALIDATE_RAG_TEST_PROMPT="${RAG_TEST_PROMPT}"

python3 <<'PY'
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request

LLMD_BASE_URL = os.environ["VALIDATE_LLMD_BASE_URL"].rstrip("/")
FASTAPI_BASE_URL = os.environ["VALIDATE_FASTAPI_BASE_URL"].rstrip("/")
MODEL_NAME = os.environ["VALIDATE_MODEL_NAME"]
LLMD_API_MODE = os.environ["VALIDATE_LLMD_API_MODE"].strip().lower()
HTTP_TIMEOUT_S = float(os.environ["VALIDATE_HTTP_TIMEOUT_S"])
RAG_TEST_PROMPT = os.environ["VALIDATE_RAG_TEST_PROMPT"]

passes: list[str] = []
fails: list[str] = []


def record(ok: bool, label: str, reason: str | None = None) -> None:
    if ok:
        line = f"PASS {label}"
        passes.append(line)
        print(line)
    else:
        line = f"FAIL {label}" + (f" - {reason}" if reason else "")
        fails.append(line)
        print(line)


def http_json(method: str, url: str, payload: dict | None = None) -> tuple[int, object, str | None]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
            try:
                body = json.loads(raw)
            except Exception:
                body = raw
            return resp.status, body, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = raw
        return exc.code, body, str(exc)
    except Exception as exc:
        return 0, None, str(exc)


def wait_for_health(url: str, timeout_s: float = 60.0) -> tuple[int, object, str | None]:
    deadline = time.time() + timeout_s
    last = (0, None, "timeout")
    while time.time() < deadline:
        status, body, err = http_json("GET", f"{url}/health")
        if status == 200:
            return status, body, err
        last = (status, body, err)
        time.sleep(2)
    return last


def recursive_find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, key)
            if found is not None:
                return found
    return None


def route_taken(resp: object) -> str | None:
    if not isinstance(resp, dict):
        return None
    route = recursive_find_key(resp, "route_taken")
    if route is None and isinstance(resp.get("_route"), dict):
        route = resp["_route"].get("route_taken")
    return str(route) if route is not None else None


def rag_used(resp: object) -> bool | None:
    if not isinstance(resp, dict):
        return None

    direct = recursive_find_key(resp, "rag_used")
    if direct is not None:
        return bool(direct)

    route = route_taken(resp)
    if route in {"rag_plus_backend", "rag_plus_vllm"}:
        return True

    return None


def sources_present(resp: object) -> bool:
    if not isinstance(resp, dict):
        return False

    sources = recursive_find_key(resp, "sources")
    return isinstance(sources, list) and len(sources) > 0


def make_chat_payload(prompt: str) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 32,
        "temperature": 0,
    }


def make_completions_payload(prompt: str) -> dict:
    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0,
    }


def llmd_test_endpoint_and_payload() -> tuple[str, dict]:
    if LLMD_API_MODE == "chat":
        return "/v1/chat/completions", make_chat_payload("Reply with exactly: ok")
    if LLMD_API_MODE == "completions":
        return "/v1/completions", make_completions_payload("Reply with exactly: ok")
    raise ValueError(f"Unsupported LLMD_API_MODE={LLMD_API_MODE}")


# 1) llm-d /v1/models
status, body, err = http_json("GET", f"{LLMD_BASE_URL}/v1/models")
record(status == 200, "llm-d /v1/models", err or f"status={status}")

# 2) llm-d generation endpoint based on LLMD_API_MODE
try:
    llmd_path, llmd_payload = llmd_test_endpoint_and_payload()
except Exception as exc:
    record(False, "llm-d generation endpoint selection", str(exc))
    llmd_path, llmd_payload = "/v1/completions", make_completions_payload("Reply with exactly: ok")

status, body, err = http_json(
    "POST",
    f"{LLMD_BASE_URL}{llmd_path}",
    llmd_payload,
)
record(status == 200, f"llm-d {llmd_path}", err or f"status={status}")

# 3) FastAPI /health
status, body, err = wait_for_health(FASTAPI_BASE_URL, timeout_s=60.0)
health_ok = status == 200 and isinstance(body, dict) and bool(body.get("ok")) is True
record(health_ok, "FastAPI /health", err or f"status={status}")

health = body if isinstance(body, dict) else {}

# 4) Semantic cache repeated prompt
nonce = uuid.uuid4().hex[:10]
semantic_prompt = f"Semantic cache validation nonce {nonce}. Reply exactly with: cache-ok"

status1, resp1, err1 = http_json(
    "POST",
    f"{FASTAPI_BASE_URL}/v1/chat/completions",
    make_chat_payload(semantic_prompt),
)
status2, resp2, err2 = http_json(
    "POST",
    f"{FASTAPI_BASE_URL}/v1/chat/completions",
    make_chat_payload(semantic_prompt),
)

route1 = route_taken(resp1)
route2 = route_taken(resp2)

first_ok = status1 == 200 and route1 in {"plain_backend", "plain_vllm", "rag_plus_backend", "rag_plus_vllm"}
second_ok = status2 == 200 and route2 == "semantic_cache"

record(first_ok, "semantic cache first prompt", err1 or f"status={status1}, route={route1}")
record(second_ok, "semantic cache second prompt", err2 or f"status={status2}, route={route2}")

# 5) RAG check only if health indicates it is provisioned
rag_enabled = bool(health.get("rag_enabled"))
rag_runtime_enabled = bool(health.get("rag_runtime_enabled"))
rag_collection_exists = bool(health.get("rag_collection_exists"))
rag_manifest_root_non_empty = bool(health.get("rag_manifest_root_non_empty"))

rag_should_run = rag_enabled and rag_runtime_enabled and rag_collection_exists and rag_manifest_root_non_empty

if rag_should_run:
    status_rag, resp_rag, err_rag = http_json(
        "POST",
        f"{FASTAPI_BASE_URL}/v1/chat/completions",
        make_chat_payload(RAG_TEST_PROMPT),
    )
    rag_ok = status_rag == 200 and (
        rag_used(resp_rag) is True or sources_present(resp_rag)
    )
    reason = err_rag or f"status={status_rag}, route={route_taken(resp_rag)}, sources_present={sources_present(resp_rag)}"
    record(rag_ok, "RAG prompt", reason)
else:
    print("PASS RAG prompt - skipped (RAG not provisioned)")

if fails:
    print(f"FAIL summary - {len(fails)} failed, {len(passes)} passed")
    sys.exit(1)

print(f"PASS summary - {len(passes)} passed")
PY