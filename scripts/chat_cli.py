#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"

ROOT_DIR = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = ROOT_DIR / "scripts" / "deploy_fastapi_fullstack.sh"

_MANAGED_PROCS: list[subprocess.Popen[str]] = []


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def color_bool(value: bool) -> str:
    return c(str(value), BRIGHT_GREEN if value else BRIGHT_RED)


def color_route(route: str) -> str:
    if route == "exact_cache":
        return c(route, BRIGHT_GREEN)
    if route == "semantic_cache":
        return c(route, BRIGHT_CYAN)
    if route in {"rag_plus_backend", "rag_plus_vllm"}:
        return c(route, BRIGHT_YELLOW)
    if route in {"plain_backend", "plain_vllm"}:
        return c(route, BRIGHT_MAGENTA)
    return c(route, WHITE)


def color_ms(ms: float) -> str:
    if ms < 5:
        return c(f"{ms:.2f} ms", BRIGHT_GREEN)
    if ms < 50:
        return c(f"{ms:.2f} ms", BRIGHT_YELLOW)
    return c(f"{ms:.2f} ms", BRIGHT_RED)


def divider(title: str = "") -> None:
    line = "─" * 80
    if title:
        print(c(f"\n{line}\n{title}\n{line}", DIM))
    else:
        print(c(f"\n{line}", DIM))


def wrap_text(text: str, width: int = 100) -> str:
    parts: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            parts.append("")
        else:
            parts.append(textwrap.fill(para, width=width))
    return "\n".join(parts)


def _cleanup_managed_procs() -> None:
    for proc in reversed(_MANAGED_PROCS):
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass


atexit.register(_cleanup_managed_procs)


def _signal_handler(signum: int, frame: Any) -> None:
    _cleanup_managed_procs()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def is_local_base_url(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost"}


def health_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/health"


def check_health(base_url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(health_url(base_url), timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception:
        return None


def wait_for_health(base_url: str, timeout_s: float, poll_s: float = 1.5) -> dict[str, Any]:
    start = time.time()
    last_err = None

    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(health_url(base_url), timeout=5) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception as exc:
            last_err = exc
            time.sleep(poll_s)

    raise RuntimeError(f"service did not become healthy at {health_url(base_url)}: {last_err}")


def start_port_forward(base_url: str, namespace: str, service_name: str) -> subprocess.Popen[str] | None:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if host not in {"127.0.0.1", "localhost"}:
        return None

    cmd = [
        "kubectl",
        "port-forward",
        "-n",
        namespace,
        f"svc/{service_name}",
        f"{port}:8080",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    _MANAGED_PROCS.append(proc)
    time.sleep(2)

    time.sleep(1.0)
    if proc.poll() is not None:
        return None

    return proc


def run_deploy_script(deploy_script: Path) -> None:
    if not deploy_script.exists():
        raise FileNotFoundError(f"deploy script not found: {deploy_script}")

    subprocess.run(["bash", str(deploy_script)], cwd=str(ROOT_DIR), check=True)


def ensure_service_up(
    *,
    base_url: str,
    namespace: str,
    service_name: str,
    deploy_script: Path,
    auto_port_forward: bool,
    auto_deploy: bool,
    health_timeout_s: float,
) -> dict[str, Any]:
    current = check_health(base_url)
    if current is not None:
        return current

    if auto_port_forward and is_local_base_url(base_url):
        print(c("Service not reachable yet. Starting kubectl port-forward...", BRIGHT_YELLOW))
        start_port_forward(base_url, namespace, service_name)
        current = check_health(base_url)
        if current is not None:
            return current

    if auto_deploy:
        print(c("Service still not reachable. Running fullstack deploy script...", BRIGHT_YELLOW))
        run_deploy_script(deploy_script)

        if auto_port_forward and is_local_base_url(base_url):
            start_port_forward(base_url, namespace, service_name)

        print(c("Waiting for service health...", BRIGHT_YELLOW))
        return wait_for_health(base_url, timeout_s=health_timeout_s)

    raise RuntimeError(
        f"service is not reachable at {base_url}. "
        "Use --no-deploy only if the service is already up."
    )


def send_request(
    base_url: str,
    tenant_id: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Tenant-Id": tenant_id,
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def print_answer(resp: dict[str, Any]) -> None:
    answer = None
    try:
        answer = resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    if answer is None:
        try:
            answer = resp["choices"][0]["text"]
        except Exception:
            answer = json.dumps(resp, indent=2, ensure_ascii=False)

    print(c("\nAssistant:", BOLD + BRIGHT_BLUE))
    print(wrap_text(str(answer)))
    print()


def print_debug(resp: dict[str, Any]) -> None:
    route = resp.get("_route", {}) or {}
    cache = resp.get("_cache", {}) or {}
    rag = resp.get("_rag", {}) or {}
    perf = resp.get("_perf", {}) or {}

    backend_target = resp.get("_backend") or route.get("backend_target")
    backend_path = resp.get("_backend_path") or route.get("backend_path")
    backend_api_mode = resp.get("_backend_api_mode") or route.get("backend_api_mode")
    backend_http_status = perf.get("model_backend_http_status")

    divider("ROUTING")
    print(f"{c('route_taken', BOLD)}: {color_route(str(route.get('route_taken', 'unknown')))}")
    print(f"{c('backend_target', BOLD)}: {backend_target}")
    print(f"{c('backend_api_mode', BOLD)}: {backend_api_mode}")
    print(f"{c('backend_path', BOLD)}: {backend_path}")
    print(f"{c('backend_http_status', BOLD)}: {backend_http_status}")
    print(f"{c('benchmark_shadow_mode', BOLD)}: {color_bool(bool(route.get('benchmark_shadow_mode', False)))}")
    print(f"{c('rag_retrieve_every_request', BOLD)}: {color_bool(bool(route.get('rag_retrieve_every_request', False)))}")

    divider("CACHE")
    print(f"{c('cache_hit', BOLD)}: {color_bool(bool(cache.get('hit', False)))}")
    print(f"{c('scope', BOLD)}: {cache.get('scope')}")
    print(f"{c('tenant', BOLD)}: {cache.get('tenant_id')}")
    print(f"{c('kb_version', BOLD)}: {cache.get('kb_version')}")

    exact = cache.get("exact", {}) or {}
    semantic = cache.get("semantic", {}) or {}

    print(f"{c('exact_enabled', BOLD)}: {color_bool(bool(cache.get('exact_enabled', False)))}")
    print(f"{c('exact_hit', BOLD)}: {color_bool(bool(cache.get('exact_hit', False)))}")
    print(f"{c('exact_reject_reason', BOLD)}: {exact.get('reject_reason')}")

    print(f"{c('semantic_enabled', BOLD)}: {color_bool(bool(cache.get('semantic_enabled', False)))}")
    print(f"{c('semantic_hit', BOLD)}: {color_bool(bool(cache.get('semantic_hit', False)))}")
    print(f"{c('semantic_reject_reason', BOLD)}: {semantic.get('reject_reason')}")
    print(f"{c('semantic_shadow_hit', BOLD)}: {color_bool(bool(semantic.get('shadow_hit', False)))}")
    print(f"{c('semantic_shadow_reject_reason', BOLD)}: {semantic.get('shadow_reject_reason')}")

    divider("RAG")
    print(f"{c('rag_enabled', BOLD)}: {color_bool(bool(rag.get('enabled', False)))}")
    print(f"{c('rag_consulted', BOLD)}: {color_bool(bool(rag.get('consulted', False)))}")
    print(f"{c('rag_retrieved', BOLD)}: {color_bool(bool(rag.get('retrieved', False)))}")
    print(f"{c('rag_used', BOLD)}: {color_bool(bool(rag.get('used', False)))}")
    print(f"{c('rag_skip_reason', BOLD)}: {rag.get('skip_reason')}")
    print(f"{c('top_score', BOLD)}: {rag.get('top_score')}")
    print(f"{c('score_threshold', BOLD)}: {rag.get('score_threshold')}")
    print(f"{c('context_fingerprint', BOLD)}: {rag.get('context_fingerprint')}")

    sources = rag.get("sources", []) or []
    if sources:
        print(c("\nTop RAG sources:", BOLD + BRIGHT_YELLOW))
        for src in sources[:4]:
            md = src.get("metadata", {}) or {}
            print(
                f"  - rank={src.get('rank')} "
                f"score={src.get('score')} "
                f"file={md.get('file_name')} "
                f"page={md.get('page_label')} "
                f"tenant={md.get('tenant_id')}"
            )
            print(
                f"    chunk_id={md.get('chunk_id')} "
                f"object_key={md.get('object_key')} "
                f"pdf_object_key={md.get('pdf_object_key')}"
            )
    else:
        print(f"{c('sources', BOLD)}: []")

    divider("PERFORMANCE")
    backend_http_ms = float(perf.get("model_backend_http_ms", perf.get("vllm_http_ms", 0.0)) or 0.0)
    backend_json_ms = float(perf.get("model_backend_json_parse_ms", perf.get("vllm_json_parse_ms", 0.0)) or 0.0)

    print(f"{c('e2e_ms', BOLD)}: {color_ms(float(perf.get('e2e_ms', 0.0) or 0.0))}")
    print(f"{c('cache_lookup_ms', BOLD)}: {color_ms(float(perf.get('cache_lookup_ms', 0.0) or 0.0))}")
    print(f"{c('exact_cache_lookup_ms', BOLD)}: {color_ms(float(perf.get('exact_cache_lookup_ms', 0.0) or 0.0))}")
    print(f"{c('semantic_cache_lookup_ms', BOLD)}: {color_ms(float(perf.get('semantic_cache_lookup_ms', 0.0) or 0.0))}")
    print(f"{c('rag_retrieve_ms', BOLD)}: {color_ms(float(perf.get('rag_retrieve_ms', 0.0) or 0.0))}")
    print(f"{c('rag_format_ms', BOLD)}: {color_ms(float(perf.get('rag_format_ms', 0.0) or 0.0))}")
    print(f"{c('model_backend_http_ms', BOLD)}: {color_ms(backend_http_ms)}")
    print(f"{c('model_backend_json_parse_ms', BOLD)}: {color_ms(backend_json_ms)}")
    print(f"{c('cache_write_ms', BOLD)}: {color_ms(float(perf.get('cache_write_ms', 0.0) or 0.0))}")
    print(f"{c('shadow_eval_ms', BOLD)}: {color_ms(float(perf.get('shadow_eval_ms', 0.0) or 0.0))}")
    print(f"{c('original_prompt_tokens', BOLD)}: {perf.get('original_prompt_tokens')}")
    print(f"{c('augmented_prompt_tokens', BOLD)}: {perf.get('augmented_prompt_tokens')}")
    print()


def print_header(args: argparse.Namespace, health: dict[str, Any] | None = None) -> None:
    divider("LLM SERVICE CLI")
    print(f"{c('Base URL', BOLD)}: {args.base_url}")
    print(f"{c('Tenant', BOLD)}: {args.tenant}")
    print(f"{c('Model', BOLD)}: {args.model}")
    print(f"{c('Max output tokens', BOLD)}: {args.max_tokens}")
    print(f"{c('Temperature', BOLD)}: {args.temperature}")
    print(f"{c('Top-p', BOLD)}: {args.top_p}")
    print(f"{c('Auto deploy', BOLD)}: {color_bool(not args.no_deploy)}")
    print(f"{c('Auto port-forward', BOLD)}: {color_bool(not args.no_port_forward)}")
    if health is not None:
        print(f"{c('Health OK', BOLD)}: {color_bool(bool(health.get('ok', False)))}")
        print(f"{c('Backend', BOLD)}: {health.get('model_backend')}")
        print(f"{c('Semantic cache runtime', BOLD)}: {color_bool(bool(health.get('semantic_cache_runtime_enabled', False)))}")
        print(f"{c('RAG runtime', BOLD)}: {color_bool(bool(health.get('rag_runtime_enabled', False)))}")
    print()
    if args.prompt:
        print(c("Running one-shot prompt mode.\n", DIM))
    else:
        print("Type prompts. Type 'exit' to quit.\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive test/debug CLI for llm-service-kernel.")
    ap.add_argument("--base-url", default="http://127.0.0.1:18081")
    ap.add_argument("--tenant", default="tenantA")
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--show-debug", action="store_true")
    ap.add_argument("--prompt", default="", help="Send one prompt and exit.")
    ap.add_argument("--namespace", default="llm-service")
    ap.add_argument("--service-name", default="llm-service-kernel")
    ap.add_argument("--deploy-script", default=str(DEPLOY_SCRIPT))
    ap.add_argument("--health-timeout-s", type=float, default=300.0)
    ap.add_argument("--no-port-forward", action="store_true", help="Do not auto-start kubectl port-forward.")
    ap.add_argument("--no-deploy", action="store_true", help="Do not auto-run the deploy script when service is down.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        health = ensure_service_up(
            base_url=args.base_url,
            namespace=args.namespace,
            service_name=args.service_name,
            deploy_script=Path(args.deploy_script),
            auto_port_forward=not args.no_port_forward,
            auto_deploy=not args.no_deploy,
            health_timeout_s=args.health_timeout_s,
        )
    except Exception as exc:
        print(c(f"Failed to reach or start the service: {exc}", BRIGHT_RED), file=sys.stderr)
        return 1

    print_header(args, health=health)

    def run_one(prompt: str) -> int:
        try:
            resp = send_request(
                base_url=args.base_url,
                tenant_id=args.tenant,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_s=args.timeout_s,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(c(f"HTTP error {exc.code}: {body}", BRIGHT_RED), file=sys.stderr)
            return 1
        except Exception as exc:
            print(c(f"Request failed: {exc}", BRIGHT_RED), file=sys.stderr)
            return 1

        print_answer(resp)
        if args.show_debug:
            print_debug(resp)
        return 0

    if args.prompt:
        return run_one(args.prompt)

    while True:
        try:
            prompt = input(c("You: ", BOLD))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            continue
        if prompt.strip().lower() in {"exit", "quit"}:
            break

        rc = run_one(prompt)
        if rc != 0:
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
