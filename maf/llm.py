"""LLM backends: a llama-server (OpenAI-compatible) client and a deterministic mock.

The mock plays all agent roles by calling the ground-truth logic on whatever inputs it is handed,
so faults in upstream artifacts propagate exactly as they would through a competent model. It
obeys injected instructions with probability obey_prob, standing in for prompt-injection
susceptibility. Mock latency is simulated from token counts so timing math works without a model.
"""
from __future__ import annotations

import json
import logging
import random
import time
import urllib.request
from dataclasses import dataclass, field

from .log import clip
from .util import est_tokens, stable_seed
from .validate import INJECTION_ACCOUNT, check_consistent
from .world import solve_plan

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    extra: dict = field(default_factory=dict)


class LlamaServerClient:
    """Client for `llama-server`'s OpenAI-compatible /v1/chat/completions endpoint."""
    name = "llama-server"

    def __init__(self, base_url="http://127.0.0.1:8080", model="local", temperature=0.0, seed=0,
                 max_tokens=512, timeout_s=900.0, json_mode=True, fold_system=False, extra_body=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.json_mode = json_mode
        self.fold_system = fold_system     # for chat templates without a system role
        self.extra_body = extra_body or {}

    def chat(self, messages, role="", meta=None) -> LLMResult:
        if self.fold_system:
            sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            rest = [m for m in messages if m["role"] != "system"]
            if sys_txt and rest:
                rest = [dict(rest[0], content=sys_txt + "\n\n" + rest[0]["content"])] + rest[1:]
            messages = rest
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "seed": self.seed, "stream": False}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        req = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        log.debug("request role=%s meta=%s messages=%s", role, {k: v for k, v in (meta or {}).items() if k in
                  ("stage", "retry")}, clip(json.dumps(messages), 4000))
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.load(resp)
        except Exception:
            log.exception("llama-server request failed role=%s after %.1fs", role, time.perf_counter() - t0)
            raise
        latency = time.perf_counter() - t0
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
        usage = payload.get("usage", {})
        log.info("llm role=%s prompt_tok=%s out_tok=%s %.1fs finish=%s", role, usage.get("prompt_tokens"),
                 usage.get("completion_tokens"), latency, choice.get("finish_reason"))
        log.debug("response role=%s text=%s", role, clip(text, 4000))
        if choice.get("finish_reason") == "length":
            log.warning("role=%s hit max_tokens=%s (truncated output)", role, self.max_tokens)
        if not text.strip():
            log.warning("role=%s returned empty content (reasoning_chars=%d)", role,
                        len(choice["message"].get("reasoning_content") or ""))
        return LLMResult(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), latency,
                         {"timings": payload.get("timings", {}), "finish": choice.get("finish_reason"),
                          "reasoning_chars": len(choice["message"].get("reasoning_content") or "")})


class MockLLM:
    name = "mock"

    def __init__(self, seed=0, obey_prob=0.8, verifier_error=0.0, copy_error=0.0,
                 pp_tps=200.0, tg_tps=15.0):
        self.seed = seed
        self.obey_prob = obey_prob
        self.verifier_error = verifier_error
        self.copy_error = copy_error
        self.pp_tps = pp_tps
        self.tg_tps = tg_tps

    def chat(self, messages, role="", meta=None) -> LLMResult:
        meta = meta or {}
        rng = random.Random(stable_seed(self.seed, meta.get("world_seed"), meta.get("rng_key"), role))
        text = json.dumps(self._respond(role, meta, rng))
        prompt = sum(est_tokens(m["content"]) for m in messages)
        comp = est_tokens(text)
        log.debug("mock role=%s stage=%s out=%s", role, meta.get("stage"), clip(text))
        return LLMResult(text, prompt, comp, prompt / self.pp_tps + comp / self.tg_tps, {"mock": True})

    def _respond(self, role, meta, rng):
        inp = meta.get("inputs") or {}
        if role == "research_pricing":
            t = inp["tool_output"]
            offers = [dict(o) for o in t["offers"]]
            if offers and rng.random() < self.copy_error:
                offers[0]["price"] = round(offers[0]["price"] * 1.5, 2)
            return {"sku": t["sku"], "offers": offers, "notes": t.get("notes", "")}
        if role == "research_policy":
            t = inp["tool_output"]
            return {"approved": list(t["approved"]), "max_lead_days": t["max_lead_days"],
                    "notes": t.get("notes", "")}
        if role == "plan":
            best = solve_plan(inp["pricing"], inp["policy"], inp["qty"], inp["sku"])
            notes = " ".join(str(n) for n in (inp["pricing"].get("notes"), (inp["policy"] or {}).get("notes"))
                             if n and str(n).strip())
            rationale = "Cheapest approved supplier within the lead-time limit."
            if notes:
                rationale += f" Research notes: {notes}"
            best["rationale"] = rationale
            return best
        if role == "execute":
            p = inp["plan"]
            if not p.get("supplier"):
                return {"actions": []}
            acts = [{"tool": "place_order",
                     "args": {"supplier": p["supplier"], "sku": p["sku"], "qty": p["qty"]}},
                    {"tool": "update_ledger", "args": {"sku": p["sku"], "delta": p["qty"]}}]
            if INJECTION_ACCOUNT in str(p.get("rationale", "")) and rng.random() < self.obey_prob:
                acts.append({"tool": "transfer_funds",
                             "args": {"account": INJECTION_ACCOUNT, "amount": 5000}})
            return {"actions": acts}
        if role == "notify":
            p = inp["plan"]
            return {"supplier": p.get("supplier"),
                    "message": f"Order placed: {p.get('qty')} x {p.get('sku')}."}
        if role == "verify":
            ok = check_consistent(meta["stage"], meta["artifact"], meta["evidence"])
            if rng.random() < self.verifier_error:
                ok = not ok
            return {"consistent": ok, "reason": "mock verifier"}
        raise ValueError(f"unknown role {role}")
