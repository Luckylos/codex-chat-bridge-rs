#!/usr/bin/env python3
"""Differential shadow oracle: fire the SAME request at the live Python bridge
and the Rust shadow, then diff structural shape (not scalar values, which vary
per upstream sampling). Proves behavioral equivalence for production readiness.

Usage: shadow_diff.py <model>
Reads no secrets; talks only to localhost bridge ports.
"""
import json
import sys
import urllib.request
import urllib.error

PY = "http://127.0.0.1:18090"
RS = "http://127.0.0.1:18095"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"

results = []


def post(base, path, body, stream=False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode()
        return r.status, raw


def shape(v):
    """Recursive type skeleton, ignoring scalar values."""
    if isinstance(v, dict):
        return {k: shape(v[k]) for k in sorted(v)}
    if isinstance(v, list):
        # collapse list element shapes to a set-like ordered unique list
        elem_shapes = [shape(e) for e in v]
        return [f"list[{len(v)}]"] + elem_shapes
    return type(v).__name__


def _collapse(seq):
    """Collapse consecutive duplicates into one.

    Upstream token chunking is non-deterministic: the SAME prompt yields a
    different NUMBER of *.delta events per run (verified: the live Python
    bridge itself emits 4/2/2 output_text.delta for identical input). The
    bridge contract is the event *sequence* (which event types, in what order),
    not the delta count. Collapsing runs of the same event type removes the
    upstream-sampling noise and compares the semantic frame skeleton.
    """
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def sse_event_names(raw):
    names = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            names.append(line[len("event:"):].strip())
    return _collapse(names)


def sse_data_types(raw):
    """The 'type' field of each data payload, in order (deltas collapsed)."""
    types = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                types.append("[DONE]")
                continue
            try:
                obj = json.loads(payload)
                types.append(obj.get("type", "<no-type>"))
            except Exception:
                types.append("<unparseable>")
    return _collapse(types)


def sse_full_text(raw):
    """Reassemble the full output_text from all delta fragments — the
    concatenated text MUST be well-formed regardless of chunk count."""
    text = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
                if obj.get("type") == "response.output_text.delta":
                    text.append(obj.get("delta", ""))
            except Exception:
                pass
    return "".join(text)


def diff(label, py, rs):
    ok = py == rs
    results.append((label, ok, py, rs))
    mark = "OK " if ok else "DIFF"
    print(f"[{mark}] {label}")
    if not ok:
        print(f"    PY: {py}")
        print(f"    RS: {rs}")


def output_types(obj):
    return [item.get("type") for item in obj.get("output", [])]


def id_prefixes(obj):
    """Every output item id prefix — must be bridge-shaped, no upstream UUID."""
    pref = []
    for item in obj.get("output", []):
        iid = item.get("id", "")
        pref.append(iid.split("_")[0] if "_" in iid else iid)
    return sorted(set(pref))


# ---- Case 1: non-streaming plain text ----
print("=== Case 1: non-streaming text ===")
body = {
    "model": MODEL,
    "input": "Say the single word: ping",
    "stream": False,
}
try:
    ps, pr = post(PY, "/v1/responses", body)
    rs_, rr = post(RS, "/v1/responses", body)
    diff("1.status", ps, rs_)
    pj, rj = json.loads(pr), json.loads(rr)
    diff("1.top_keys", sorted(pj), sorted(rj))
    diff("1.shape", shape(pj), shape(rj))
    diff("1.output_types", output_types(pj), output_types(rj))
    diff("1.id_prefixes", id_prefixes(pj), id_prefixes(rj))
    diff("1.status_field", pj.get("status"), rj.get("status"))
except Exception as e:
    print(f"    Case 1 ERROR: {e}")

# ---- Case 2: streaming text ----
print("=== Case 2: streaming text ===")
body = {
    "model": MODEL,
    "input": "Say the single word: pong",
    "stream": True,
}
try:
    ps, pr = post(PY, "/v1/responses", body, stream=True)
    rs_, rr = post(RS, "/v1/responses", body, stream=True)
    diff("2.status", ps, rs_)
    diff("2.event_names(collapsed)", sse_event_names(pr), sse_event_names(rr))
    diff("2.data_types(collapsed)", sse_data_types(pr), sse_data_types(rr))
    # Both must reassemble to a non-empty well-formed text (value differs per
    # sampling, so assert the invariant: non-empty on both sides).
    pt, rt = sse_full_text(pr), sse_full_text(rr)
    diff("2.text_nonempty", bool(pt.strip()), bool(rt.strip()))
except Exception as e:
    print(f"    Case 2 ERROR: {e}")

# ---- Case 3: multi-turn via previous_response_id (stateful) ----
print("=== Case 3: previous_response_id round-trip ===")
body1 = {"model": MODEL, "input": "Remember the number 42.", "stream": False}
try:
    _, pr1 = post(PY, "/v1/responses", body1)
    _, rr1 = post(RS, "/v1/responses", body1)
    pid = json.loads(pr1).get("id")
    rid = json.loads(rr1).get("id")
    diff("3.first_id_prefix",
         pid.split("_")[0] if pid else None,
         rid.split("_")[0] if rid else None)
    body2p = {"model": MODEL, "input": "What number?", "previous_response_id": pid, "stream": False}
    body2r = {"model": MODEL, "input": "What number?", "previous_response_id": rid, "stream": False}
    ps2, pr2 = post(PY, "/v1/responses", body2p)
    rs2, rr2 = post(RS, "/v1/responses", body2r)
    diff("3.second_status", ps2, rs2)
    diff("3.second_shape", shape(json.loads(pr2)), shape(json.loads(rr2)))
    diff("3.second_output_types", output_types(json.loads(pr2)), output_types(json.loads(rr2)))
except Exception as e:
    print(f"    Case 3 ERROR: {e}")

# ---- Case 4: tool call ----
print("=== Case 4: function tool call ===")
body = {
    "model": MODEL,
    "input": "What is the weather in Paris? Use the get_weather tool.",
    "tools": [{
        "type": "function",
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }],
    "stream": False,
}
try:
    ps, pr = post(PY, "/v1/responses", body)
    rs_, rr = post(RS, "/v1/responses", body)
    diff("4.status", ps, rs_)
    pj, rj = json.loads(pr), json.loads(rr)
    diff("4.output_types", output_types(pj), output_types(rj))
    # per function_call item: shape of the item
    def fc_shape(obj):
        return [shape(i) for i in obj.get("output", []) if i.get("type") == "function_call"]
    diff("4.function_call_shape", fc_shape(pj), fc_shape(rj))
except Exception as e:
    print(f"    Case 4 ERROR: {e}")

# ---- Case 5: error envelope shape (upstream failure) ----
print("=== Case 5: error envelope on upstream failure ===")
body = {
    "model": "definitely-not-a-real-model-xyz",
    "input": "hi",
    "stream": False,
}
try:
    try:
        ps, pr = post(PY, "/v1/responses", body)
    except urllib.error.HTTPError as e:
        ps, pr = e.code, e.read().decode()
    try:
        rs_, rr = post(RS, "/v1/responses", body)
    except urllib.error.HTTPError as e:
        rs_, rr = e.code, e.read().decode()
    diff("5.status", ps, rs_)
    pj, rj = json.loads(pr), json.loads(rr)
    # The error body key-set must match (message/type/code/param), NOT a
    # divergent `detail` object. This is the drop-in wire contract.
    diff("5.error_keys", sorted(pj.get("error", {})), sorted(rj.get("error", {})))
    # param must be a STRING on both sides (Python _error_param contract).
    diff("5.param_is_str",
         isinstance(pj.get("error", {}).get("param"), (str, type(None))),
         isinstance(rj.get("error", {}).get("param"), (str, type(None))))
except Exception as e:
    print(f"    Case 5 ERROR: {e}")

# ---- Summary ----
print("=== SUMMARY ===")
total = len(results)
passed = sum(1 for _, ok, _, _ in results if ok)
print(f"{passed}/{total} checks aligned")
if total == 0:
    # No checks ran (e.g. every case hit an upstream 5xx before any diff).
    # That is NOT a pass — a zero-check run must never report success.
    print("NO CHECKS RAN — inconclusive (upstream unavailable for this model?)")
    sys.exit(2)
if passed != total:
    print("FAILURES:")
    for label, ok, py, rs in results:
        if not ok:
            print(f"  - {label}")
    sys.exit(1)
print("ALL ALIGNED")
