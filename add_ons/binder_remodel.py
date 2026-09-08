"""Binder remodel add-on: expand JSON config → per-binder Boltz YAML docs.

Used automatically by the UI "Binder remodel" mode (no `addon` field needed).

Example config
--------------
{
  "job_name": "my_target_remodel",
  "target": {
    "id": "A",
    "name": "Target",
    "sequence": "M..."
  },
  "binder_id": "B",
  "binders": [
    {"name": "binder_1", "sequence": "S..."},
    {"name": "binder_2", "sequence": "M..."}
  ]
}

`binders` may also be a mapping: {"binder_1": "S...", "binder_2": "M..."}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from .common import clean_polymer_seq, parse_chain_id, sanitize_name

SCHEMA_HELP = """
<b>Binder remodel</b> JSON schema
<pre style="font-size:12px;white-space:pre-wrap;margin:6px 0;">
{
  "job_name": "optional_job_name",
  "target": { "id": "A", "name": "...", "sequence": "..." },
  "binder_id": "B",
  "binders": [
    { "name": "binder_1", "sequence": "..." },
    { "name": "binder_2", "sequence": "..." }
  ]
}
</pre>
<span style="color:#57606a;font-size:12px;">
<code>binders</code> can also be <code>{"name": "SEQ", ...}</code>.
This mode writes one Boltz YAML per binder (native predict inputs).
</span>
"""


def _normalize_binders(raw: Any) -> List[Tuple[str, str]]:
    binders: List[Tuple[str, str]] = []
    if raw is None:
        return binders
    if isinstance(raw, dict):
        for name, seq in raw.items():
            s = clean_polymer_seq(str(seq))
            if s:
                binders.append((sanitize_name(str(name)), s))
        return binders
    if isinstance(raw, list):
        for i, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                name = item.get("name") or item.get("id") or f"binder_{i}"
                seq = item.get("sequence") or item.get("seq") or ""
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name, seq = item[0], item[1]
            else:
                raise ValueError(f"Invalid binder entry at index {i}: {item!r}")
            s = clean_polymer_seq(str(seq))
            if not s:
                raise ValueError(f"Binder {name!r} has empty sequence.")
            binders.append((sanitize_name(str(name)), s))
        return binders
    raise ValueError("`binders` must be a list or object mapping.")


def load_config(source: str | Path | dict) -> dict:
    """Load binder_remodel config from dict, JSON string, or file path."""
    if isinstance(source, dict):
        data = dict(source)
    else:
        text = str(source).strip()
        path = Path(text).expanduser()
        if path.is_file():
            data = json.loads(path.read_text())
        else:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    # Ignore legacy "addon" key if present in older configs
    data.pop("addon", None)
    return data


def validate_config(config: dict) -> dict:
    config = load_config(config)
    target = config.get("target")
    if not isinstance(target, dict):
        raise ValueError("`target` must be an object with at least `sequence`.")
    tseq = clean_polymer_seq(str(target.get("sequence") or target.get("seq") or ""))
    if not tseq:
        raise ValueError("`target.sequence` is required.")
    binders = _normalize_binders(config.get("binders"))
    if not binders:
        raise ValueError("`binders` must contain at least one sequence.")
    parse_chain_id(target.get("id", "A"))
    parse_chain_id(config.get("binder_id", "B"))
    return config


def expand(config: dict) -> Tuple[str, List[Tuple[str, dict]]]:
    """Return (job_name, [(yaml_stem, yaml_dict), ...])."""
    config = validate_config(config)
    target = config["target"]
    tseq = clean_polymer_seq(str(target.get("sequence") or target.get("seq") or ""))
    tid = parse_chain_id(target.get("id", "A"))
    bid = parse_chain_id(config.get("binder_id", "B"))
    binders = _normalize_binders(config.get("binders"))

    job_name = str(config.get("job_name") or target.get("name") or "binder_remodel").strip()
    job_name = sanitize_name(job_name)

    docs: List[Tuple[str, dict]] = []
    for name, bseq in binders:
        docs.append(
            (
                sanitize_name(name),
                {
                    "version": 1,
                    "sequences": [
                        {"protein": {"id": tid, "sequence": tseq}},
                        {"protein": {"id": bid, "sequence": bseq}},
                    ],
                },
            )
        )
    return job_name, docs


def summary(config: dict) -> str:
    config = validate_config(config)
    target = config["target"]
    tseq = clean_polymer_seq(str(target.get("sequence") or ""))
    binders = _normalize_binders(config.get("binders"))
    tname = target.get("name") or "target"
    return (
        f"binder_remodel  target={tname} (len={len(tseq)})  "
        f"binders={len(binders)}  job_name={config.get('job_name') or tname}"
    )
