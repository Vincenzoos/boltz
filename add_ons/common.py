"""Shared helpers for add-on config parsing."""

from __future__ import annotations

import re
from typing import Any, List, Tuple


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name.strip("._") or "item"


def clean_polymer_seq(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq or "").upper()


def parse_chain_id(raw: Any) -> Any:
    """Accept 'A', ['A','B'], or 'A,B' → str or list[str]."""
    if isinstance(raw, list):
        parts = [str(p).strip() for p in raw if str(p).strip()]
        if not parts:
            raise ValueError("Chain id list is empty.")
        return parts[0] if len(parts) == 1 else parts
    raw_s = str(raw or "").strip()
    if not raw_s:
        raise ValueError("Chain id is required.")
    raw_s = raw_s.strip("[]")
    parts = [p.strip().strip("'\"") for p in re.split(r"[,\s]+", raw_s) if p.strip()]
    if not parts:
        raise ValueError("Chain id is required.")
    return parts[0] if len(parts) == 1 else parts


def parse_binders_text(raw: str) -> List[Tuple[str, str]]:
    """Parse binders from FASTA / name:SEQ / name SEQ text (legacy textarea format)."""
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    binders: List[Tuple[str, str]] = []
    i = 0
    anon = 0
    while i < len(lines):
        ln = lines[i]
        if not ln or ln.startswith("#"):
            i += 1
            continue
        if ln.startswith(">"):
            name = sanitize_name(ln[1:].strip() or f"binder_{anon + 1}")
            seq_parts: List[str] = []
            i += 1
            while i < len(lines) and lines[i] and not lines[i].startswith(">"):
                if ":" in lines[i]:
                    break
                toks = lines[i].split(None, 1)
                if (
                    len(toks) == 2
                    and re.fullmatch(r"[A-Za-z]+", clean_polymer_seq(toks[1]))
                    and not re.fullmatch(r"[A-Za-z]+", clean_polymer_seq(lines[i]))
                ):
                    break
                seq_parts.append(lines[i])
                i += 1
            seq = clean_polymer_seq("".join(seq_parts))
            if seq:
                binders.append((name, seq))
                anon += 1
            continue
        if ":" in ln:
            name, seq = ln.split(":", 1)
            name, seq = sanitize_name(name), clean_polymer_seq(seq)
            if seq:
                binders.append((name or f"binder_{anon + 1}", seq))
                anon += 1
            i += 1
            continue
        toks = ln.split(None, 1)
        if len(toks) == 2 and re.fullmatch(r"[A-Za-z]+", clean_polymer_seq(toks[1])):
            binders.append((sanitize_name(toks[0]), clean_polymer_seq(toks[1])))
            anon += 1
            i += 1
            continue
        seq = clean_polymer_seq(ln)
        if seq:
            anon += 1
            binders.append((f"binder_{anon}", seq))
        i += 1
    return binders
