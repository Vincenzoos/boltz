"""
Boltz prediction interactive UI (ipywidgets).

Supports the full Boltz YAML input surface:
  sequences: protein | dna | rna | ligand (smiles/ccd)
  optional msa / modifications / cyclic
  constraints: bond | pocket | contact
  templates: cif | pdb
  properties: affinity (ligand binder)

Launch from notebooks/Boltz_Remodel_UI.ipynb (or any notebook):

    from add_ons.predict_ui import launch_ui
    launch_ui()

Best viewed in JupyterLab / classic Notebook in a browser.
Cursor's notebook widget renderer may fail with ipywidgetsKernel errors.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets as widgets
import yaml
from IPython.display import display

# ---------------------------------------------------------------------------
# Paths / styling
# ---------------------------------------------------------------------------

BOLTZ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = BOLTZ_ROOT / "outputs"
DEFAULT_CACHE = Path(os.environ.get("BOLTZ_CACHE", Path.home() / ".boltz"))
EXAMPLES_DIR = BOLTZ_ROOT / "examples"
CONFIGS_DIR_REL = "add_ons/configs"
EXAMPLE_CONFIG_NAME = "binder_remodel.example.json"
EXAMPLE_CONFIG_REL = f"{CONFIGS_DIR_REL}/{EXAMPLE_CONFIG_NAME}"
PROTEIN_AA = "ACDEFGHIKLMNPQRSTVWY"
NAME_MAX_LEN = 100
SEQ_MAX_LEN = 500
# Letter start; only underscore/hyphen as specials; max 100 chars.
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,99}$")

BANNER = (
    "background:#dbeafe;padding:12px 16px;border-radius:6px;"
    "margin:8px 0;font-family:sans-serif;"
)
OK = "color:#1a7f37;font-weight:600;"
ERR = "color:#cf222e;font-weight:600;"
MUTED = "color:#57606a;"

RUNNING_JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

from . import CONFIGS_DIR, list_config_files
from . import binder_remodel as binder_remodel_addon

EXAMPLE_BINDER_REMODEL_JSON = """{
  "job_name": "example_binder_remodel",
  "target": {
    "id": "A",
    "name": "Target",
    "sequence": "REPLACE_WITH_TARGET_SEQUENCE"
  },
  "binder_id": "B",
  "binders": [
    {"name": "binder_1", "sequence": "REPLACE_WITH_BINDER_SEQUENCE"}
  ]
}
"""

EXAMPLE_YAML = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG
  - protein:
      id: B
      sequence: MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA
"""

SCHEMA_HELP = """
<b>Supported Boltz YAML input</b>
<ul style="margin:4px 0 0 16px;padding:0;font-size:13px;">
  <li><code>sequences</code>: <b>protein</b> / <b>dna</b> / <b>rna</b> (sequence) or <b>ligand</b> (smiles <i>or</i> ccd)</li>
  <li>protein options: <code>msa</code> path / <code>empty</code>, <code>modifications</code>, <code>cyclic</code></li>
  <li>chain <code>id</code>: single (<code>A</code>) or copies (<code>[A, B]</code>)</li>
  <li><code>constraints</code>: <code>bond</code>, <code>pocket</code>, <code>contact</code></li>
  <li><code>templates</code>: <code>cif</code> / <code>pdb</code> (+ optional chain mapping, force, threshold)</li>
  <li><code>properties</code>: <code>affinity</code> with ligand <code>binder</code> chain (Boltz-2, small molecules only)</li>
</ul>
<p style="margin:6px 0 0 0;font-size:12px;color:#57606a;">
Input to <code>boltz predict</code> can be a single <code>.yaml</code>/<code>.fasta</code> or a directory of them.
</p>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(text: str) -> widgets.HTML:
    return widgets.HTML(f'<div style="{BANNER}"><b>{text}</b></div>')


def _list_output_zip_options(outputs_dir: Path) -> List[Tuple[str, str]]:
    """Return (label, value) pairs for zipping outputs/ or one subfolder."""
    options: List[Tuple[str, str]] = [("Entire outputs/ folder", "__all__")]
    if outputs_dir.is_dir():
        for p in sorted(outputs_dir.iterdir(), key=lambda x: x.name.lower()):
            if p.is_dir() and not p.name.startswith("."):
                options.append((p.name, p.name))
    return options


def _zip_outputs(
    outputs_dir: Path,
    selection: str,
    zip_name: str = "outputs.zip",
    *,
    dest_root: Optional[Path] = None,
) -> Path:
    """Zip outputs_dir (or one subfolder) into dest_root / basename(zip_name)."""
    name = (zip_name or "outputs.zip").strip() or "outputs.zip"
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    root = dest_root if dest_root is not None else BOLTZ_ROOT
    dest = root / Path(name).name

    if selection == "__all__":
        source = outputs_dir
        arc_root = Path(outputs_dir.name or "outputs")
    else:
        source = outputs_dir / selection
        if not source.is_dir():
            raise FileNotFoundError(f"Output folder not found: {source}")
        arc_root = Path(selection)

    if not source.exists():
        raise FileNotFoundError(f"Nothing to zip: {source}")

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_dir():
                    continue
                rel = path.relative_to(source)
                zf.write(path, arcname=str(arc_root / rel))
        else:
            zf.write(source, arcname=str(arc_root))
    return dest


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name.strip("._") or "item"


def _rel_to_root(path: Path) -> str:
    """Display path relative to the Boltz repo root when possible."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(BOLTZ_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_under_root(raw: str) -> Path:
    """Resolve a UI path: absolute as-is, otherwise under BOLTZ_ROOT."""
    p = Path((raw or "").strip()).expanduser()
    if not str(p):
        raise ValueError("Path is empty.")
    return p.resolve() if p.is_absolute() else (BOLTZ_ROOT / p).resolve()


def _clean_polymer_seq(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq or "").upper()


def _parse_id_field(raw: str) -> Any:
    """Parse chain id: 'A' or 'A,B' / '[A, B]' → str or list."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Chain id is required.")
    raw = raw.strip("[]")
    parts = [p.strip().strip("'\"") for p in re.split(r"[,\s]+", raw) if p.strip()]
    if not parts:
        raise ValueError("Chain id is required.")
    return parts[0] if len(parts) == 1 else parts



def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def _parse_yaml_docs(raw: str) -> List[dict]:
    """Parse one or more YAML documents (separated by ---)."""
    docs = list(yaml.safe_load_all(raw or ""))
    out = [d for d in docs if d]
    if not out:
        raise ValueError("YAML is empty.")
    for i, d in enumerate(out):
        if not isinstance(d, dict):
            raise ValueError(f"YAML document {i + 1} must be a mapping.")
        if "sequences" not in d:
            raise ValueError(f"YAML document {i + 1} missing required key: sequences")
    return out


def _parse_modifications(raw: str) -> List[dict]:
    """Lines like: 12 MSE   or  12:MSE"""
    mods: List[dict] = []
    for ln in (raw or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        ln = ln.replace(",", " ")
        if ":" in ln:
            pos_s, ccd = ln.split(":", 1)
        else:
            parts = ln.split()
            if len(parts) != 2:
                raise ValueError(f"Bad modification line: {ln!r} (use 'POS CCD')")
            pos_s, ccd = parts
        mods.append({"position": int(pos_s.strip()), "ccd": ccd.strip()})
    return mods


def _optional_yaml_section(raw: str, key: str) -> Optional[Any]:
    raw = (raw or "").strip()
    if not raw:
        return None
    data = yaml.safe_load(raw)
    if data is None:
        return None
    # allow either bare list or {key: [...]}
    if isinstance(data, dict) and key in data:
        return data[key]
    return data


def _build_entity_dict(
    *,
    entity_type: str,
    chain_id: Any,
    sequence: str,
    smiles: str,
    ccd: str,
    msa: str,
    cyclic: bool,
    modifications_raw: str,
) -> dict:
    body: Dict[str, Any] = {"id": chain_id}
    if entity_type in ("protein", "dna", "rna"):
        seq = _clean_polymer_seq(sequence) if entity_type == "protein" else re.sub(r"\s+", "", sequence or "").upper()
        if not seq:
            raise ValueError(f"{entity_type} requires a sequence.")
        body["sequence"] = seq
        if entity_type == "protein":
            msa = (msa or "").strip()
            if msa:
                body["msa"] = msa
        if cyclic:
            body["cyclic"] = True
        mods = _parse_modifications(modifications_raw)
        if mods:
            body["modifications"] = mods
    elif entity_type == "ligand":
        smiles = (smiles or "").strip()
        ccd = (ccd or "").strip()
        if bool(smiles) == bool(ccd):
            raise ValueError("Ligand requires exactly one of SMILES or CCD.")
        if smiles:
            body["smiles"] = smiles
        else:
            body["ccd"] = ccd
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return {entity_type: body}


def _list_gpu_options() -> List[Tuple[str, str]]:
    """Return (label, CUDA_VISIBLE_DEVICES value) pairs — GPUs only."""
    options: List[Tuple[str, str]] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return [("No GPUs detected", "")]
        for row in csv.reader((proc.stdout or "").splitlines()):
            if len(row) < 5:
                continue
            idx_raw, name, total_raw, _used_raw, free_raw = [p.strip() for p in row[:5]]
            try:
                idx = str(int(idx_raw))
                total = int(float(total_raw))
                free = int(float(free_raw))
            except (TypeError, ValueError):
                continue
            if total < 0 or free < 0:
                continue
            label = f"GPU {idx}: {name} — {free} MiB free / {total} MiB"
            options.append((label, idx))
    except Exception:
        return [("No GPUs detected", "")]
    return options or [("No GPUs detected", "")]


def _prefer_freest_gpu(gpu_opts: List[Tuple[str, str]]) -> str:
    """Pick the GPU with the most free memory; fall back to first option."""
    best: Optional[Tuple[int, str]] = None
    for label, val in gpu_opts:
        if not val:
            continue
        if " MiB free" not in label:
            continue
        try:
            free = int(label.split(" — ")[1].split(" MiB free")[0].replace(",", ""))
        except (IndexError, ValueError):
            continue
        if best is None or free > best[0]:
            best = (free, val)
    if best is not None:
        return best[1]
    return gpu_opts[0][1] if gpu_opts else ""


def _readonly_textarea(value: str, height: str = "300px") -> widgets.Textarea:
    return widgets.Textarea(
        value=value,
        description="",
        disabled=True,
        layout=widgets.Layout(width="95%", height=height),
        style={"description_width": "0px"},
    )


def _nvidia_smi_text() -> str:
    """Full nvidia-smi report text for the monitoring panel."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return (
            "nvidia-smi is unavailable (not installed or not on PATH).\n"
            "GPU details cannot be displayed."
        )
    except Exception as exc:
        return f"Unable to run nvidia-smi: {exc}"

    report = "\n".join(
        part for part in (proc.stdout or "", proc.stderr or "") if part
    ).strip()
    if proc.returncode != 0:
        message = f"nvidia-smi returned exit code {proc.returncode}."
        return f"{message}\n{report}" if report else message
    return report or "nvidia-smi returned no output."


def _parse_process_memory_output(output: str) -> List[Tuple[int, str]]:
    """Parse ``pid, used_gpu_memory`` rows from nvidia-smi."""
    rows: List[Tuple[int, str]] = []
    for row in csv.reader((output or "").splitlines()):
        if len(row) < 2:
            continue
        try:
            pid = int(row[0].strip())
        except (TypeError, ValueError):
            continue
        memory = row[1].strip()
        if memory.lower() in {"n/a", "na", "unknown"}:
            continue
        rows.append((pid, memory))
    return rows


def _is_boltz_cmdline(command_line: str) -> bool:
    """True if the process looks like a Boltz predict / main entrypoint."""
    cl = command_line.lower()
    if "boltz.main" in cl:
        return True
    if "boltz" in cl and "predict" in cl:
        return True
    return False


def _boltz_gpu_processes_text() -> str:
    """Table of active Boltz processes currently using GPU memory."""
    try:
        gpu_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "nvidia-smi is unavailable; active Boltz GPU jobs cannot be listed."
    except Exception as exc:
        return f"Unable to query visible GPUs: {exc}"

    if gpu_proc.returncode != 0:
        detail = (gpu_proc.stderr or gpu_proc.stdout or "").strip()
        message = (
            "nvidia-smi could not list visible GPUs; "
            "active Boltz GPU jobs cannot be listed."
        )
        return f"{message}\n{detail}" if detail else message

    gpu_indices: List[str] = []
    for line in (gpu_proc.stdout or "").splitlines():
        try:
            gpu_indices.append(str(int(line.strip())))
        except (TypeError, ValueError):
            continue

    rows: List[Tuple[str, str, str, str, str, str]] = []
    for gpu_index in gpu_indices:
        try:
            apps_proc = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    gpu_index,
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "nvidia-smi is unavailable; active Boltz GPU jobs cannot be listed."
        except Exception as exc:
            return f"Unable to query GPU {gpu_index}: {exc}"
        if apps_proc.returncode != 0:
            detail = (apps_proc.stderr or apps_proc.stdout or "").strip()
            message = f"Unable to query compute applications on GPU {gpu_index}."
            return f"{message}\n{detail}" if detail else message

        for pid, memory in _parse_process_memory_output(apps_proc.stdout):
            try:
                ps_proc = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "user=,comm=,args="],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return "ps is unavailable; process details cannot be resolved."
            except Exception as exc:
                return f"Unable to read process {pid}: {exc}"
            if ps_proc.returncode != 0 or not (ps_proc.stdout or "").strip():
                continue

            ps_fields = ps_proc.stdout.strip().split(None, 2)
            if len(ps_fields) < 3:
                continue
            user, process_name, command_line = ps_fields
            if not _is_boltz_cmdline(command_line):
                continue
            try:
                working_directory = os.readlink(f"/proc/{pid}/cwd")
            except Exception:
                working_directory = "?"
            mem_label = memory if "mib" in memory.lower() else f"{memory} MiB"
            rows.append(
                (gpu_index, str(pid), user, process_name, working_directory, mem_label)
            )

    if not rows:
        return "No active Boltz jobs are currently using GPU memory."

    column_widths = (8, 12, 16, 24, 48)
    header = (
        f"{'GPU':<{column_widths[0]}}"
        f"{'PID':<{column_widths[1]}}"
        f"{'USER':<{column_widths[2]}}"
        f"{'PROCESS':<{column_widths[3]}}"
        f"{'CWD':<{column_widths[4]}}"
        "GPU MEMORY"
    ).rstrip()
    separator = "-" * len(header)

    formatted_rows = []
    for gpu, pid, user, process, cwd, memory in rows:
        values = [gpu, pid, user, process, cwd]
        cells = []
        for value, width in zip(values, column_widths):
            value = str(value)
            if len(value) > width:
                value = value[: max(0, width - 3)] + "..."
            cells.append(f"{value:<{width}}")
        cells.append(str(memory))
        formatted_rows.append("".join(cells).rstrip())

    return "\n".join([header, separator, *formatted_rows])


def _find_boltz() -> str:
    which = shutil.which("boltz")
    if which:
        return which
    try:
        import boltz  # noqa: F401

        return "python -m boltz.main"
    except Exception:
        return "boltz"


def _list_examples() -> List[Tuple[str, str]]:
    opts = [("— select example —", "")]
    if EXAMPLES_DIR.is_dir():
        for p in sorted(EXAMPLES_DIR.glob("*.yaml")):
            opts.append((p.name, str(p.resolve())))
    return opts


def _build_cmd(
    boltz_bin: str,
    input_path: Path,
    out_dir: Path,
    *,
    model: str,
    use_msa_server: bool,
    use_potentials: bool,
    diffusion_samples: int,
    recycling_steps: int,
    sampling_steps: int,
    override: bool,
    output_format: str,
    cache: str,
    accelerator: str,
    devices: int,
    write_full_pae: bool,
    seed: Optional[int],
) -> List[str]:
    if boltz_bin.startswith("python "):
        cmd = boltz_bin.split() + ["predict", str(input_path)]
    else:
        cmd = [boltz_bin, "predict", str(input_path)]
    cmd += [
        "--out_dir",
        str(out_dir),
        "--model",
        model,
        "--diffusion_samples",
        str(diffusion_samples),
        "--recycling_steps",
        str(recycling_steps),
        "--sampling_steps",
        str(sampling_steps),
        "--output_format",
        output_format,
        "--cache",
        cache,
        "--accelerator",
        accelerator,
        "--devices",
        str(devices),
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if use_potentials:
        cmd.append("--use_potentials")
    if override:
        cmd.append("--override")
    if write_full_pae:
        cmd.append("--write_full_pae")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


TMUX_SESSION_PREFIX = "boltz_"


def _tmux_session_name(out_dir: Path) -> str:
    """Stable tmux session name for a job output directory."""
    name = _sanitize_name(out_dir.parent.name) or "job"
    return f"{TMUX_SESSION_PREFIX}{name}"[:64]


def _job_log_path(out_dir: Path) -> Path:
    return out_dir.parent / "run.log"


def _tmux_session_running(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def _tmux_start(session: str, shell_script: str) -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_script],
        check=True,
    )


def _tmux_stop(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


def _list_tmux_boltz_sessions() -> List[str]:
    """Return names of running tmux sessions started by this UI."""
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [name for name in result.stdout.splitlines() if name.startswith(TMUX_SESSION_PREFIX)]


def _session_job_name(session: str) -> str:
    if session.startswith(TMUX_SESSION_PREFIX):
        return session[len(TMUX_SESSION_PREFIX) :]
    return session


def _list_running_boltz_jobs() -> List[Dict[str, Any]]:
    """Merge live tmux sessions with in-memory job metadata."""
    sessions = _list_tmux_boltz_sessions()
    with JOBS_LOCK:
        by_session = {info["session"]: info for info in RUNNING_JOBS.values()}
    jobs: List[Dict[str, Any]] = []
    for session in sorted(sessions):
        info = by_session.get(session, {})
        job_name = _session_job_name(session)
        started = info.get("started")
        age = ""
        if started:
            mins = max(0, int((time.time() - started) / 60))
            age = f", {mins}m"
        jobs.append(
            {
                "session": session,
                "job_name": job_name,
                "log_path": info.get("log_path", ""),
                "started": started,
                "label": f"{job_name} ({session}{age})",
            }
        )
    return jobs


def _build_tmux_shell(cmd: List[str], env: Dict[str, str], log_path: Path) -> str:
    """Build a bash script that runs cmd, tees output to log_path, and records exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exports = [
        f"export {k}={shlex.quote(str(v))}"
        for k, v in sorted(env.items())
        if env.get(k) != os.environ.get(k)
    ]
    cmd_str = subprocess.list2cmdline(cmd)
    log_q = shlex.quote(str(log_path))
    parts = exports + [
        "set -o pipefail",
        f"{cmd_str} 2>&1 | tee -a {log_q}",
        "ec=${PIPESTATUS[0]}",
        f'echo "[done] exit=$ec $(date +%Y-%m-%d\\ %H:%M:%S)" >> {log_q}',
        "exit $ec",
    ]
    return " && ".join(parts)


def _read_log_tail(log_path: Path, offset: int) -> Tuple[int, str]:
    if not log_path.is_file():
        return offset, ""
    with log_path.open("r", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        return handle.tell(), chunk


def _parse_log_exit(log_path: Path) -> Optional[int]:
    if not log_path.is_file():
        return None
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        match = re.search(r"\[done\] exit=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _predictions_dir(out_dir: Path) -> Optional[Path]:
    """Resolve Boltz predictions folder (out/predictions or out/boltz_results_*/predictions)."""
    direct = out_dir / "predictions"
    if direct.is_dir():
        return direct
    matches = sorted(out_dir.glob("boltz_results_*/predictions"))
    for path in matches:
        if path.is_dir():
            return path
    return None


def _collect_results(out_dir: Path) -> List[Dict[str, Any]]:
    pred = _predictions_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    if pred is None:
        return rows
    for folder in sorted(pred.iterdir()):
        if not folder.is_dir():
            continue
        confs = sorted(folder.glob("confidence_*_model_*.json"))
        aff_files = list(folder.glob("affinity_*.json"))
        aff: Dict[str, Any] = {}
        if aff_files:
            try:
                aff = json.loads(aff_files[0].read_text())
            except Exception:
                aff = {}
        if not confs:
            row = {"design": folder.name, "model": "-", "status": "no confidence yet"}
            if aff:
                row["affinity_pred_value"] = aff.get("affinity_pred_value")
                row["affinity_probability_binary"] = aff.get("affinity_probability_binary")
            rows.append(row)
            continue
        for conf_path in confs:
            try:
                data = json.loads(conf_path.read_text())
            except Exception as exc:
                rows.append({"design": folder.name, "model": conf_path.name, "status": f"read error: {exc}"})
                continue
            m = re.search(r"model_(\d+)", conf_path.name)
            model_idx = m.group(1) if m else "?"
            row = {
                "design": folder.name,
                "model": model_idx,
                "confidence_score": _round(data.get("confidence_score")),
                "iptm": _round(data.get("iptm")),
                "ptm": _round(data.get("ptm")),
                "complex_plddt": _round(data.get("complex_plddt")),
                "protein_iptm": _round(data.get("protein_iptm")),
                "ligand_iptm": _round(data.get("ligand_iptm")),
            }
            if aff:
                row["affinity_pred_value"] = _round(aff.get("affinity_pred_value"))
                row["affinity_probability_binary"] = _round(aff.get("affinity_probability_binary"))
            rows.append(row)
    rows.sort(
        key=lambda r: (
            -(r.get("iptm") if isinstance(r.get("iptm"), (int, float)) else -1),
            -(r.get("confidence_score") if isinstance(r.get("confidence_score"), (int, float)) else -1),
        )
    )
    return rows


def _round(v: Any) -> Any:
    try:
        if v is None:
            return ""
        return round(float(v), 4)
    except Exception:
        return v


def _results_html(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f'<p style="{MUTED}">No prediction results found yet.</p>'
    cols = [
        "design",
        "model",
        "iptm",
        "confidence_score",
        "ptm",
        "complex_plddt",
        "protein_iptm",
        "ligand_iptm",
        "affinity_pred_value",
        "affinity_probability_binary",
    ]
    # drop empty optional columns
    used = [c for c in cols if c in ("design", "model") or any(r.get(c) not in (None, "") for r in rows)]
    head = "".join(f"<th style='text-align:left;padding:4px 8px;white-space:nowrap;'>{c}</th>" for c in used)
    body = []
    for r in rows:
        if "status" in r and "iptm" not in r:
            body.append(
                f"<tr><td style='padding:4px 8px;'>{r.get('design','')}</td>"
                f"<td colspan='{len(used)-1}' style='padding:4px 8px;{MUTED}'>{r.get('status')}</td></tr>"
            )
            continue
        tds = "".join(
            f"<td style='padding:4px 8px;font-family:monospace;'>{r.get(c, '')}</td>" for c in used
        )
        body.append(f"<tr>{tds}</tr>")
    return (
        "<div style='overflow:auto;max-height:420px;'>"
        "<table style='border-collapse:collapse;font-size:13px;'>"
        f"<thead><tr style='border-bottom:1px solid #ccc;'>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def launch_ui(*, outputs_root: Optional[Path] = None) -> None:
    """Build and display the general Boltz prediction UI."""
    # Display relative to project root (e.g. "outputs"); resolve under BOLTZ_ROOT at run time.
    default_outputs = Path(outputs_root) if outputs_root is not None else Path("outputs")
    gpu_opts = _list_gpu_options()
    default_gpu = _prefer_freest_gpu(gpu_opts)
    boltz_default = _find_boltz()
    example_opts = _list_examples()

    # ---- Setup ----
    job_name = widgets.Text(
        value="boltz_job",
        description="Job name:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )
    outputs_root_w = widgets.Text(
        value=str(default_outputs),
        description="Outputs root:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )
    boltz_bin = widgets.Text(
        value=boltz_default,
        description="boltz cmd:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )
    cache_dir = widgets.Text(
        value=str(DEFAULT_CACHE),
        description="Cache:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )

    # ---- Input mode ----
    # Temporarily show Binder remodel only; uncomment other modes to restore.
    input_mode = widgets.ToggleButtons(
        options=[
            # ("YAML editor", "yaml"),
            # ("Entity builder", "builder"),
            ("Binder remodel", "remodel"),
            # ("Existing path", "path"),
        ],
        value="remodel",
        description="Input:",
        style={"description_width": "120px", "button_width": "140px"},
    )

    # YAML editor
    yaml_editor = widgets.Textarea(
        value=EXAMPLE_YAML,
        layout=widgets.Layout(width="95%", height="360px"),
        description="",
    )
    example_dd = widgets.Dropdown(
        options=example_opts,
        value="",
        description="Load example:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="60%"),
    )

    # Existing path
    existing_path = widgets.Text(
        value="",
        description="YAML/dir:",
        placeholder="/path/to/file.yaml  or  /path/to/yaml_dir",
        layout=widgets.Layout(width="85%"),
        style={"description_width": "120px"},
    )

    # Entity builder state
    built_entities: List[dict] = []
    entities_view = widgets.HTML(value=f'<span style="{MUTED}">No entities yet.</span>')
    ent_type = widgets.Dropdown(
        options=["protein", "dna", "rna", "ligand"],
        value="protein",
        description="Type:",
        style={"description_width": "120px"},
    )
    ent_id = widgets.Text(
        value="A",
        description="Chain id(s):",
        placeholder="A  or  A,B",
        layout=widgets.Layout(width="40%"),
        style={"description_width": "120px"},
    )
    ent_seq = widgets.Textarea(
        value="",
        description="Sequence:",
        layout=widgets.Layout(width="95%", height="80px"),
        style={"description_width": "120px"},
    )
    ent_smiles = widgets.Text(
        value="",
        description="SMILES:",
        layout=widgets.Layout(width="85%"),
        style={"description_width": "120px"},
    )
    ent_ccd = widgets.Text(
        value="",
        description="CCD:",
        layout=widgets.Layout(width="40%"),
        style={"description_width": "120px"},
    )
    ent_msa = widgets.Text(
        value="",
        description="MSA path:",
        placeholder="optional .a3m / .csv, or 'empty'",
        layout=widgets.Layout(width="85%"),
        style={"description_width": "120px"},
    )
    ent_cyclic = widgets.Checkbox(value=False, description="Cyclic polymer")
    ent_mods = widgets.Textarea(
        value="",
        description="Mods:",
        placeholder="one per line: 12 MSE",
        layout=widgets.Layout(width="85%", height="60px"),
        style={"description_width": "120px"},
    )
    btn_add_ent = widgets.Button(description="Add entity", button_style="primary", icon="plus")
    btn_clear_ent = widgets.Button(description="Clear entities", icon="trash")
    btn_sync_yaml = widgets.Button(description="Sync → YAML editor", icon="exchange")

    constraints_yaml = widgets.Textarea(
        value="",
        description="Constraints:",
        placeholder="YAML list, e.g.\n- pocket:\n    binder: B\n    contacts: [[A, 10], [A, 20]]",
        layout=widgets.Layout(width="95%", height="100px"),
        style={"description_width": "120px"},
    )
    templates_yaml = widgets.Textarea(
        value="",
        description="Templates:",
        placeholder="YAML list, e.g.\n- cif: /path/to/template.cif\n  chain_id: [A]\n  force: false",
        layout=widgets.Layout(width="95%", height="100px"),
        style={"description_width": "120px"},
    )
    affinity_binder = widgets.Text(
        value="",
        description="Affinity binder:",
        placeholder="ligand chain id (Boltz-2 only), e.g. B",
        layout=widgets.Layout(width="50%"),
        style={"description_width": "120px"},
    )

    # Binder remodel — form UI (loads/saves JSON under add_ons/configs)
    BINDERS_PER_PAGE = 5
    binders_data: List[Dict[str, str]] = [
        {"name": "binder_1", "sequence": "REPLACE_WITH_BINDER_SEQUENCE"}
    ]
    binder_page = {"i": 0}
    _form_busy = {"v": False}

    def _addon_config_options() -> List[Tuple[str, str]]:
        opts = [("— select saved config —", "")]
        for p in list_config_files():
            opts.append((p.name, _rel_to_root(p)))
        return opts

    example_rel = EXAMPLE_CONFIG_REL if (BOLTZ_ROOT / EXAMPLE_CONFIG_REL).is_file() else ""
    addon_opts = _addon_config_options()
    if example_rel and example_rel not in {v for _, v in addon_opts}:
        example_rel = ""
    default_addon_value = example_rel if example_rel else ""

    addon_config_dd = widgets.Dropdown(
        options=addon_opts,
        value=default_addon_value,
        description="Saved config:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="70%"),
    )
    addon_new_name = widgets.Text(
        value="",
        description="File name:",
        placeholder="e.g. my_target_remodel",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )

    remodel_job = widgets.Text(
        value="example_binder_remodel",
        description="Job name:",
        placeholder="e.g. IFIT5_cropped_remodel",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )
    remodel_target_name = widgets.Text(
        value="Target",
        description="Target name:",
        placeholder="e.g. IFIT5_cropped",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "120px"},
    )
    remodel_target_id = widgets.Text(
        value="A",
        description="Target id:",
        layout=widgets.Layout(width="30%"),
        style={"description_width": "120px"},
    )
    remodel_binder_id = widgets.Text(
        value="B",
        description="Binder id:",
        layout=widgets.Layout(width="30%"),
        style={"description_width": "120px"},
    )
    remodel_target_seq = widgets.Textarea(
        value="REPLACE_WITH_TARGET_SEQUENCE",
        description="Target seq:",
        placeholder="Paste protein sequence here…",
        layout=widgets.Layout(width="95%", height="90px"),
        style={"description_width": "120px"},
    )
    remodel_n_binders = widgets.BoundedIntText(
        value=1,
        min=1,
        max=500,
        description="# binders:",
        layout=widgets.Layout(width="30%"),
        style={"description_width": "120px"},
    )
    btn_apply_n_binders = widgets.Button(
        description="Update binder fields",
        icon="list",
        tooltip="Resize binder list to match # binders (keeps existing entries)",
    )
    binder_page_label = widgets.HTML(value="")
    btn_binder_prev = widgets.Button(description="Prev", icon="arrow-left", disabled=True)
    btn_binder_next = widgets.Button(description="Next", icon="arrow-right", disabled=True)
    binders_box = widgets.VBox(
        [],
        layout=widgets.Layout(
            width="95%",
            max_height="360px",
            overflow_y="auto",
            border="1px solid #d0d7de",
            padding="8px",
            margin="4px 0",
        ),
    )
    addon_summary = widgets.HTML(
        value=(
            f'<span style="{MUTED}">Load a saved config or edit the form, '
            "then Save config with a file name.</span>"
        )
    )
    btn_refresh_addon_list = widgets.Button(description="Refresh list", icon="refresh")
    btn_save_addon = widgets.Button(
        description="Save config",
        button_style="success",
        icon="save",
        tooltip=f"Validate names/sequences and save to {CONFIGS_DIR_REL}/",
    )

    page_name_widgets: List[widgets.Text] = []
    page_seq_widgets: List[widgets.Textarea] = []

    # Options
    model = widgets.Dropdown(
        options=[("Boltz-2", "boltz2"), ("Boltz-1", "boltz1")],
        value="boltz2",
        description="Model:",
        style={"description_width": "120px"},
    )
    gpu = widgets.Dropdown(
        options=gpu_opts,
        value=default_gpu if any(v == default_gpu for _, v in gpu_opts) else gpu_opts[0][1],
        description="GPU:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="85%"),
    )
    refresh_gpu_btn = widgets.Button(
        description="Refresh GPU list",
        icon="refresh",
        button_style="info",
        layout=widgets.Layout(width="180px"),
    )
    gpu_status = widgets.HTML(
        "<span style='color:#555;'>Sets <code>CUDA_VISIBLE_DEVICES</code> for the Boltz job. "
        "Memory values are point-in-time snapshots; use Refresh GPU list to update them.</span>"
    )
    nvidia_smi_panel = _readonly_textarea(_nvidia_smi_text(), height="360px")
    boltz_gpu_processes_panel = _readonly_textarea(_boltz_gpu_processes_text(), height="130px")
    nvidia_smi_heading = widgets.HTML("<b>nvidia-smi details for the current selection</b>")
    nvidia_smi_help = widgets.HTML(
        "<p><b>How to read nvidia-smi:</b> The report lists every GPU visible on this machine. "
        "<b>Memory-Usage</b> is current VRAM use / capacity; <b>GPU-Util</b> shows how busy the "
        "GPU is: 0% means it is mostly idle, while 100% means it is very busy. This is separate "
        "from memory usage. <b>Pwr-Usage/Cap</b> is current power draw / power limit, and "
        "<b>Perf</b> is the performance state (P0 is high performance). The "
        "<b>Processes</b> section shows programs currently using GPU memory.</p>"
    )
    boltz_gpu_processes_heading = widgets.HTML(
        "<b>Active Boltz jobs using GPU memory</b>"
    )
    boltz_gpu_processes_help = widgets.HTML(
        "<p><b>Active Boltz jobs:</b> This table shows only Boltz processes currently "
        "using GPU memory. <b>GPU</b> is the GPU number, <b>PID</b> identifies the running "
        "process, <b>User</b> is the account running it, <b>Process</b> is the program name, "
        "<b>CWD</b> shows the project folder it is running from, and <b>GPU Memory</b> "
        "shows how much memory that job is using. If the table is empty, no Boltz job is "
        "currently using a GPU.</p>"
    )
    gpu_refreshing = {"active": False}

    def refresh_gpu_dropdown(_=None) -> None:
        if gpu_refreshing["active"]:
            return
        gpu_refreshing["active"] = True
        try:
            opts = _list_gpu_options()
            cur = gpu.value
            gpu.options = opts
            values = [v for _, v in opts]
            gpu.value = cur if cur in values else _prefer_freest_gpu(opts)
        finally:
            gpu_refreshing["active"] = False

    def refresh_monitoring_panels(_=None) -> None:
        nvidia_smi_panel.value = _nvidia_smi_text()
        boltz_gpu_processes_panel.value = _boltz_gpu_processes_text()

    def on_gpu_selection_change(change) -> None:
        if change.get("name") != "value" or change.get("new") == change.get("old"):
            return
        refresh_gpu_dropdown()
        refresh_monitoring_panels()

    def on_refresh_gpu(_=None) -> None:
        refresh_gpu_dropdown()
        refresh_monitoring_panels()

    gpu.observe(on_gpu_selection_change, names="value")
    refresh_gpu_btn.on_click(on_refresh_gpu)

    diffusion_samples = widgets.IntSlider(
        value=5, min=1, max=25, step=1, description="Samples:", style={"description_width": "120px"}
    )
    recycling_steps = widgets.IntSlider(
        value=10, min=1, max=10, step=1, description="Recycling:", style={"description_width": "120px"}
    )
    sampling_steps = widgets.IntSlider(
        value=200, min=50, max=400, step=10, description="Sampling:", style={"description_width": "120px"}
    )
    output_format = widgets.Dropdown(
        options=["mmcif", "pdb"],
        value="mmcif",
        description="Format:",
        style={"description_width": "120px"},
    )
    seed = widgets.IntText(
        value=-1,
        description="Seed (-1=none):",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="30%"),
    )
    use_msa_server = widgets.Checkbox(value=True, description="Use MSA server (--use_msa_server)")
    use_potentials = widgets.Checkbox(value=True, description="Use potentials (--use_potentials)")
    override = widgets.Checkbox(value=True, description="Override existing (--override)")
    write_full_pae = widgets.Checkbox(value=True, description="Write full PAE (--write_full_pae)")

    # Run
    status = widgets.HTML(value=f'<span style="{MUTED}">Ready.</span>')
    log = widgets.Output(
        layout=widgets.Layout(width="95%", height="280px", border="1px solid #d0d7de", overflow="auto")
    )
    results_view = widgets.HTML(value=_results_html([]))
    preview = widgets.HTML(value="")

    btn_preview = widgets.Button(description="Preview inputs", icon="eye")
    btn_write = widgets.Button(description="Write inputs", button_style="primary", icon="save")
    btn_cmd = widgets.Button(description="Show command", icon="terminal")
    btn_run = widgets.Button(description="Run predict", button_style="success", icon="play")
    btn_results = widgets.Button(description="Refresh results", icon="table")
    abort_job_dd = widgets.Dropdown(
        options=[("(none)", "")],
        description="Running:",
        layout=widgets.Layout(width="55%"),
        style={"description_width": "70px"},
        disabled=True,
    )
    btn_refresh_jobs = widgets.Button(description="Refresh jobs", icon="refresh")
    btn_abort = widgets.Button(description="Abort", button_style="danger", icon="stop", disabled=True)

    # Panels per mode
    yaml_panel = widgets.VBox(
        [
            widgets.HTML(SCHEMA_HELP),
            example_dd,
            widgets.HTML(
                f"<span style='{MUTED}'>One YAML job, or multiple documents separated by <code>---</code> "
                "(each becomes its own .yaml file).</span>"
            ),
            yaml_editor,
        ]
    )
    polymer_fields = widgets.VBox([ent_seq, ent_msa, ent_cyclic, ent_mods])
    ligand_fields = widgets.VBox([ent_smiles, ent_ccd])
    builder_panel = widgets.VBox(
        [
            widgets.HTML(
                f"<span style='{MUTED}'>Add protein / DNA / RNA / ligand entities, then optional "
                "constraints, templates, and affinity. Sync to YAML editor anytime.</span>"
            ),
            ent_type,
            ent_id,
            polymer_fields,
            ligand_fields,
            widgets.HBox([btn_add_ent, btn_clear_ent, btn_sync_yaml]),
            entities_view,
            constraints_yaml,
            templates_yaml,
            affinity_binder,
        ]
    )
    remodel_panel = widgets.VBox(
        [
            widgets.HTML(
                f"<div style='background:#ede9fe;padding:10px 14px;border-radius:6px;margin:4px 0;'>"
                f"<b>Edit or create a binder remodel config</b><br/>"
                f"<span style='{MUTED}'>Pair one target with many binders — each binder becomes its own Boltz job. <br/>"
                f"<span style='{MUTED}'>Protein sequences supported only (no DNA, RNA, or ligands).</span></div>"
            ),
            widgets.HBox([addon_config_dd, btn_refresh_addon_list]),
            widgets.HTML(f"<b>Target</b>"),
            widgets.HTML(
                f"<div style='font-size:12px;color:#57606a;margin:0 0 8px 0;line-height:1.5;'>"
                f"<b>Tips for names</b> (job, target, binders)<br/>"
                f"• For target, use something like, e.g. <code>IFIT5_cropped</code> or <code>binder-1</code><br/>"
                f"• For binder(s), use exact name(s) from BindCraft output, e.g. <code>IFIT5_bindcraft_ewok-Binder_l145_s476456_mpnn4</code><br/>"
                f"• Start with a letter, no spaces, only <b><code>_</code></b> or <b><code>-</code></b> as separators, "
                f"up to {NAME_MAX_LEN} characters<br/>"
                f"<b style='display:inline-block;margin-top:6px;'>Tips for sequences</b><br/>"
                f"• Paste a protein sequence (standard amino acids only)<br/>"
                f"• Letters are uppercased automatically, no spaces, up to {SEQ_MAX_LEN} residues<br/>"
                f"• Replace any placeholder text before saving"
                f"</div>"
            ),
            remodel_job,
            remodel_target_name,
            widgets.HBox([remodel_target_id, remodel_binder_id]),
            remodel_target_seq,
            widgets.HTML(f"<b>Binders</b>"),
            widgets.HTML(
                f"<div style='font-size:12px;color:#57606a;margin:0 0 8px 0;line-height:1.5;'>"
                "Same naming and sequence constraints as above for every binder.<br/>"
                f"Set how many binders you need, then click <b>Update binder fields</b>. "
                f"If the list is long, use Prev/Next (shows {BINDERS_PER_PAGE} at a time) or scroll."
                f"</div>"
            ),
            widgets.HBox([remodel_n_binders, btn_apply_n_binders]),
            widgets.HBox([btn_binder_prev, binder_page_label, btn_binder_next]),
            binders_box,
            widgets.HTML(f"<b>Save</b>"),
            widgets.HTML(
                f"<div style='font-size:12px;color:#57606a;margin:0 0 4px 0;line-height:1.5;'>"
                f"Pick a file name (same naming rules applied)."
                f"Config files always saved under <code>{CONFIGS_DIR_REL}</code>."
                f"</div>"
            ),
            addon_new_name,
            btn_save_addon,
            addon_summary,
        ]
    )
    path_panel = widgets.VBox(
        [
            widgets.HTML(
                f"<span style='{MUTED}'>Point at an existing <code>.yaml</code> / <code>.fasta</code> "
                "file or a directory of inputs. Nothing is rewritten unless you use Write.</span>"
            ),
            existing_path,
        ]
    )
    input_body = widgets.VBox([yaml_panel])

    def _set_status(msg: str, ok: Optional[bool] = None) -> None:
        if ok is True:
            status.value = f'<span style="{OK}">{msg}</span>'
        elif ok is False:
            status.value = f'<span style="{ERR}">{msg}</span>'
        else:
            status.value = f'<span style="{MUTED}">{msg}</span>'

    def _paths() -> Tuple[Path, Path]:
        raw = Path(outputs_root_w.value.strip() or str(default_outputs)).expanduser()
        root = raw.resolve() if raw.is_absolute() else (BOLTZ_ROOT / raw).resolve()
        job = _sanitize_name(job_name.value) or "boltz_job"
        return root / job / "yaml", root / job / "out"

    def _refresh_entities_view() -> None:
        if not built_entities:
            entities_view.value = f'<span style="{MUTED}">No entities yet.</span>'
            return
        items = []
        for i, ent in enumerate(built_entities):
            etype = next(iter(ent))
            body = ent[etype]
            cid = body.get("id")
            extra = ""
            if "sequence" in body:
                extra = f"len={len(body['sequence'])}"
            elif "smiles" in body:
                extra = f"smiles={body['smiles'][:40]}"
            elif "ccd" in body:
                extra = f"ccd={body['ccd']}"
            items.append(f"<li><b>{i+1}. {etype}</b> id={cid} <span style='{MUTED}'>{extra}</span></li>")
        entities_view.value = "<ul style='margin:4px 0 0 18px;'>" + "".join(items) + "</ul>"

    def _on_ent_type(_=None) -> None:
        is_lig = ent_type.value == "ligand"
        polymer_fields.layout.display = "none" if is_lig else None
        ligand_fields.layout.display = None if is_lig else "none"
        ent_msa.layout.display = None if ent_type.value == "protein" else "none"

    def _on_mode(change=None) -> None:
        mode = input_mode.value if change is None else change["new"]
        mapping = {
            "yaml": yaml_panel,
            "builder": builder_panel,
            "remodel": remodel_panel,
            "path": path_panel,
        }
        input_body.children = [mapping[mode]]

    def _builder_document() -> dict:
        if not built_entities:
            raise ValueError("Add at least one entity in the builder.")
        doc: Dict[str, Any] = {"version": 1, "sequences": list(built_entities)}
        cons = _optional_yaml_section(constraints_yaml.value, "constraints")
        if cons:
            doc["constraints"] = cons
        tmpls = _optional_yaml_section(templates_yaml.value, "templates")
        if tmpls:
            doc["templates"] = tmpls
        aff = (affinity_binder.value or "").strip()
        if aff:
            doc["properties"] = [{"affinity": {"binder": aff}}]
        return doc

    def _documents_from_ui() -> Tuple[str, List[Tuple[str, dict]]]:
        """Return (mode, [(stem, doc), ...])."""
        mode = input_mode.value
        if mode == "yaml":
            docs = _parse_yaml_docs(yaml_editor.value)
            if len(docs) == 1:
                return mode, [("input", docs[0])]
            return mode, [(f"input_{i+1:02d}", d) for i, d in enumerate(docs)]
        if mode == "builder":
            return mode, [("input", _builder_document())]
        if mode == "remodel":
            config = _config_from_form()
            job, docs = binder_remodel_addon.expand(config)
            if job:
                job_name.value = job
            return mode, docs
        if mode == "path":
            raise ValueError("Existing-path mode uses the path directly (no docs to build).")
        raise ValueError(f"Unknown mode: {mode}")

    def _resolve_input_for_run(write: bool) -> Path:
        mode = input_mode.value
        yaml_dir, _ = _paths()
        if mode == "path":
            p = Path(existing_path.value.strip()).expanduser()
            if not p.exists():
                raise ValueError(f"Path does not exist: {p}")
            return p
        if not write and yaml_dir.is_dir() and any(yaml_dir.glob("*.yaml")):
            return yaml_dir
        # write
        _, docs = _documents_from_ui()
        yaml_dir.mkdir(parents=True, exist_ok=True)
        for old in yaml_dir.glob("*.yaml"):
            old.unlink()
        for stem, doc in docs:
            (yaml_dir / f"{stem}.yaml").write_text(_dump_yaml(doc))
        return yaml_dir

    def on_load_example(_=None) -> None:
        path = example_dd.value
        if not path:
            _set_status("Select an example first.", False)
            return
        text = Path(path).read_text()
        yaml_editor.value = text
        input_mode.value = "yaml"
        job_name.value = Path(path).stem
        _set_status(f"Loaded example {Path(path).name}", True)

    def on_example_select(change) -> None:
        if change.get("name") != "value" or not example_dd.value:
            return
        on_load_example()

    def on_add_ent(_=None) -> None:
        try:
            ent = _build_entity_dict(
                entity_type=ent_type.value,
                chain_id=_parse_id_field(ent_id.value),
                sequence=ent_seq.value,
                smiles=ent_smiles.value,
                ccd=ent_ccd.value,
                msa=ent_msa.value,
                cyclic=ent_cyclic.value,
                modifications_raw=ent_mods.value,
            )
            built_entities.append(ent)
            _refresh_entities_view()
            # bump default chain id
            cur = ent_id.value.strip().split(",")[0].strip().strip("[]")
            if len(cur) == 1 and cur.isalpha():
                ent_id.value = chr(ord(cur.upper()) + 1)
            _set_status(f"Added {next(iter(ent))} entity.", True)
        except Exception as exc:
            _set_status(str(exc), False)

    def on_clear_ent(_=None) -> None:
        built_entities.clear()
        _refresh_entities_view()
        ent_id.value = "A"
        _set_status("Cleared entities.")

    def on_sync_yaml(_=None) -> None:
        try:
            doc = _builder_document()
            yaml_editor.value = _dump_yaml(doc)
            input_mode.value = "yaml"
            _set_status("Synced builder → YAML editor.", True)
        except Exception as exc:
            _set_status(str(exc), False)

    def _flush_binder_page() -> None:
        """Write visible page widgets back into binders_data."""
        start = binder_page["i"] * BINDERS_PER_PAGE
        for j, (nw, sw) in enumerate(zip(page_name_widgets, page_seq_widgets)):
            idx = start + j
            if idx < len(binders_data):
                binders_data[idx]["name"] = (nw.value or "").strip() or f"binder_{idx + 1}"
                binders_data[idx]["sequence"] = (sw.value or "").upper()

    def _binder_page_count() -> int:
        n = max(1, len(binders_data))
        return max(1, (n + BINDERS_PER_PAGE - 1) // BINDERS_PER_PAGE)

    def _autocap_seq_widget(change) -> None:
        """Force sequence textareas to uppercase as the user types."""
        if _form_busy["v"] or change.get("name") != "value":
            return
        w = change["owner"]
        raw = change.get("new")
        if raw is None:
            return
        upper = str(raw).upper()
        if upper != raw:
            _form_busy["v"] = True
            try:
                w.value = upper
            finally:
                _form_busy["v"] = False

    def _render_binder_page() -> None:
        nonlocal page_name_widgets, page_seq_widgets
        _form_busy["v"] = True
        try:
            n_pages = _binder_page_count()
            binder_page["i"] = max(0, min(binder_page["i"], n_pages - 1))
            start = binder_page["i"] * BINDERS_PER_PAGE
            end = min(start + BINDERS_PER_PAGE, len(binders_data))
            page_name_widgets = []
            page_seq_widgets = []
            rows = []
            for idx in range(start, end):
                entry = binders_data[idx]
                nw = widgets.Text(
                    value=entry.get("name") or f"binder_{idx + 1}",
                    description=f"Binder {idx + 1}:",
                    placeholder="e.g. binder_1",
                    layout=widgets.Layout(width="95%"),
                    style={"description_width": "120px"},
                )
                sw = widgets.Textarea(
                    value=(entry.get("sequence") or "").upper(),
                    description="Sequence:",
                    placeholder="Paste protein sequence here…",
                    layout=widgets.Layout(width="95%", height="70px"),
                    style={"description_width": "120px"},
                )
                sw.observe(_autocap_seq_widget, names="value")
                page_name_widgets.append(nw)
                page_seq_widgets.append(sw)
                rows.append(
                    widgets.VBox(
                        [
                            widgets.HTML(
                                f"<span style='{MUTED}'>#{idx + 1} of {len(binders_data)}</span>"
                            ),
                            nw,
                            sw,
                        ],
                        layout=widgets.Layout(margin="0 0 10px 0"),
                    )
                )
            binders_box.children = tuple(rows) if rows else (
                widgets.HTML(f'<span style="{MUTED}">No binders.</span>'),
            )
            binder_page_label.value = (
                f"<span style='padding:0 10px;'>Page {binder_page['i'] + 1} / {n_pages} "
                f"({len(binders_data)} binders, {BINDERS_PER_PAGE}/page)</span>"
            )
            btn_binder_prev.disabled = binder_page["i"] <= 0
            btn_binder_next.disabled = binder_page["i"] >= n_pages - 1
        finally:
            _form_busy["v"] = False

    def _resize_binders(n: int) -> None:
        _flush_binder_page()
        n = max(1, int(n))
        while len(binders_data) < n:
            i = len(binders_data) + 1
            binders_data.append({"name": f"binder_{i}", "sequence": ""})
        while len(binders_data) > n:
            binders_data.pop()
        remodel_n_binders.value = n
        # Jump to last page if current page is out of range
        n_pages = _binder_page_count()
        if binder_page["i"] >= n_pages:
            binder_page["i"] = n_pages - 1
        _render_binder_page()

    def _config_from_form() -> dict:
        _flush_binder_page()
        binders = []
        for i, b in enumerate(binders_data, start=1):
            name = (b.get("name") or "").strip() or f"binder_{i}"
            seq = b.get("sequence") or ""
            binders.append({"name": name, "sequence": seq})
        return {
            "job_name": (remodel_job.value or "").strip() or "binder_remodel",
            "target": {
                "id": (remodel_target_id.value or "A").strip() or "A",
                "name": (remodel_target_name.value or "").strip() or "Target",
                "sequence": remodel_target_seq.value or "",
            },
            "binder_id": (remodel_binder_id.value or "B").strip() or "B",
            "binders": binders,
        }

    def _form_from_config(config: dict) -> None:
        """Populate form widgets from a binder_remodel config dict."""
        target = config.get("target") if isinstance(config.get("target"), dict) else {}
        remodel_job.value = str(config.get("job_name") or target.get("name") or "binder_remodel")
        remodel_target_name.value = str(target.get("name") or "Target")
        remodel_target_id.value = str(target.get("id") or "A")
        remodel_binder_id.value = str(config.get("binder_id") or "B")
        remodel_target_seq.value = str(target.get("sequence") or target.get("seq") or "").upper()

        raw = config.get("binders")
        parsed: List[Dict[str, str]] = []
        if isinstance(raw, dict):
            for name, seq in raw.items():
                parsed.append({"name": str(name), "sequence": str(seq or "").upper()})
        elif isinstance(raw, list):
            for i, item in enumerate(raw, start=1):
                if isinstance(item, dict):
                    parsed.append(
                        {
                            "name": str(item.get("name") or item.get("id") or f"binder_{i}"),
                            "sequence": str(item.get("sequence") or item.get("seq") or "").upper(),
                        }
                    )
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    parsed.append({"name": str(item[0]), "sequence": str(item[1]).upper()})
        if not parsed:
            parsed = [{"name": "binder_1", "sequence": ""}]

        binders_data.clear()
        binders_data.extend(parsed)
        remodel_n_binders.value = len(binders_data)
        binder_page["i"] = 0
        _render_binder_page()

        stem = _sanitize_name(Path(str(config.get("job_name") or remodel_job.value)).stem)
        if not (addon_new_name.value or "").strip():
            addon_new_name.value = stem
        job_name.value = remodel_job.value

    def on_apply_n_binders(_=None) -> None:
        try:
            _resize_binders(int(remodel_n_binders.value))
            _set_status(f"Binder fields updated ({len(binders_data)} binders).", True)
        except Exception as exc:
            _set_status(str(exc), False)

    def on_binder_prev(_=None) -> None:
        _flush_binder_page()
        if binder_page["i"] > 0:
            binder_page["i"] -= 1
            _render_binder_page()

    def on_binder_next(_=None) -> None:
        _flush_binder_page()
        if binder_page["i"] < _binder_page_count() - 1:
            binder_page["i"] += 1
            _render_binder_page()

    def on_refresh_addon_list(_=None) -> None:
        current = addon_config_dd.value
        opts = _addon_config_options()
        addon_config_dd.options = opts
        values = {v for _, v in opts}
        if current in values:
            addon_config_dd.value = current
        elif EXAMPLE_CONFIG_REL in values:
            addon_config_dd.value = EXAMPLE_CONFIG_REL
        else:
            addon_config_dd.value = ""
        _set_status(
            f"Found {len(opts) - 1} config file(s) in {CONFIGS_DIR_REL}",
            True,
        )

    def _load_addon_from_path(path_s: str) -> None:
        path = _resolve_under_root(path_s)
        if not path.is_file():
            raise ValueError(f"Config file not found: {_rel_to_root(path)}")
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object.")
        # Soft-validate so placeholder example still loads into the form
        try:
            job, docs = binder_remodel_addon.expand(config)
            n_docs = len(docs)
            summary_ok = True
        except Exception:
            job = str(config.get("job_name") or "binder_remodel")
            n_docs = len(config.get("binders") or []) or 0
            summary_ok = False
        _form_from_config(config)
        job_name.value = job if summary_ok else remodel_job.value
        rel = _rel_to_root(path)
        # Prefer file stem for "create new from this"
        addon_new_name.value = path.stem
        if rel in {v for _, v in addon_config_dd.options}:
            addon_config_dd.value = rel
        if summary_ok:
            addon_summary.value = (
                f'<span style="{OK}">{binder_remodel_addon.summary(config)}</span>'
                f'<br/><span style="{MUTED}">Loaded — will write {n_docs} JSON config file(s). '
                "Edit fields and Save with a new file name to create a new config.</span>"
            )
        else:
            addon_summary.value = (
                f'<span style="{MUTED}">Loaded template (fill sequences before save/run). '
                f"{len(binders_data)} binder slot(s).</span>"
            )

    def on_load_addon(_=None) -> bool:
        try:
            path_s = (addon_config_dd.value or "").strip()
            if not path_s:
                raise ValueError("Select a saved config from the dropdown.")
            _load_addon_from_path(path_s)
            _set_status(f"Loaded add-on config: {Path(path_s).name}", True)
            return True
        except Exception as exc:
            _set_status(str(exc), False)
            return False

    def on_addon_config_select(change) -> None:
        if change.get("name") != "value" or not addon_config_dd.value:
            return
        on_load_addon()

    def _validate_remodel_fields() -> Tuple[dict, str, List[Tuple[str, dict]]]:
        """Validate names + protein sequences; return (config, job, docs) or raise ValueError."""
        _flush_binder_page()
        errors: List[str] = []

        def _check_name(label: str, raw: str) -> str:
            name = (raw or "").strip()
            if not name:
                errors.append(f"{label}: name is required.")
                return ""
            if any(ch.isspace() for ch in name):
                errors.append(f"{label}: name cannot contain spaces.")
            if len(name) > NAME_MAX_LEN:
                errors.append(f"{label}: name max length is {NAME_MAX_LEN} characters.")
            if name[:1].isdigit():
                errors.append(f"{label}: name cannot start with a number.")
            if not NAME_RE.fullmatch(name):
                errors.append(
                    f"{label}: invalid name {name!r} "
                    "(must start with a letter; only letters, digits, underscore, hyphen)."
                )
            return name

        def _check_seq(label: str, raw: str) -> str:
            seq = (raw or "").upper()
            if any(ch.isspace() for ch in (raw or "")):
                errors.append(f"{label}: sequence cannot contain spaces.")
            # Keep only letters for further checks; reject other junk explicitly
            non_letter = sorted({c for c in seq if not c.isalpha() and not c.isspace()})
            if non_letter:
                errors.append(
                    f"{label}: sequence may contain only protein letters "
                    f"(found {''.join(non_letter)})."
                )
            letters = "".join(c for c in seq if c.isalpha())
            if not letters:
                errors.append(f"{label}: protein sequence is empty.")
                return ""
            if letters.startswith("REPLACEWITH") or "REPLACE" in letters:
                errors.append(f"{label}: replace placeholder sequence with a real protein sequence.")
            bad = sorted({c for c in letters if c not in PROTEIN_AA})
            if bad:
                errors.append(
                    f"{label}: invalid amino-acid letter(s) {''.join(bad)} "
                    f"(allowed: {PROTEIN_AA})."
                )
            if len(letters) > SEQ_MAX_LEN:
                errors.append(
                    f"{label}: sequence length {len(letters)} exceeds max {SEQ_MAX_LEN} residues."
                )
            return letters

        job = _check_name("Job name", remodel_job.value)
        tname = _check_name("Target name", remodel_target_name.value)
        _check_name("Target id", remodel_target_id.value)
        _check_name("Binder id", remodel_binder_id.value)
        tseq = _check_seq("Target seq", remodel_target_seq.value)

        if not binders_data:
            errors.append("Add at least one binder.")

        seen_names: Dict[str, int] = {}
        binders_out: List[Dict[str, str]] = []
        for i, b in enumerate(binders_data, start=1):
            bname = _check_name(f"Binder {i} name", b.get("name") or "")
            bseq = _check_seq(f"Binder {i} sequence", b.get("sequence") or "")
            if bname:
                key = bname.lower()
                if key in seen_names:
                    errors.append(
                        f"Binder {i} name {bname!r} duplicates binder {seen_names[key]}."
                    )
                else:
                    seen_names[key] = i
            binders_out.append({"name": bname or f"binder_{i}", "sequence": bseq})

        file_raw = (addon_new_name.value or "").strip() or job or "binder_remodel"
        file_stem = _sanitize_name(Path(file_raw).stem)
        if not file_stem:
            errors.append("File name is required.")
        elif not NAME_RE.fullmatch(file_stem):
            # File stem uses sanitize which may differ; still require letter start / length
            if file_stem[:1].isdigit():
                errors.append("File name cannot start with a number.")
            if len(file_stem) > NAME_MAX_LEN:
                errors.append(f"File name max length is {NAME_MAX_LEN} characters.")

        if errors:
            raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

        config = {
            "job_name": job,
            "target": {
                "id": (remodel_target_id.value or "A").strip() or "A",
                "name": tname,
                "sequence": tseq,
            },
            "binder_id": (remodel_binder_id.value or "B").strip() or "B",
            "binders": binders_out,
        }
        # Final schema check via expand
        out_job, docs = binder_remodel_addon.expand(config)
        return config, out_job, docs

    def on_save_addon(_=None) -> None:
        try:
            config, job, docs = _validate_remodel_fields()

            name_raw = (addon_new_name.value or "").strip() or job
            stem = _sanitize_name(Path(name_raw).stem)
            filename = f"{stem}.json"

            CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = CONFIGS_DIR / filename
            out_path.write_text(json.dumps(config, indent=2) + "\n")

            rel = _rel_to_root(out_path)
            addon_new_name.value = stem
            remodel_job.value = job
            job_name.value = job

            # Refresh list without re-triggering a full reload mid-save message
            opts = _addon_config_options()
            addon_config_dd.unobserve(on_addon_config_select, names="value")
            try:
                addon_config_dd.options = opts
                if rel in {v for _, v in opts}:
                    addon_config_dd.value = rel
            finally:
                addon_config_dd.observe(on_addon_config_select, names="value")

            addon_summary.value = (
                f'<span style="{OK}">Validation OK — saved {rel}</span>'
                f'<br/><span style="{MUTED}">{binder_remodel_addon.summary(config)} '
                f"— {len(docs)} YAML file(s).</span>"
            )
            _set_status(f"Saved remodel config: {rel}", True)
        except Exception as exc:
            msg = str(exc)
            # Keep multiline validation errors readable in HTML
            html_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_msg = html_msg.replace("\n", "<br/>")
            addon_summary.value = f'<span style="{ERR}">{html_msg}</span>'
            _set_status("Validation failed — config not saved.", False)

    def on_preview(_=None) -> None:
        try:
            mode = input_mode.value
            if mode == "path":
                p = Path(existing_path.value.strip()).expanduser()
                if not p.exists():
                    raise ValueError(f"Path does not exist: {p}")
                if p.is_dir():
                    files = sorted(list(p.glob("*.yaml")) + list(p.glob("*.fasta")))
                    preview.value = (
                        f"<pre style='font-size:12px;'>path mode → directory {p}\n"
                        + "\n".join(f.name for f in files)
                        + "</pre>"
                    )
                else:
                    preview.value = f"<pre style='font-size:12px;max-height:240px;overflow:auto;'>{p.read_text()[:4000]}</pre>"
                _set_status(f"Previewing {p}", True)
                return
            _, docs = _documents_from_ui()
            chunks = [f"# {stem}.yaml\n{_dump_yaml(doc)}" for stem, doc in docs]
            text = "\n---\n".join(chunks)
            if len(text) > 6000:
                text = text[:6000] + "\n... [truncated]"
            preview.value = f"<pre style='font-size:12px;max-height:240px;overflow:auto;'>{text}</pre>"
            _set_status(f"Preview: {len(docs)} input file(s).", True)
        except Exception as exc:
            preview.value = ""
            _set_status(str(exc), False)

    def on_write(_=None) -> None:
        try:
            if input_mode.value == "path":
                p = _resolve_input_for_run(write=False)
                _set_status(f"Path mode — using existing input: {p}", True)
                return
            inp = _resolve_input_for_run(write=True)
            n = len(list(inp.glob('*.yaml'))) if inp.is_dir() else 1
            _set_status(f"Wrote {n} input file(s) → {inp}", True)
            with log:
                print(f"[write] {inp}")
                if inp.is_dir():
                    for p in sorted(inp.glob("*.yaml")):
                        print(f"  - {p.name}")
        except Exception as exc:
            _set_status(str(exc), False)

    def current_cmd() -> Tuple[List[str], Dict[str, str], Path, Path]:
        inp = _resolve_input_for_run(write=input_mode.value != "path")
        _, out_dir = _paths()
        gpu_val = gpu.value
        env = os.environ.copy()
        accelerator, devices = "gpu", 1
        if not gpu_val:
            raise ValueError("Select a GPU device before running (no GPU selected).")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_val)

        seed_v = None if seed.value is None or int(seed.value) < 0 else int(seed.value)
        cmd = _build_cmd(
            boltz_bin.value.strip() or "boltz",
            inp,
            out_dir,
            model=model.value,
            use_msa_server=use_msa_server.value,
            use_potentials=use_potentials.value,
            diffusion_samples=diffusion_samples.value,
            recycling_steps=recycling_steps.value,
            sampling_steps=sampling_steps.value,
            override=override.value,
            output_format=output_format.value,
            cache=cache_dir.value.strip() or str(DEFAULT_CACHE),
            accelerator=accelerator,
            devices=devices,
            write_full_pae=write_full_pae.value,
            seed=seed_v,
        )
        return cmd, env, inp, out_dir

    def on_cmd(_=None) -> None:
        try:
            cmd, env, inp, out_dir = current_cmd()
            with log:
                print("[command]")
                print("input =", inp)
                print("out   =", out_dir)
                print("CUDA_VISIBLE_DEVICES=" + env.get("CUDA_VISIBLE_DEVICES", "<default>"))
                print(subprocess.list2cmdline(cmd))
            _set_status("Command printed in log.", True)
        except Exception as exc:
            _set_status(str(exc), False)

    def _job_key() -> str:
        return str(_paths()[1])

    def _refresh_abort_jobs(_=None) -> None:
        jobs = _list_running_boltz_jobs()
        if jobs:
            abort_job_dd.options = [(job["label"], job["session"]) for job in jobs]
            abort_job_dd.disabled = False
            btn_abort.disabled = False
        else:
            abort_job_dd.options = [("(none)", "")]
            abort_job_dd.value = ""
            abort_job_dd.disabled = True
            btn_abort.disabled = True

    def _log_job(session: str, msg: str, *, end: str = "\n") -> None:
        with log:
            print(f"[{session}] {msg}", end=end, flush=True)

    def on_run(_=None) -> None:
        if shutil.which("tmux") is None:
            _set_status("tmux not found on PATH — install tmux to run jobs.", False)
            return

        key = _job_key()
        try:
            cmd, env, inp, out_dir = current_cmd()
            out_dir.mkdir(parents=True, exist_ok=True)
            session = _tmux_session_name(out_dir)
            log_path = _job_log_path(out_dir)
            job_label = out_dir.parent.name
        except Exception as exc:
            _set_status(str(exc), False)
            return

        if _tmux_session_running(session):
            _set_status(
                f"Job '{job_label}' already running in {session}. "
                f"Change job name for a parallel run, or abort it below.",
                False,
            )
            _refresh_abort_jobs()
            return

        _set_status(
            f"Started {session} (job: {job_label}). "
            f"Change job name and Run again for parallel jobs. Attach: tmux attach -t {session}",
            True,
        )

        def worker() -> None:
            header = (
                f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"tmux session = {session}\n"
                f"input = {inp}\n"
                f"out   = {out_dir}\n"
                f"log   = {log_path}\n"
                f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<default>')}\n"
                f"{subprocess.list2cmdline(cmd)}\n"
                f"attach: tmux attach -t {session}\n"
                + "=" * 60 + "\n"
            )
            log_path.write_text(header)
            shell_script = _build_tmux_shell(cmd, env, log_path)
            _log_job(session, header, end="")
            try:
                _tmux_start(session, shell_script)
            except FileNotFoundError:
                _log_job(session, "[error] boltz not found. Install or set 'boltz cmd'.")
                _set_status(f"{session}: boltz not found — see log.", False)
                _refresh_abort_jobs()
                return
            except subprocess.CalledProcessError as exc:
                _log_job(session, f"[error] failed to start tmux session: {exc}")
                _set_status(f"{session}: failed to start tmux session.", False)
                _refresh_abort_jobs()
                return
            except Exception as exc:
                _log_job(session, f"[error] failed to start: {exc}")
                _set_status(f"{session}: {exc}", False)
                _refresh_abort_jobs()
                return

            _refresh_abort_jobs()
            _monitor_tmux_job(session, log_path, out_dir, key)

        threading.Thread(target=worker, daemon=True).start()

    def _monitor_tmux_job(session: str, log_path: Path, out_dir: Path, key: str) -> None:
        with JOBS_LOCK:
            RUNNING_JOBS[key] = {"session": session, "log_path": str(log_path), "started": time.time()}
        _refresh_abort_jobs()

        offset = len(log_path.read_text(errors="replace")) if log_path.is_file() else 0
        while _tmux_session_running(session):
            offset, chunk = _read_log_tail(log_path, offset)
            if chunk:
                _log_job(session, chunk, end="")
            time.sleep(0.5)

        offset, chunk = _read_log_tail(log_path, offset)
        if chunk:
            _log_job(session, chunk, end="")

        with JOBS_LOCK:
            RUNNING_JOBS.pop(key, None)
        _refresh_abort_jobs()

        rc = _parse_log_exit(log_path)
        if rc is None:
            _log_job(session, f"[done] session ended (exit unknown)  {time.strftime('%Y-%m-%d %H:%M:%S')}")
            _set_status(f"{session}: ended (exit unknown). See run.log.", False)
        elif rc == 0:
            _log_job(session, f"[done] exit={rc}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if _job_key() == key:
                rows = _collect_results(out_dir)
                results_view.value = _results_html(rows)
                _set_status(f"{session}: finished OK. {len(rows)} result row(s).", True)
            else:
                _set_status(f"{session}: finished OK.", True)
        else:
            _log_job(session, f"[done] exit={rc}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
            _set_status(f"{session}: finished with exit code {rc}. See {log_path}", False)

    def on_abort(_=None) -> None:
        session = abort_job_dd.value
        if not session:
            _set_status("No running job selected.", False)
            _refresh_abort_jobs()
            return
        if not _tmux_session_running(session):
            _set_status(f"{session} is no longer running.", False)
            _refresh_abort_jobs()
            return
        try:
            _tmux_stop(session)
            _log_job(session, "[abort] tmux kill-session")
            _set_status(f"Aborted {session}.", True)
        except Exception as exc:
            _set_status(f"Abort failed: {exc}", False)
        _refresh_abort_jobs()

    def on_results(_=None) -> None:
        _, out_dir = _paths()
        rows = _collect_results(out_dir)
        results_view.value = _results_html(rows)
        pred_dir = _predictions_dir(out_dir)
        _set_status(
            f"Loaded {len(rows)} result row(s) from {pred_dir or out_dir / 'predictions'}",
            True,
        )

    # wire events
    ent_type.observe(_on_ent_type, names="value")
    input_mode.observe(_on_mode, names="value")
    example_dd.observe(on_example_select, names="value")
    btn_add_ent.on_click(on_add_ent)
    btn_clear_ent.on_click(on_clear_ent)
    btn_sync_yaml.on_click(on_sync_yaml)
    addon_config_dd.observe(on_addon_config_select, names="value")
    btn_refresh_addon_list.on_click(on_refresh_addon_list)
    btn_apply_n_binders.on_click(on_apply_n_binders)
    btn_binder_prev.on_click(on_binder_prev)
    btn_binder_next.on_click(on_binder_next)
    remodel_target_seq.observe(_autocap_seq_widget, names="value")
    btn_save_addon.on_click(on_save_addon)
    btn_preview.on_click(on_preview)
    btn_write.on_click(on_write)
    btn_cmd.on_click(on_cmd)
    btn_run.on_click(on_run)
    btn_refresh_jobs.on_click(_refresh_abort_jobs)
    btn_abort.on_click(on_abort)
    btn_results.on_click(on_results)

    _on_ent_type()
    _on_mode()
    _render_binder_page()
    if default_addon_value:
        try:
            _load_addon_from_path(default_addon_value)
        except Exception as exc:
            _set_status(f"Could not load default config: {exc}", False)

    setup_tab = widgets.VBox(
        [
            _banner("Boltz Predict UI"),
            widgets.HTML(
                "<p style='margin:0 0 8px 0;'>Configure outputs directory for boltz prediction job.</p>"
            ),
            job_name,
            outputs_root_w,
            # boltz_bin,
            # cache_dir,
        ]
    )
    input_tab = widgets.VBox([_banner("Input"), input_mode, input_body, preview])
    opts_tab = widgets.VBox(
        [
            _banner("Prediction options"),
            model,
            gpu,
            widgets.HBox([refresh_gpu_btn]),
            gpu_status,
            nvidia_smi_heading,
            nvidia_smi_help,
            nvidia_smi_panel,
            boltz_gpu_processes_heading,
            boltz_gpu_processes_help,
            boltz_gpu_processes_panel,
            diffusion_samples,
            recycling_steps,
            sampling_steps,
            output_format,
            seed,
            use_msa_server,
            use_potentials,
            override,
            write_full_pae,
        ]
    )
    run_tab = widgets.VBox(
        [
            _banner("Run"),
            widgets.HBox([btn_preview, btn_write, btn_cmd, btn_run, btn_results]),
            status,
            widgets.HTML("<b>Log</b>"),
            log,
            widgets.HTML(
                f"<b>Running jobs</b> <span style='{MUTED}'>"
                "(parallel runs need different job names — select one to abort)</span>"
            ),
            widgets.HBox([abort_job_dd, btn_refresh_jobs, btn_abort]),
            widgets.HTML("<b>Results</b> <span style='color:#57606a;'>(sorted by iptm)</span>"),
            results_view,
        ]
    )

    # ---- Download / zip ----
    def _outputs_dir() -> Path:
        raw = Path(outputs_root_w.value.strip() or str(default_outputs)).expanduser()
        return raw.resolve() if raw.is_absolute() else (BOLTZ_ROOT / raw).resolve()

    zip_dropdown = widgets.Dropdown(
        options=_list_output_zip_options(_outputs_dir()),
        value="__all__",
        description="Zip source:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "100px"},
    )
    zip_name_w = widgets.Text(
        value="outputs.zip",
        description="Zip file:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "100px"},
    )
    refresh_zip_btn = widgets.Button(
        description="Refresh folders",
        icon="refresh",
        button_style="warning",
        layout=widgets.Layout(width="180px"),
    )
    zip_btn = widgets.Button(
        description="Create zip in project root",
        button_style="success",
        icon="file-archive-o",
        layout=widgets.Layout(width="100%", height="40px"),
    )
    zip_status = widgets.HTML("")
    zip_help = widgets.HTML(
        f"<p style='margin:0 0 8px 0;'>Choose the entire <code>outputs/</code> folder "
        f"or a single design folder. The archive is written to the project root "
        f"(parent of <code>notebooks/</code>): <code>{BOLTZ_ROOT}</code>.</p>"
    )

    def on_refresh_zip(_=None):
        prev = zip_dropdown.value
        opts = _list_output_zip_options(_outputs_dir())
        zip_dropdown.options = opts
        values = [v for _, v in opts]
        if prev in values:
            zip_dropdown.value = prev
        else:
            zip_dropdown.value = "__all__"

    def on_zip_source_change(change=None):
        val = zip_dropdown.value
        if val == "__all__":
            zip_name_w.value = "outputs.zip"
        elif val:
            zip_name_w.value = f"{val}.zip"

    def on_create_zip(_=None):
        try:
            dest = _zip_outputs(
                _outputs_dir(),
                zip_dropdown.value,
                zip_name_w.value,
                dest_root=BOLTZ_ROOT,
            )
            size_mb = dest.stat().st_size / (1024 * 1024)
            zip_status.value = (
                f'<span style="{OK}">Created {dest} ({size_mb:.2f} MB)</span>'
            )
        except Exception as e:
            zip_status.value = f'<span style="{ERR}">Zip failed: {e}</span>'

    refresh_zip_btn.on_click(on_refresh_zip)
    zip_btn.on_click(on_create_zip)
    zip_dropdown.observe(on_zip_source_change, names="value")

    download_tab = widgets.VBox(
        [
            _banner("Zip outputs for download"),
            zip_help,
            zip_dropdown,
            zip_name_w,
            widgets.HBox([refresh_zip_btn]),
            zip_btn,
            zip_status,
        ]
    )

    tabs = widgets.Tab(children=[setup_tab, input_tab, opts_tab, run_tab, download_tab])
    tabs.set_title(0, "Setup")
    tabs.set_title(1, "Input")
    tabs.set_title(2, "Options")
    tabs.set_title(3, "Run")
    tabs.set_title(4, "Download")
    display(tabs)
    _refresh_abort_jobs()
    _set_status(
        "UI ready. Run always starts a new job; use different job names for parallel runs. "
        "Jobs run in detached tmux sessions.",
        True,
    )


# Back-compat alias for older notebook import
def launch_remodel_ui() -> None:
    launch_ui()
