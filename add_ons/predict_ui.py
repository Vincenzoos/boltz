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

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
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


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name.strip("._") or "item"


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


def _detect_gpus() -> List[Tuple[str, str]]:
    opts = [("auto (CUDA default)", ""), ("CPU", "cpu")]
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, name, free = parts[0], parts[1], parts[2]
                opts.append((f"GPU {idx}: {name} ({free} MiB free)", idx))
    except Exception:
        pass
    return opts


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
    gpu_opts = _detect_gpus()
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

    # Binder remodel (JSON-driven add-on)
    def _addon_config_options() -> List[Tuple[str, str]]:
        opts = [("— select saved config —", "")]
        for p in list_config_files():
            opts.append((p.name, str(p.resolve())))
        return opts


    addon_config_dd = widgets.Dropdown(
        options=_addon_config_options(),
        value="",
        description="Saved config:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="70%"),
    )
    addon_config_path = widgets.Text(
        value=str(CONFIGS_DIR),
        description="Config path:",
        placeholder="/path/to/binder_remodel.json",
        layout=widgets.Layout(width="85%"),
        style={"description_width": "120px"},
    )
    addon_json = widgets.Textarea(
        value=EXAMPLE_BINDER_REMODEL_JSON,
        description="JSON:",
        layout=widgets.Layout(width="95%", height="280px"),
        style={"description_width": "120px"},
    )
    addon_summary = widgets.HTML(value=f'<span style="{MUTED}">Select a saved config or paste a binder_remodel JSON config.</span>')
    btn_refresh_addon_list = widgets.Button(description="Refresh list", icon="refresh")

    # Options
    model = widgets.Dropdown(
        options=[("Boltz-2", "boltz2"), ("Boltz-1", "boltz1")],
        value="boltz2",
        description="Model:",
        style={"description_width": "120px"},
    )
    gpu = widgets.Dropdown(
        options=gpu_opts,
        value=gpu_opts[0][1],
        description="Device:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="60%"),
    )
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
            widgets.HTML(binder_remodel_addon.SCHEMA_HELP),
            widgets.HTML(
                f"<span style='{MUTED}'>Add-on only — expands JSON into native Boltz YAMLs "
                f"(one complex per binder). Configs live in <code>{CONFIGS_DIR}</code>.</span>"
            ),
            widgets.HBox([addon_config_dd, btn_refresh_addon_list]),
            addon_config_path,
            addon_json,
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
            config = json.loads(addon_json.value)
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

    def on_refresh_addon_list(_=None) -> None:
        addon_config_dd.options = _addon_config_options()
        _set_status(f"Found {len(addon_config_dd.options) - 1} config file(s) in {CONFIGS_DIR}", True)

    def _load_addon_from_path(path_s: str) -> None:
        path = Path(path_s.strip()).expanduser()
        if not path.is_file():
            raise ValueError(f"Config file not found: {path}")
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object.")
        job, docs = binder_remodel_addon.expand(config)
        addon_json.value = json.dumps(config, indent=2)
        job_name.value = job
        addon_config_path.value = str(path.resolve())
        addon_summary.value = (
            f'<span style="{OK}">{binder_remodel_addon.summary(config)}</span>'
            f'<br/><span style="{MUTED}">Will write {len(docs)} YAML file(s).</span>'
        )

    def on_load_addon(_=None) -> bool:
        try:
            path_s = addon_config_dd.value or addon_config_path.value
            if addon_config_dd.value:
                path_s = addon_config_dd.value
            elif Path(addon_config_path.value.strip()).expanduser().is_file():
                path_s = addon_config_path.value
            else:
                raise ValueError("Select a saved config or set Config path to a .json file.")
            _load_addon_from_path(path_s)
            _set_status(f"Loaded add-on config: {Path(path_s).name}", True)
            return True
        except Exception as exc:
            _set_status(str(exc), False)
            return False

    def on_addon_config_select(change) -> None:
        if change.get("name") != "value" or not addon_config_dd.value:
            return
        if on_load_addon():
            on_validate_addon()

    def on_validate_addon(_=None) -> None:
        try:
            config = json.loads(addon_json.value)
            job, docs = binder_remodel_addon.expand(config)
            job_name.value = job
            addon_summary.value = (
                f'<span style="{OK}">{binder_remodel_addon.summary(config)}</span>'
                f'<br/><span style="{MUTED}">OK — {len(docs)} YAML file(s) will be written.</span>'
            )
            _set_status(f"Valid binder_remodel config ({len(docs)} binders).", True)
        except Exception as exc:
            addon_summary.value = f'<span style="{ERR}">{exc}</span>'
            _set_status(str(exc), False)

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
        if gpu_val == "cpu":
            accelerator, devices = "cpu", 1
        else:
            accelerator, devices = "gpu", 1
            if gpu_val:
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
    btn_preview.on_click(on_preview)
    btn_write.on_click(on_write)
    btn_cmd.on_click(on_cmd)
    btn_run.on_click(on_run)
    btn_refresh_jobs.on_click(_refresh_abort_jobs)
    btn_abort.on_click(on_abort)
    btn_results.on_click(on_results)

    _on_ent_type()
    _on_mode()

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

    tabs = widgets.Tab(children=[setup_tab, input_tab, opts_tab, run_tab])
    tabs.set_title(0, "Setup")
    tabs.set_title(1, "Input")
    tabs.set_title(2, "Options")
    tabs.set_title(3, "Run")
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
