"""Shared branch runner.

Every ``branches/*/run.py`` is a thin shim over this module, which loads the
branch's ``branch_config.yaml`` as an override on the root ``config.yaml`` and
calls the one shared :func:`pipeline.run_pipeline`. No branch contains a copy of
any module's logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline            # noqa: E402
import utils               # noqa: E402

LOGGER = utils.get_logger()


def load_branch_overrides(branch_dir: Path) -> Dict[str, Any]:
    """Read ``branch_config.yaml`` from a branch directory."""
    path = Path(branch_dir) / "branch_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing branch config: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_branch_config(branch_dir: Path, args) -> Dict[str, Any]:
    """Root config + branch overrides + CLI flags, with paths made absolute."""
    overrides = load_branch_overrides(branch_dir)
    # A branch may carry its own `smoke:` block (e.g. branch E narrowing the set
    # of held-out families). Merge it in when smoke mode is active, after the
    # root smoke block has been applied.
    branch_smoke = overrides.pop("smoke", None)
    cfg = pipeline.cfg_from_args(args, str(ROOT / "config.yaml"), overrides)
    if branch_smoke and cfg["run"].get("smoke_test"):
        cfg = utils.deep_merge(cfg, branch_smoke)
    # Branch outputs are declared relative to the repo root.
    for key in ("output_dir", "interim_dir", "cache_dir", "model_dir"):
        value = cfg["run"].get(key)
        if value and not Path(value).is_absolute():
            cfg["run"][key] = str(ROOT / value)
    return cfg


def run_branch(branch_dir: Path, description: str,
               extra: Optional[Mapping[str, Any]] = None):
    """Parse the CLI, resolve the config and run the shared pipeline."""
    ap = pipeline.build_arg_parser(description)
    args = ap.parse_args()
    cfg = resolve_branch_config(Path(branch_dir), args)
    if extra:
        cfg = utils.deep_merge(cfg, extra)

    result = pipeline.run_pipeline(
        data_dir=args.data_dir, cfg=cfg,
        out_dir=cfg["run"]["output_dir"],
        tag=cfg["run"]["name"], make_plots=not args.no_plots,
    )
    print(f"\n=== {description} ===")
    print(result.metrics.to_string(index=False))
    return result
