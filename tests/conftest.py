from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PSAI_FORCE_CPU", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PSAI_FORCE_CPU": "1", "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run([sys.executable, "-m", module, *args], cwd=str(REPO_ROOT),
                          env=env, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="session")
def generated_data(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dummy_data")
    proc = run_module("src.data.make_dummy_data", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def write_phase_videos(root: Path, lengths: dict[str, int], *, size: int = 32,
                       num_steps: int = 4, num_instruments: int = 3,
                       seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    videos = []
    for vid, n in lengths.items():
        vdir = root / vid
        vdir.mkdir(exist_ok=True)
        for fi in range(n):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(vdir / f"frame_{fi:06d}.jpg")
        # one contiguous run per step, so segmental metrics see real segments
        steps = [min(fi * num_steps // max(n, 1), num_steps - 1) for fi in range(n)]
        instr = (rng.random((n, num_instruments)) > 0.5).astype(int).tolist()
        videos.append({"video_id": vid, "frames_dir": vid, "fps": 1,
                       "steps": steps, "instruments": instr})
    (root / "videos.json").write_text(json.dumps(videos))
    return root
