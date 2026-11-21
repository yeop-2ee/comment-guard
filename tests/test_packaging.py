"""Vercel은 pyproject.toml의 dependencies로 함수 의존성을 설치하므로, requirements.txt와 같은 고정 버전이어야 합니다."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_dependencies_match_requirements():
    pins = [ln for ln in (ROOT / "requirements.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert sorted(deps, key=str.lower) == sorted(pins, key=str.lower)
    assert all("==" in d for d in deps)
