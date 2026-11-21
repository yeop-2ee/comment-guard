"""Vercel은 pyproject.toml의 dependencies로 함수 의존성을 설치합니다.
서빙에 필요한 것만 두고(함수가 처음 뜰 때 /tmp에 설치됨), 버전은 requirements.txt와 같아야 합니다."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_dependencies_are_pinned_subset_of_requirements():
    pins = {ln.lower() for ln in (ROOT / "requirements.txt").read_text().splitlines() if ln and not ln.startswith("#")}
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert all("==" in d for d in deps)
    assert {d.lower() for d in deps} <= pins
    names = {d.split("==")[0].lower() for d in deps}
    assert {"fastapi", "onnxruntime", "tokenizers", "vercel-blob"} <= names
    assert not names & {"scikit-learn", "scipy", "pandas", "torch"}  # 서빙에 필요 없는 큰 패키지
