"""모든 파이프라인 단계가 공유하는 실행 기록.

각 실행은 runs/<run_id>/run.json 하나와 runs/index.csv 한 줄을 남깁니다.
실패한 실행도 status="failed"로 기록하고, 예외는 다시 던져 다음 단계가 진행되지 않게 합니다.

파이썬 코드에서:
    with run_step("train", config=cfg, inputs=[path]) as run:
        ...
        run.model_version = "m1-20261115-a1b2c3d"
        run.outputs.append("models/m1-20261115-a1b2c3d/")

셸 스크립트를 감쌀 때:
    python -m commentguard.runlog wrap --step deploy -- bash scripts/deploy_vercel.sh
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

KST = timezone(timedelta(hours=9))
REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_FIELDS = ["run_id", "step", "status", "started_at", "finished_at",
                "git_commit", "git_dirty", "model_version", "error"]


def runs_dir() -> Path:
    # 테스트에서 실제 runs/를 더럽히지 않도록 환경변수로 바꿀 수 있게 함
    return Path(os.environ.get("COMMENTGUARD_RUNS_DIR", REPO_ROOT / "runs"))


def mask(text: str) -> str:
    """COMMENTGUARD_MASK(쉼표 구분)에 든 문자열(예: Blob 저장소 ID)을 기록에서 가림."""
    for secret in filter(None, (x.strip() for x in os.environ.get("COMMENTGUARD_MASK", "").split(","))):
        text = text.replace(secret, "*" * len(secret))
    return text


def rel(path: str | Path) -> str:
    """저장소 기준 상대 경로. 저장소 밖이면(테스트용 임시 폴더 등) 절대 경로."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def now_kst() -> datetime:
    return datetime.now(KST)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def git_info() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                 text=True, check=True)
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _git("rev-parse", "HEAD")
    # runs/는 실행 기록 자체라서, 앞 단계가 남긴 기록 때문에 dirty로 잡히지 않게 제외
    status = _git("status", "--porcelain", "--", ".", ":(exclude)runs")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def make_model_version(model_id: str, when: datetime | None = None) -> str:
    """<모델ID>-<YYYYMMDD>-<git 커밋 7자리>. 커밋이 없으면 'nocommit'."""
    when = when or now_kst()
    commit = git_info()["git_commit"]
    return f"{model_id.lower()}-{when:%Y%m%d}-{commit[:7] if commit else 'nocommit'}"


@dataclass
class RunRecord:
    run_id: str
    step: str
    started_at: str
    git_commit: str | None
    git_dirty: bool | None
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    model_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    finished_at: str | None = None
    status: str = "running"
    error: str | None = None

    @property
    def dir(self) -> Path:
        return runs_dir() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "model_version": self.model_version,
            "inputs": self.inputs,
            "config": self.config,
            "outputs": self.outputs,
            "extra": self.extra,
            "error": self.error,
            "env": {"python": platform.python_version(), "platform": platform.platform()},
        }


def _new_run_id(step: str, suffix: str | None) -> str:
    base = f"{now_kst():%Y%m%d-%H%M%S}-{step}" + (f"-{suffix}" if suffix else "")
    run_id, n = base, 1
    while (runs_dir() / run_id).exists():  # 같은 초에 두 번 실행되는 경우
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def _write(record: RunRecord) -> None:
    record.dir.mkdir(parents=True, exist_ok=True)
    with open(record.dir / "run.json", "w", encoding="utf-8") as f:
        f.write(mask(json.dumps(record.to_dict(), ensure_ascii=False, indent=2)))


def _append_index(record: RunRecord) -> None:
    index = runs_dir() / "index.csv"
    new = not index.exists()
    row = {k: record.to_dict()[k] for k in INDEX_FIELDS}
    if row["error"]:
        row["error"] = mask(row["error"].splitlines()[0][:200])
    with open(index, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


@contextmanager
def run_step(step: str, config: dict[str, Any] | None = None,
             inputs: list[str | Path] | None = None, model_version: str | None = None,
             suffix: str | None = None) -> Iterator[RunRecord]:
    record = RunRecord(
        run_id=_new_run_id(step, suffix),
        step=step,
        started_at=now_kst().isoformat(timespec="seconds"),
        config=dict(config or {}),
        model_version=model_version,
        **git_info(),
    )
    for p in inputs or []:
        record.inputs[str(p)] = sha256_file(p) if Path(p).is_file() else "missing"
    _write(record)  # 중간에 프로세스가 죽어도 "running" 기록이 남게 먼저 씀
    try:
        yield record
    except BaseException as e:
        record.status = "failed"
        record.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
        raise
    else:
        record.status = "success"
    finally:
        record.finished_at = now_kst().isoformat(timespec="seconds")
        _write(record)
        _append_index(record)


def _wrap(args: argparse.Namespace) -> int:
    """셸 명령을 실행하고 종료 코드로 성공/실패를 기록. 출력은 run 폴더의 output.txt에 저장."""
    config = json.loads(args.config) if args.config else {}
    code = 1
    try:
        with run_step(args.step, config=config, inputs=args.input,
                      model_version=args.model_version, suffix=args.suffix) as run:
            log_path = run.dir / "output.txt"
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.Popen(args.cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
                assert proc.stdout is not None
                for line in proc.stdout:  # 화면과 파일에 동시에 출력 (파일에는 가린 값)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(mask(line))
                code = proc.wait()
            run.outputs.append(rel(log_path))
            run.extra["command"] = args.cmd
            run.extra["exit_code"] = code
            if code != 0:
                raise RuntimeError(f"command exited with code {code}")
    except Exception as e:  # 명령을 찾지 못한 경우 등도 실패로 기록된 뒤 여기로 옴
        print(f"[runlog] {args.step} failed: {e}", file=sys.stderr)
        return code or 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m commentguard.runlog")
    sub = parser.add_subparsers(dest="cmd_name", required=True)
    w = sub.add_parser("wrap", help="셸 명령을 실행하고 run.json을 남깁니다")
    w.add_argument("--step", required=True)
    w.add_argument("--suffix")
    w.add_argument("--model-version")
    w.add_argument("--config", help="JSON 문자열")
    w.add_argument("--input", action="append", default=[])
    w.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("실행할 명령이 없습니다 (-- 뒤에 적으세요)")
    return _wrap(args)


if __name__ == "__main__":
    sys.exit(main())
