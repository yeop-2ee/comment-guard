import csv
import json

import pytest

from commentguard import runlog


@pytest.fixture(autouse=True)
def tmp_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMENTGUARD_RUNS_DIR", str(tmp_path))
    return tmp_path


def _read(run):
    return json.loads((run.dir / "run.json").read_text(encoding="utf-8"))


def test_success_writes_run_json_and_index(tmp_runs, tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("abc", encoding="utf-8")
    with runlog.run_step("demo", config={"C": 1.0}, inputs=[inp]) as run:
        run.outputs.append("out/")
    rec = _read(run)
    assert rec["status"] == "success"
    assert rec["config"] == {"C": 1.0}
    assert rec["inputs"][str(inp)].startswith("sha256:")
    assert rec["finished_at"] is not None
    rows = list(csv.DictReader(open(tmp_runs / "index.csv", encoding="utf-8")))
    assert [r["status"] for r in rows] == ["success"]


def test_failure_is_recorded_and_reraised(tmp_runs):
    with pytest.raises(ValueError):
        with runlog.run_step("demo") as run:
            raise ValueError("boom")
    rec = _read(run)
    assert rec["status"] == "failed"
    assert "boom" in rec["error"]


def test_wrap_returns_nonzero_on_failed_command(tmp_runs):
    code = runlog.main(["wrap", "--step", "shell", "--", "sh", "-c", "echo hi; exit 3"])
    assert code == 3
    rows = list(csv.DictReader(open(tmp_runs / "index.csv", encoding="utf-8")))
    assert rows[-1]["status"] == "failed"
    run_dir = tmp_runs / rows[-1]["run_id"]
    assert (run_dir / "output.txt").read_text(encoding="utf-8").strip() == "hi"


def test_model_version_format():
    v = runlog.make_model_version("M1")
    model_id, date, commit = v.split("-")
    assert model_id == "m1" and len(date) == 8 and len(commit) in (7, 8)


def test_mask_hides_secrets_in_records(tmp_runs, monkeypatch):
    monkeypatch.setenv("COMMENTGUARD_MASK", "AbCdEf123456")
    with runlog.run_step("demo", config={"model_uri": "blob://store_AbCdEf123456/models/x/"}) as run:
        pass
    text = (run.dir / "run.json").read_text(encoding="utf-8")
    assert "AbCdEf123456" not in text and "************" in text
