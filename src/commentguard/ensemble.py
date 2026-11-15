"""ENS: 구성 모델의 확률을 같은 가중치로 평균하는 앙상블 (명세서 4.5장).

python -m commentguard.ensemble [구성 모델 폴더 ...]
    폴더를 주지 않으면 configs/train.yaml의 ens.members (models/LATEST_<ID>).
    구성 모델의 valid 확률을 평균해서 심각도 기준값을 다시 조정합니다. eval은 읽지 않습니다.
    → models/ens-<날짜>-<커밋>/model_card.json (가중치 파일 없음, 구성 모델 버전만 기록), models/LATEST_ENS

로드할 때는 같은 models/ 폴더(또는 같은 저장소 경로) 아래의 구성 모델 폴더를 읽습니다.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from commentguard import model_store
from commentguard.data import PROCESSED_DIR, read_tsv
from commentguard.runlog import REPO_ROOT, make_model_version, now_kst, rel, run_step, sha256_file
from commentguard.severity import (CLASSES, SeverityConfig, filter_rates, grade_macro_f1,
                                   scores_from_probs, tune_thresholds)

TRAIN_CFG = REPO_ROOT / "configs" / "train.yaml"
SEVERITY_CFG = REPO_ROOT / "configs" / "severity.yaml"


@dataclass
class EnsembleModel:
    """다른 모델과 같은 인터페이스 (predict_proba, version, severity, card)."""
    members: list[Any]
    card: dict[str, Any]

    @property
    def version(self) -> str:
        return self.card["model_version"]

    @property
    def severity(self) -> SeverityConfig:
        return SeverityConfig.from_dict(self.card["severity"])

    def predict_proba(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        texts = list(texts)
        outs = [m.predict_proba(texts) for m in self.members]
        return (np.mean([np.asarray(h, dtype=float) for h, _ in outs], axis=0),
                np.mean([np.asarray(b, dtype=float) for _, b in outs], axis=0))


def load_ensemble(model_dir: str | Path) -> EnsembleModel:
    """구성 모델은 같은 상위 폴더의 <버전>/ 에서 읽습니다. 이미 불러온 모델은 다시 읽지 않습니다."""
    model_dir = Path(model_dir)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    members = []
    for m in card["members"]:
        d = model_dir.parent / m["model_version"]
        if not (d / "model_card.json").is_file():
            raise FileNotFoundError(f"앙상블 구성 모델 없음: {d}")
        members.append(model_store.load_cached(d))
    return EnsembleModel(members, card)


def _macro_f1(y_true, y_pred, labels) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def main(argv: Sequence[str] = ()) -> int:
    cfg_all = yaml.safe_load(open(TRAIN_CFG, encoding="utf-8"))
    sev_cfg = yaml.safe_load(open(SEVERITY_CFG, encoding="utf-8"))
    fit_path, valid_path = PROCESSED_DIR / "train_fit.tsv", PROCESSED_DIR / "valid.tsv"
    try:
        dirs = [Path(a) for a in argv] or [model_store.latest_dir(k) for k in cfg_all["ens"]["members"]]
        with run_step("train", suffix="ens", config={"members": [rel(d) for d in dirs], "weights": "equal",
                                                    "severity_tuning": sev_cfg["tuning"]},
                      inputs=[valid_path, SEVERITY_CFG, *[d / "model_card.json" for d in dirs]]) as run:
            version = make_model_version("ens")
            run.model_version = version
            valid = read_tsv(valid_path)
            members = [model_store.load_cached(d) for d in dirs]
            model = EnsembleModel(members, {"model_version": version})
            hp, bp = model.predict_proba(list(valid["comments"]))
            f1_h = _macro_f1(valid["hate"], np.array(CLASSES)[hp.argmax(1)], CLASSES)
            f1_b = _macro_f1(valid["bias"], np.array(model_store.BIAS_CLASSES)[bp.argmax(1)], model_store.BIAS_CLASSES)
            scores = scores_from_probs(hp)
            tuning = tune_thresholds(scores, valid["hate"], sev_cfg["sensitivity"],
                                     max_false_block=sev_cfg["tuning"]["max_none_false_block_medium"],
                                     min_hate_block=sev_cfg["tuning"]["min_hate_block_medium"])
            severity = SeverityConfig(tuning["caution"], tuning["severe"], sev_cfg["sensitivity"])
            start = SeverityConfig.from_dict(sev_cfg)

            def member_info(m, d: Path) -> dict[str, Any]:
                mid = m.card.get("model_id", "m1")
                return {"model_id": mid, "model_version": m.version, "git_dirty": m.card.get("git_dirty"),
                        "valid_hate_macro_f1": (m.card.get("valid_metrics") or {}).get(f"{mid}_hate_macro_f1"),
                        "artifact_bytes": sum(p.stat().st_size for p in d.iterdir() if p.is_file())}

            infos = [member_info(m, d) for m, d in zip(members, dirs)]
            card = {
                "model_version": version, "model_id": "ens",
                "created_at": now_kst().isoformat(timespec="seconds"),
                "git_commit": run.git_commit,
                # 구성 모델 중 하나라도 dirty면 앙상블도 제출용으로 쓰지 않음
                "git_dirty": bool(run.git_dirty or any(i["git_dirty"] for i in infos)),
                "method": "구성 모델의 hate·bias 클래스 확률을 같은 가중치로 평균",
                "members": infos,
                "data": {"train_fit": sha256_file(fit_path), "valid": sha256_file(valid_path)},
                "severity": severity.to_dict(),
                "severity_tuning": {"selected": tuning, "start_30_60": {
                    "caution": start.caution, "severe": start.severe,
                    "grade_macro_f1": grade_macro_f1(scores, valid["hate"], start.caution, start.severe),
                    **filter_rates(scores, valid["hate"], start)["medium"]}},
                "valid_metrics": {"ens_hate_macro_f1": round(f1_h, 4), "ens_bias_macro_f1": round(f1_b, 4)},
                "artifact_bytes": sum(i["artifact_bytes"] for i in infos),
            }
            model_dir = model_store.MODELS_DIR / version
            model_dir.mkdir(parents=True, exist_ok=True)
            model_store.write_card(model_dir, card)
            (model_store.MODELS_DIR / "LATEST_ENS").write_text(version + "\n")
            shutil.copy(model_dir / "model_card.json", run.dir / "model_card.json")
            run.outputs += [f"models/{version}/", rel(run.dir / "model_card.json")]
            run.extra.update(valid_metrics=card["valid_metrics"], members=[i["model_version"] for i in infos],
                             severity=card["severity"]["thresholds"])
    except Exception as e:
        print(f"[ens] 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"model_version={version} members={[i['model_version'] for i in infos]} valid={card['valid_metrics']} severity={tuning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
