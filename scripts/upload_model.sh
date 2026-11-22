#!/usr/bin/env bash
# 서빙 모델을 올립니다 (명세서 6.3장). MODEL_STORE=release(기본): 이 저장소의 GitHub Release `model-<version>`,
# MODEL_STORE=blob: Vercel Blob 비공개 저장소 models/<version>/ (Hobby 한도 초과로 저장소가 정지된 뒤 release로 바꿈).
# dirty 모델과 ONNX 동등성 확인을 통과하지 못한 모델은 올리지 않습니다.
# 함께 올리는 serving.json에 파일별 sha256과 크기를 적어, 함수가 받은 뒤 확인합니다.
source "$(dirname "$0")/vercel_env.sh"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"

MODEL_DIR="${MODEL_DIR:-$("$PY" -c 'from commentguard.model_store import serving_dir; print(serving_dir())')}"
VERSION="$(basename "$MODEL_DIR")"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

"$PY" - "$MODEL_DIR" "$STAGE" <<'PYEOF'
import hashlib, json, shutil, sys
from pathlib import Path
from commentguard.onnx_model import SERVING_FILES
src, stage = Path(sys.argv[1]), Path(sys.argv[2])
card = json.loads((src / "model_card.json").read_text(encoding="utf-8"))
if card.get("git_dirty"):
    sys.exit(f"dirty 상태로 만든 모델은 올리지 않습니다: {card['model_version']}")
if card.get("model_id", "m1")[:2] in ("m2", "m3", "m4"):
    check = src / "onnx_check.json"
    if not check.is_file() or not json.loads(check.read_text())["passed"]:
        sys.exit("ONNX 변환과 동등성 확인이 먼저 필요합니다: python -m commentguard.onnx_export")
    names = list(SERVING_FILES)
else:
    names = ["model_card.json", "model.joblib"]
files = {}
for n in names:
    shutil.copy(src / n, stage / n)
    data = (stage / n).read_bytes()
    files[n] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
manifest = {"model_version": card["model_version"], "files": files}
(stage / "serving.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PYEOF

if [[ "${MODEL_STORE:-release}" == "release" ]]; then
  : "${GITHUB_REPO:?deploy.env에 GITHUB_REPO(예: yeop-2ee/comment-guard)가 필요합니다}"
  TAG="model-$VERSION"
  cat > "$STAGE/NOTES.md" <<NOTES
댓글 가드 서빙 모델 \`$VERSION\` (ONNX). Vercel 함수가 시작할 때 이 파일들을 받아 \`serving.json\`의 sha256으로 확인합니다.

- 기반 모델: beomi/KcELECTRA-base (MIT), 리비전 681201a
- 학습 데이터: Korean HateSpeech Dataset (BEEP!), CC BY-SA 4.0 — https://github.com/kocohub/korean-hate-speech
- 이 가중치는 위 데이터로 미세조정한 결과라 CC BY-SA 4.0 조건(출처 표시, 같은 조건으로 공유)을 따릅니다.
- 법률 판단용이 아닙니다. 평가와 한계는 저장소의 docs/EVALUATION.md를 보세요.
NOTES
  files=("$STAGE"/model_card.json "$STAGE"/serving.json)
  for f in "$STAGE"/*; do case "$(basename "$f")" in serving.json|model_card.json|NOTES.md) ;; *) files+=("$f") ;; esac; done
  if gh release view "$TAG" -R "$GITHUB_REPO" >/dev/null 2>&1; then
    echo "[upload] Release $TAG 있음 → 파일 교체"
    for old in $(gh release view "$TAG" -R "$GITHUB_REPO" --json assets --jq '.assets[].name'); do
      [[ -f "$STAGE/$old" ]] || { echo "[upload] 목록에 없는 파일 삭제: $old"; gh release delete-asset "$TAG" "$old" -R "$GITHUB_REPO" --yes; }
    done
    gh release upload "$TAG" "${files[@]}" -R "$GITHUB_REPO" --clobber
  else
    echo "[upload] Release $TAG 생성"
    gh release create "$TAG" "${files[@]}" -R "$GITHUB_REPO" --title "모델 $VERSION" --notes-file "$STAGE/NOTES.md" --latest=false
  fi
  echo "[upload] 확인"
  gh release view "$TAG" -R "$GITHUB_REPO" --json assets --jq '.assets[] | "\(.name) \(.size)"'
  echo "MODEL_URI=https://github.com/$GITHUB_REPO/releases/download/$TAG/"
  exit 0
fi

# 같은 경로에 이번 목록에 없는 파일(예: 예전 형식의 model.onnx)이 있으면 먼저 지움 (Hobby 저장 한도 1GB)
for old in $(vc blob list 2>&1 | grep -oE "models/$VERSION/[^ ]+" | sort -u); do
  [[ -f "$STAGE/$(basename "$old")" ]] && continue
  echo "[upload] 목록에 없는 파일 삭제: $old"
  vc blob delete "$old" --confirm-name "$old" >/dev/null
done

# serving.json은 마지막에 올려서, 파일이 다 올라가기 전에는 이 경로가 쓰이지 않게 함
for f in "$STAGE"/*; do
  name="$(basename "$f")"
  [[ "$name" == "serving.json" ]] && continue
  echo "[upload] models/$VERSION/$name ($(du -h "$f" | cut -f1))"
  vc blob put "$f" --access private --pathname "models/$VERSION/$name" --force >/dev/null
done
vc blob put "$STAGE/serving.json" --access private --pathname "models/$VERSION/serving.json" --force >/dev/null
echo "[upload] 저장소 확인"
vc blob list 2>&1 | grep -E "models/$VERSION/" || { echo "업로드한 파일이 목록에 없습니다" >&2; exit 1; }
echo "MODEL_URI=blob://models/$VERSION/"
