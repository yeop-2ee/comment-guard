#!/usr/bin/env bash
# Korean HateSpeech Dataset을 고정 커밋에서 받아 data/raw/에 두고, MANIFEST.sha256과 대조합니다.
# 데이터는 저장소에 포함되어 있으므로(D3) 없는 파일만 받고, 해시 검사는 항상 합니다.
# FORCE=1 이면 있는 파일도 다시 받습니다.
set -euo pipefail

COMMIT=f8d05dce2b22007bb149e5139c0060c68ad8f94b
BASE="https://raw.githubusercontent.com/kocohub/korean-hate-speech/${COMMIT}"
RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

declare -a FILES=("labeled/train.tsv" "labeled/dev.tsv" "test.no_label.tsv")

for src in "${FILES[@]}"; do
  dst="${RAW_DIR}/$(basename "$src")"
  if [[ -f "$dst" && "${FORCE:-0}" != "1" ]]; then
    echo "있음: $(basename "$dst")"
    continue
  fi
  echo "다운로드: ${src}"
  curl -fsSL --retry 3 "${BASE}/${src}" -o "${dst}.tmp"
  mv "${dst}.tmp" "${dst}"
done

echo "해시 검사 (MANIFEST.sha256)"
cd "$RAW_DIR"
if ! shasum -a 256 -c MANIFEST.sha256; then
  echo "오류: 데이터 해시가 매니페스트와 다릅니다. 다음 단계로 진행하지 않습니다." >&2
  exit 1
fi
