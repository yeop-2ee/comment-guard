#!/usr/bin/env bash
# 자원 정리와 정리 후 상태 재조회 (명세서 10.3장). 결과는 run 폴더의 cleanup_status.txt (저장소 ID는 가림).
#   bash scripts/cleanup.sh --keep-production   지난 배포만 지우고 프로덕션 배포·모델·프로젝트는 남김 (채점 기간용)
#   bash scripts/cleanup.sh --all               Blob 저장소(모델 포함)와 Vercel 프로젝트를 모두 지움
source "$(dirname "$0")/vercel_env.sh"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"
MODE="${1:---keep-production}"
OUT="${COMMENTGUARD_RUN_DIR:-$ROOT/.state}/cleanup_status.txt"
mkdir -p "$(dirname "$OUT")"
masked() { "$PY" -c 'import sys; from commentguard.runlog import mask; sys.stdout.write(mask(sys.stdin.read()))'; }

case "$MODE" in
  --keep-production)
    prod_id="$("$PY" -c 'import json; print(json.load(open(".state/last_deploy.json"))["deployment_id"])')"
    prod_url="$("$PY" -c 'import json; print(json.load(open(".state/last_deploy.json"))["deployment_url"])')"
    echo "[cleanup] 남김: 프로덕션 배포 $prod_id ($prod_url), Blob 모델, 프로젝트"
    for url in $(vc ls "$VERCEL_PROJECT" 2>&1 | grep -oE 'https://[a-z0-9-]+\.vercel\.app' | sort -u); do
      [[ "$url" == "$prod_url" ]] && continue
      echo "[cleanup] 지난 배포 삭제: $url"
      vc rm "$url" --confirm-name "$url" >/dev/null
    done
    rm -f .env.local  # 개발용 토큰 파일 (다시 필요하면 make vercel-setup)
    ;;
  --all)
    echo "[cleanup] Blob 저장소의 모델 삭제 후 저장소 삭제: $BLOB_STORE"
    vc blob empty-store --confirm-name "$BLOB_STORE_ID" || true
    vc blob delete-store "$BLOB_STORE_ID" --confirm-name "$BLOB_STORE_ID"
    echo "[cleanup] Vercel 프로젝트 삭제: $VERCEL_PROJECT (배포, 도메인, 환경 변수 포함)"
    vc project rm "$VERCEL_PROJECT" --confirm-name "$VERCEL_PROJECT"
    rm -rf .env.local .vercel
    ;;
  *) echo "사용법: $0 --keep-production | --all" >&2; exit 1 ;;
esac

{
  echo "# 정리 후 상태 재조회 ($MODE) $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo; echo "## vercel project ls"; vc project ls 2>&1 | grep -v -E '^Vercel CLI|Fetching' || true
  echo; echo "## vercel ls $VERCEL_PROJECT"; vc ls "$VERCEL_PROJECT" 2>&1 | grep -E 'Age|https://|No deployments|not found' || true
  echo; echo "## vercel blob list-stores"; vc blob list-stores 2>&1 | grep -v -E '^Vercel CLI|Fetching' || true
  if [[ "$MODE" == "--keep-production" ]]; then
    echo; echo "## vercel blob list"; vc blob list 2>&1 | grep -E 'Pathname|models/' || true
  fi
  echo; echo "## 프로덕션 도메인 ${PROD_URL:-없음}/health"
  [[ -n "${PROD_URL:-}" ]] && curl -s -m 90 -w ' HTTP %{http_code}\n' "$PROD_URL/health" || true
  echo; echo "## 로컬 토큰 파일"; ls -a | grep -E '^\.env\.local$|^\.vercel$' || echo "(.env.local 없음)"
} 2>&1 | masked | tee "$OUT"
