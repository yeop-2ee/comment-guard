#!/usr/bin/env bash
# Vercel 프로젝트 연결과 Blob 저장소(비공개) 1회 준비 (명세서 6.2장). 이미 있으면 그대로 둡니다.
source "$(dirname "$0")/vercel_env.sh"

echo "[setup] 프로젝트 연결: $VERCEL_PROJECT"
vc link --yes --project "$VERCEL_PROJECT"

if vc blob list-stores 2>&1 | grep -q -- "$BLOB_STORE"; then
  echo "[setup] Blob 저장소 있음: $BLOB_STORE"
else
  echo "[setup] Blob 저장소 생성: $BLOB_STORE (비공개, $REGION)"
  vc blob create-store "$BLOB_STORE" --access private --region "$REGION" --yes
fi

# 로컬 CLI가 저장소를 쓸 때 필요한 개발용 변수(BLOB_STORE_ID, 단기 OIDC 토큰)를 .env.local로 받음 (Git 제외)
vc env pull .env.local --yes >/dev/null
store_id="$(grep -E '^BLOB_STORE_ID=' .env.local | cut -d= -f2- | tr -d '"')"
if [[ -z "$store_id" ]]; then
  echo "[setup] .env.local에 BLOB_STORE_ID가 없습니다. 저장소가 프로젝트에 연결됐는지 확인하세요" >&2
  exit 1
fi
if grep -q '^BLOB_STORE_ID=' configs/deploy.env; then
  sed -i.bak "s|^BLOB_STORE_ID=.*|BLOB_STORE_ID=$store_id|" configs/deploy.env && rm -f configs/deploy.env.bak
else
  echo "BLOB_STORE_ID=$store_id" >> configs/deploy.env
fi
echo "[setup] 완료. 저장소 ID는 configs/deploy.env에만 저장(기록에서는 가림)"
vc project ls 2>&1 | grep -E "$VERCEL_PROJECT|Project Name" || true
