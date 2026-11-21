#!/usr/bin/env bash
# Vercel 배포 (명세서 6.2장): 새 배포를 프로덕션 도메인 없이 만들고(--skip-domain),
# 그 배포의 /health가 통과해야 vercel promote로 프로덕션 도메인을 옮깁니다.
# 실패하면 승격하지 않으므로 프로덕션 도메인은 이전 정상 배포를 계속 서비스합니다.
# MODEL_URI는 배포마다 --env로 지정합니다 (기본: 서빙 모델의 blob://models/<version>/).
source "$(dirname "$0")/vercel_env.sh"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"

VERSION="$(basename "$("$PY" -c 'from commentguard.model_store import serving_dir; print(serving_dir())')")"
MODEL_URI="${MODEL_URI:-blob://models/$VERSION/}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "[deploy] MODEL_URI=$MODEL_URI region=$REGION"

vc deploy --prod --skip-domain --yes --json --env MODEL_URI="$MODEL_URI" > "$TMP/deploy.json"
read -r DEPLOY_ID DEPLOY_URL < <("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); d=d.get("deployment", d); u=d["url"]; print(d["id"], u if u.startswith("http") else "https://"+u)' "$TMP/deploy.json")
echo "[deploy] 새 배포 $DEPLOY_ID $DEPLOY_URL"

# 새 배포의 /health 확인. 첫 요청에서 모델을 받으므로 시간이 걸림. 프로덕션 외 주소는 보호되어 vercel curl로 요청
code=000
for attempt in 1 2 3; do
  code="$(vc curl /health --deployment "$DEPLOY_URL" -- --silent --output "$TMP/health.json" --write-out '%{http_code}' --max-time 120 || true)"
  code="${code: -3}"
  echo "[deploy] /health 시도 $attempt: HTTP $code $(head -c 300 "$TMP/health.json" 2>/dev/null || true)"
  [[ "$code" == "200" ]] && break
  sleep 5
done
if [[ "$code" != "200" ]] || ! grep -q '"status":"ok"' "$TMP/health.json"; then
  echo "[deploy] 새 배포의 /health 실패 → 승격하지 않음. 프로덕션 도메인은 이전 배포를 계속 서비스합니다" >&2
  exit 1
fi

vc promote "$DEPLOY_URL" --yes
PROD="${PROD_URL:-}"
if [[ -z "$PROD" ]]; then  # 첫 배포: 프로덕션 도메인을 찾아 deploy.env에 적어 둠
  PROD="https://$(vc inspect "$DEPLOY_URL" 2>&1 | grep -oE '[a-z0-9-]+\.vercel\.app' | grep -v "$(echo "$DEPLOY_URL" | sed 's|https://||')" | awk '{print length, $0}' | sort -n | head -1 | cut -d' ' -f2)"
  sed -i.bak "s|^PROD_URL=.*|PROD_URL=$PROD|" configs/deploy.env && rm -f configs/deploy.env.bak
fi
mkdir -p .state
"$PY" - "$PROD" "$DEPLOY_ID" "$DEPLOY_URL" "$REGION" "$MODEL_URI" "$VERSION" <<'PYEOF'
import json, sys
from commentguard.runlog import now_kst
keys = ["url", "deployment_id", "deployment_url", "region", "model_uri", "model_version"]
d = dict(zip(keys, sys.argv[1:]), promoted_at=now_kst().isoformat(timespec="seconds"))
open(".state/last_deploy.json", "w").write(json.dumps(d, ensure_ascii=False, indent=2))
print(json.dumps(d, ensure_ascii=False, indent=2))
PYEOF
echo "[deploy] 프로덕션 승격 완료: $PROD"
