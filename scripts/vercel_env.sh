# 배포 스크립트 공통: configs/deploy.env를 읽고 Vercel CLI 로그인을 확인합니다. (source로 불러 씀)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/configs/deploy.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "configs/deploy.env 없음. configs/deploy.env.example을 복사해서 채우세요" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${VERCEL_PROJECT:?deploy.env에 VERCEL_PROJECT가 필요합니다}"
: "${REGION:?deploy.env에 REGION이 필요합니다}"
: "${BLOB_STORE:?deploy.env에 BLOB_STORE가 필요합니다}"
VERCEL_CLI="${VERCEL_CLI:-vercel@66.6.0}"
SCOPE_ARGS=()
[[ -n "${VERCEL_SCOPE:-}" ]] && SCOPE_ARGS=(--scope "$VERCEL_SCOPE")

vc() {  # 고정한 버전의 Vercel CLI. 토큰은 쓰지 않고 vercel login의 세션을 씀
  npx -y "$VERCEL_CLI" ${SCOPE_ARGS[@]+"${SCOPE_ARGS[@]}"} "$@"
}

if ! vc whoami >/dev/null 2>&1; then
  echo "Vercel 로그인 필요: npx -y $VERCEL_CLI login" >&2
  exit 1
fi
cd "$ROOT"
