"""Vercel 진입점 (명세서 6.2장). Vercel이 이 파일의 app을 Python 함수로 실행합니다.

로컬에서는 make serve(uvicorn commentguard.api:app)로 같은 앱을 띄웁니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from commentguard.api import app  # noqa: E402,F401
