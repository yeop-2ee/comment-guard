"""학습과 서빙이 공유하는 유일한 전처리 함수."""

import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC 정규화, 앞뒤 공백 제거, 연속 공백(탭·줄바꿈 포함)을 공백 하나로. 이모지와 특수문자는 유지."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()
