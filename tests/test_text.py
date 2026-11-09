import unicodedata

from commentguard.text import normalize


def test_strips_and_collapses_whitespace():
    assert normalize("  안녕\t\t하세요\n\n반가워요  ") == "안녕 하세요 반가워요"


def test_nfd_becomes_nfc():
    nfd = unicodedata.normalize("NFD", "한글")
    assert nfd != "한글"
    assert normalize(nfd) == "한글"


def test_keeps_emoji_symbols_and_chosung():
    s = "ㅋㅋㅋ 대박😂 #최고!! ㅅㅂ?? lol"
    assert normalize(s) == s


def test_idempotent():
    s = " 가  나　다 "
    assert normalize(normalize(s)) == normalize(s)
