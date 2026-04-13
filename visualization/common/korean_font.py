"""한국어 matplotlib 폰트 자동 설정 모듈.

Windows, macOS, Linux 환경을 자동 감지하여
시스템에 설치된 한글 폰트를 탐지하고 matplotlib에 등록한다.
폰트가 없을 경우 영문 폴백으로 동작한다.
"""

from __future__ import annotations  # Python 3.9 호환을 위한 forward reference

import platform
import os
import matplotlib
import matplotlib.pyplot as plt
from utils.logger import get_logger

log = get_logger()

# 플랫폼별 한글 폰트 우선순위 목록
_FONT_CANDIDATES = {
    "Windows": [
        "Malgun Gothic",     # 맑은 고딕 (Windows 기본)
        "NanumGothic",
        "NanumBarunGothic",
        "Gungsuh",
        "Batang",
    ],
    "Darwin": [             # macOS
        "AppleGothic",
        "Apple SD Gothic Neo",
        "NanumGothic",
        "NanumBarunGothic",
    ],
    "Linux": [
        "NanumGothic",
        "NanumBarunGothic",
        "UnDotum",
        "Baekmuk Gulim",
        "Noto Sans CJK KR",
    ],
}

_APPLIED_FONT: str | None = None   # 캐시된 적용 폰트명


def _get_installed_fonts() -> set[str]:
    """matplotlib 폰트 매니저에서 설치된 폰트 패밀리 목록을 반환한다."""
    from matplotlib.font_manager import fontManager
    return {f.name for f in fontManager.ttflist}


def apply_korean_font(fallback: str = "DejaVu Sans") -> str:
    """시스템 환경에 맞는 한글 폰트를 matplotlib에 적용한다.

    Parameters
    ----------
    fallback : str
        한글 폰트를 찾지 못했을 때 사용할 폴백 폰트 이름.

    Returns
    -------
    str
        실제로 적용된 폰트 패밀리 이름.
    """
    global _APPLIED_FONT
    if _APPLIED_FONT is not None:
        return _APPLIED_FONT

    system = platform.system()  # "Windows" | "Darwin" | "Linux"
    candidates = _FONT_CANDIDATES.get(system, _FONT_CANDIDATES["Linux"])
    installed = _get_installed_fonts()

    chosen = None
    for font_name in candidates:
        if font_name in installed:
            chosen = font_name
            break

    # Windows 환경에서 "Malgun Gothic"이 다른 이름으로 등록되는 경우 처리
    if chosen is None and system == "Windows":
        for font_name in installed:
            lower = font_name.lower()
            if "malgun" in lower or "nanum" in lower or "gothic" in lower:
                chosen = font_name
                break

    if chosen is not None:
        matplotlib.rcParams["font.family"] = chosen
        matplotlib.rcParams["axes.unicode_minus"] = False
        log.info("한글 폰트 적용: %s", chosen)
    else:
        matplotlib.rcParams["font.family"] = fallback
        log.warning(
            "한글 폰트를 찾을 수 없습니다. 폴백 폰트 '%s' 사용 중. "
            "한글 텍스트가 깨질 수 있습니다. "
            "pip install matplotlib-malgun-gothic 또는 "
            "apt install fonts-nanum 으로 설치하세요.",
            fallback,
        )
        chosen = fallback

    _APPLIED_FONT = chosen
    return chosen


def reset_font() -> None:
    """캐시를 초기화하여 다음 호출 시 폰트 재탐지를 허용한다.

    주로 테스트 환경에서 사용한다.
    """
    global _APPLIED_FONT
    _APPLIED_FONT = None
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
