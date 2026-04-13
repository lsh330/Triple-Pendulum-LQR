"""모든 figure와 animation을 images/ 디렉터리에 저장한다.

평형점별 서브디렉터리를 지원한다.
"""

from __future__ import annotations  # Python 3.9 호환을 위한 forward reference

import os

from utils.logger import get_logger

log = get_logger()

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def ensure_dir(subdir: str | None = None) -> str:
    """저장 디렉터리를 생성하고 경로를 반환한다.

    Parameters
    ----------
    subdir : str or None
        None이면 최상위 images/ 디렉터리.
        문자열이면 images/{subdir}/ 서브디렉터리.

    Returns
    -------
    str
        생성된(혹은 기존) 디렉터리의 절대 경로.
    """
    if subdir is not None:
        target = os.path.join(IMAGES_DIR, subdir)
    else:
        target = IMAGES_DIR
    os.makedirs(target, exist_ok=True)
    return target


def save_figure(fig, name: str, dpi: int = 150, subdir: str | None = None) -> str:
    """matplotlib figure를 PNG로 저장한다.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        파일 기본 이름 (확장자 제외).
    dpi : int
        저장 해상도.
    subdir : str or None
        평형점 이름(예: "UUU") 지정 시 images/UUU/ 에 저장.

    Returns
    -------
    str
        저장된 파일의 절대 경로.
    """
    target_dir = ensure_dir(subdir)
    path = os.path.join(target_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("  저장됨: %s", path)
    return path


def save_animation(ani, name: str, fps: int = 30, subdir: str | None = None) -> str:
    """matplotlib animation을 GIF로 저장한다.

    Parameters
    ----------
    ani : matplotlib.animation.FuncAnimation
    name : str
        파일 기본 이름 (확장자 제외).
    fps : int
        프레임 레이트.
    subdir : str or None
        평형점 이름 지정 시 images/{subdir}/ 에 저장.

    Returns
    -------
    str
        저장된 파일의 절대 경로.
    """
    target_dir = ensure_dir(subdir)
    path = os.path.join(target_dir, f"{name}.gif")
    try:
        ani.save(path, writer="pillow", fps=fps)
        log.info("  저장됨: %s", path)
    except Exception as e:
        log.warning("  GIF 저장 실패 (%s). pillow 설치 필요: pip install pillow", e)
    return path
