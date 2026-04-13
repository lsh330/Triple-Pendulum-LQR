"""공통 축 스타일 헬퍼 함수 모음.

한글 폰트 적용 및 공통 스타일 파라미터를 포함한다.
"""

import matplotlib.pyplot as plt


def apply_grid(ax):
    """*ax* 에 연한 그리드를 추가한다."""
    ax.grid(True, alpha=0.3)


def apply_zero_line(ax):
    """*ax* 에 수평 영 기준선을 추가한다."""
    ax.axhline(0, color="k", ls="--", lw=0.5)


def apply_publication_style():
    """출판 품질 matplotlib rcParams를 전역 적용하고 한글 폰트를 설정한다.

    이 함수를 figure 생성 전에 한 번 호출하면
    모든 subplot에 일관된 스타일이 적용된다.
    """
    from visualization.common.korean_font import apply_korean_font
    apply_korean_font()

    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "axes.unicode_minus": False,
    })
