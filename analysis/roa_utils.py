"""ROA 공통 유틸리티 — W1: 두 ROA 모듈의 공유 함수.

``analysis/region_of_attraction.py`` 와
``control/supervisor/roa_estimation.py`` 가 동일한 액추에이터 포화
한계(u_max)와 적응형 Wilson-score 샘플링 전략을 사용하도록 중앙화.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Wilson score 95% CI 폭 계산
# ---------------------------------------------------------------------------

def wilson_ci_width(n_success: int, n_total: int, z: float = 1.96) -> float:
    """Wilson score 95% 신뢰구간 폭을 계산한다.

    Parameters
    ----------
    n_success : int
        성공(수렴) 샘플 수.
    n_total : int
        전체 샘플 수.
    z : float
        신뢰도 계수 (기본 1.96 → 95 %).

    Returns
    -------
    float
        CI 폭 = upper_bound - lower_bound (0~1 사이).
    """
    if n_total == 0:
        return 1.0
    rate = n_success / n_total
    n = n_total
    denom = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt((rate * (1.0 - rate) + z * z / (4.0 * n)) / n) / denom
    return float(2.0 * margin)


# ---------------------------------------------------------------------------
# 적응형 샘플 수 결정
# ---------------------------------------------------------------------------

def adaptive_sample_count(
    n_success: int,
    n_total: int,
    target_ci_width: float = 0.05,
    n_min: int = 300,
    n_max: int = 2000,
    z: float = 1.96,
) -> bool:
    """현재 Wilson CI 폭이 목표 이하인지 판단한다.

    Parameters
    ----------
    n_success : int
        수렴 샘플 수.
    n_total : int
        현재까지 누적 샘플 수.
    target_ci_width : float
        수렴 목표 CI 폭 (기본 0.05).
    n_min : int
        최소 샘플 수 (기본 300).
    n_max : int
        최대 샘플 수 (기본 2000).
    z : float
        신뢰도 계수.

    Returns
    -------
    bool
        True이면 충분히 수렴 → 루프 종료 가능.
    """
    if n_total < n_min:
        return False
    if n_total >= n_max:
        return True
    ci = wilson_ci_width(n_success, n_total, z)
    return ci < target_ci_width


# ---------------------------------------------------------------------------
# cfg 에서 u_max 안전하게 읽기
# ---------------------------------------------------------------------------

def get_u_max(cfg, fallback: float = 200.0) -> float:
    """``cfg.actuator_saturation`` 을 반환하고, 없으면 fallback 사용.

    W1: 모든 ROA/제어 모듈이 이 함수를 통해 포화 한계를 읽도록 통일.
    """
    return float(getattr(cfg, "actuator_saturation", fallback))
