"""W3: 개별 상태 채널별 RMS 추적 오차 계산.

``compute_channel_rms(states, eq_state, t)`` 함수는 8개 상태 채널
(cart x, phi1, phi2, phi3, dx, w1, w2, w3) 각각의 시간평균 RMS 오차를
반환한다.  ``pipeline/runner.py`` 에서 시뮬레이션 직후 호출되어 요약
로그에 포함된다.
"""

import numpy as np


def compute_channel_rms(
    states: np.ndarray,
    eq_state: np.ndarray,
    t: np.ndarray,
) -> dict:
    """각 상태 채널별 RMS 추적 오차를 계산한다.

    Parameters
    ----------
    states : np.ndarray, shape (N, 8)
        시뮬레이션 상태 시계열.
        열 순서: [x, theta1, theta2, theta3, dx, dtheta1, dtheta2, dtheta3].
    eq_state : np.ndarray, shape (8,)
        목표 평형 상태.  일반적으로 [x_eq, theta1_eq, ..., 0, 0, 0, 0].
    t : np.ndarray, shape (N,)
        시간 배열 (s).

    Returns
    -------
    dict
        채널 이름 → RMS 오차(float) 의 딕셔너리.
        키: ``cart_rms``, ``phi1_rms``, ``phi2_rms``, ``phi3_rms``,
            ``vx_rms``, ``w1_rms``, ``w2_rms``, ``w3_rms``,
            ``total_rms`` (8채널 합산 RMS).
    """
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError(
            f"states must be shape (N, 8), got {states.shape}"
        )
    if eq_state.shape != (8,):
        raise ValueError(
            f"eq_state must be shape (8,), got {eq_state.shape}"
        )
    if len(t) != states.shape[0]:
        raise ValueError(
            f"t length {len(t)} must match states rows {states.shape[0]}"
        )

    # 오차 행렬 (N x 8)
    err = states - eq_state[np.newaxis, :]

    # 채널별 RMS: sqrt(mean(err^2))
    channel_rms = np.sqrt(np.mean(err ** 2, axis=0))

    keys = [
        "cart_rms",   # x   [m]
        "phi1_rms",   # theta1 [rad]
        "phi2_rms",   # theta2 [rad]
        "phi3_rms",   # theta3 [rad]
        "vx_rms",     # dx  [m/s]
        "w1_rms",     # dtheta1 [rad/s]
        "w2_rms",     # dtheta2 [rad/s]
        "w3_rms",     # dtheta3 [rad/s]
    ]

    result = {k: float(v) for k, v in zip(keys, channel_rms)}

    # 합산 RMS (전체 상태 오차 스칼라 지표)
    result["total_rms"] = float(np.sqrt(np.mean(err ** 2)))

    return result
