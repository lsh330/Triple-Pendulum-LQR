"""W4: Trim Solver — 격자점 (θ1, θ2, θ3) 에서 G(q) = B_u * u_eq 를 만족하는 u_eq 계산.

이론적 배경
-----------
비평형 격자점 q_op = q_eq + [0, δ1, δ2, δ3] 에서 dq = 0, 가속도 = 0
조건(트림)을 만족하려면:

    M(q_op) * 0 + C(q_op, 0)*0 + G(q_op) = τ(u_eq)

즉 G(q_op) = B_u * u_eq 이어야 한다.
여기서 B_u = M(q_op)^{-1} [1, 0, 0, 0]^T 는 입력 방향 벡터이며,
G(q_op) 는 4차원 중력 벡터, [1,0,0,0]^T 는 카트 방향 입력 선택 벡터.

최소 제곱 해:
    u_eq = (B_u^T B_u)^{-1} B_u^T G_sub
여기서 G_sub = M^{-1} @ G(q_op) 는 가속도 공간 중력 벡터이다.

실제로 B_u = M^{-1} @ e_0 이므로:
    u_eq = B_u^T @ G_sub / (B_u^T @ B_u)
       = [M^{-1} e_0]^T [M^{-1} G] / [M^{-1} e_0]^T [M^{-1} e_0]
"""

import numpy as np
from dynamics.mass_matrix.assembly import mass_matrix
from dynamics.gravity.gravity_vector import gravity_vector


def compute_trim_u(q_op: np.ndarray, p: np.ndarray) -> float:
    """격자점 q_op 에서 트림 입력 u_eq 를 계산한다.

    Parameters
    ----------
    q_op : np.ndarray, shape (4,)
        작동점 구성 [x, theta1, theta2, theta3].
    p : np.ndarray, shape (13,)
        시스템 파라미터 (packed).

    Returns
    -------
    float
        트림 입력 u_eq (N). 이 값을 피드포워드로 추가하면 격자점에서
        중력 토크가 정확히 보상된다.
    """
    M = mass_matrix(q_op, p)                     # (4, 4)
    G = gravity_vector(q_op, p)                  # (4,)
    M_inv = np.linalg.inv(M)

    # 가속도 공간 중력 벡터
    G_acc = M_inv @ G                             # (4,)

    # 입력 방향 벡터 (B_u = M^{-1} e_0)
    e0 = np.zeros(4)
    e0[0] = 1.0
    B_u = M_inv @ e0                              # (4,)

    # 최소제곱 트림 입력
    denom = float(B_u @ B_u)
    if denom < 1e-14:
        return 0.0
    u_eq = float(B_u @ G_acc) / denom
    return u_eq
