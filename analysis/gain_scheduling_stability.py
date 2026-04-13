"""Gain scheduling stability verification via eigenvalue analysis.

W5: ``verify_slow_variation`` 추가 — Shamma-Athans 느린 변화 조건 검증.
W6: ``verify_common_lyapunov`` 추가 — 모든 격자점에서 공통 Lyapunov P 존재 여부 LMI 검증.
"""

import numpy as np
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati


def verify_gain_scheduling_stability(cfg, gs):
    """Verify stability of gain-scheduled controller.

    For each operating point, computes the closed-loop system matrix
    A_cl = A - B @ K and checks eigenvalue stability. Also checks
    stability at interpolated intermediate points.

    Parameters
    ----------
    cfg : SystemConfig
    gs : GainScheduler

    Returns
    -------
    dict with keys:
        'operating_points_deg' : array of deviation angles in degrees
        'all_points_stable' : bool
        'interpolated_all_stable' : bool
        'max_eigenvalue_real' : float (worst-case max Re(eigenvalue))
        'condition_numbers' : P matrix condition numbers at each point
        'eigenvalues_per_point' : list of eigenvalue arrays
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium
    Q = default_Q()
    R = default_R()

    dev_angles = gs.deviation_angles  # radians, sorted
    K_gains = gs.K_gains              # (n_pts, 8)
    n_pts = len(dev_angles)

    operating_points_deg = np.rad2deg(dev_angles)

    eigenvalues_per_point = []
    condition_numbers = []
    all_points_stable = True
    max_eig_real = -np.inf

    # Step 1-3: Check each operating point
    for i in range(n_pts):
        q_op = q_eq.copy()
        q_op[1] += dev_angles[i]

        A, B = linearize(q_op, p)
        K_i = K_gains[i].reshape(1, 8)
        A_cl = A - B @ K_i

        eigs = np.linalg.eigvals(A_cl)
        eigenvalues_per_point.append(eigs)

        max_re = np.max(eigs.real)
        if max_re > max_eig_real:
            max_eig_real = max_re

        if max_re >= 0:
            all_points_stable = False

        # Compute P matrix condition number
        P = solve_riccati(A, B, Q, R)
        cond = np.linalg.cond(P)
        condition_numbers.append(cond)

    # Step 4: Check interpolated stability
    interpolated_all_stable = True
    n_interp = 100

    for i in range(n_pts - 1):
        for j in range(1, n_interp + 1):
            t = j / (n_interp + 1)
            # Interpolated angle
            delta = (1 - t) * dev_angles[i] + t * dev_angles[i + 1]

            # Linearize at interpolated point
            q_interp = q_eq.copy()
            q_interp[1] += delta

            A, B = linearize(q_interp, p)

            # Interpolated gain
            K_interp = (1 - t) * K_gains[i] + t * K_gains[i + 1]
            K_interp_mat = K_interp.reshape(1, 8)

            A_cl = A - B @ K_interp_mat
            eigs = np.linalg.eigvals(A_cl)

            max_re = np.max(eigs.real)
            if max_re > max_eig_real:
                max_eig_real = max_re

            if max_re >= 0:
                interpolated_all_stable = False

    condition_numbers = np.array(condition_numbers)

    return {
        'operating_points_deg': operating_points_deg,
        'all_points_stable': bool(all_points_stable),
        'interpolated_all_stable': bool(interpolated_all_stable),
        'max_eigenvalue_real': float(max_eig_real),
        'condition_numbers': condition_numbers,
        'eigenvalues_per_point': eigenvalues_per_point,
    }


# ---------------------------------------------------------------------------
# W5: Shamma-Athans 느린 변화(slow-variation) 조건 검증
# ---------------------------------------------------------------------------

def verify_slow_variation(scheduler, max_rate: float) -> dict:
    """Shamma-Athans 조건 — 파라미터 변화율이 이론적 상한 이하인지 검증.

    Shamma & Athans (1990) 에 따르면 게인 스케줄링이 안정성을 보장하려면
    스케줄링 파라미터 σ 의 변화율 |dσ/dt| 이 다음 상한 이하이어야 한다:

        γ_max = min_i  σ_min(-A_cl,i^T P - P A_cl,i) / (2 ‖B_i P‖)

    여기서 P 는 각 격자점 i 의 Lyapunov 행렬(CARE 해).  공통 P 를 가정하면
    더 보수적인 상한을 얻는다.

    Parameters
    ----------
    scheduler : GainScheduler
        1D cubic Hermite gain scheduler.
    max_rate : float
        파라미터(angle) 변화율의 예상 최대값 (rad/s).

    Returns
    -------
    dict with keys:
        'max_dK_dsigma'      : float — 격자점 간 최대 |ΔK/Δσ|
        'gamma_max'          : float — Shamma-Athans 이론적 상한
        'slow_variation_ok'  : bool  — max_rate < gamma_max
        'margin'             : float — gamma_max - max_rate (양이면 여유 있음)
        'details_per_gap'    : list of dict — 각 구간 상세 정보
    """
    cfg = scheduler.cfg
    p = cfg.pack()
    Q = default_Q()
    R = default_R()

    dev_angles = scheduler.deviation_angles   # (n_pts,) rad
    K_gains = scheduler.K_gains               # (n_pts, 8)
    n_pts = len(dev_angles)

    # 각 격자점의 (A_cl, B, P) 계산
    A_cls = []
    Bs = []
    Ps = []
    q_eq = cfg.equilibrium

    for i in range(n_pts):
        q_op = q_eq.copy()
        q_op[1] += dev_angles[i]
        A_i, B_i = linearize(q_op, p)
        K_i = K_gains[i].reshape(1, 8)
        A_cl_i = A_i - B_i @ K_i
        P_i = solve_riccati(A_i, B_i, Q, R)
        A_cls.append(A_cl_i)
        Bs.append(B_i)
        Ps.append(P_i)

    # γ_max = min_i  σ_min(-A_cl_i^T P_i - P_i A_cl_i) / (2 ‖B_i P_i‖_F)
    gamma_candidates = []
    for i in range(n_pts):
        lyap_lhs = -(A_cls[i].T @ Ps[i] + Ps[i] @ A_cls[i])
        sigma_min = float(np.min(np.linalg.eigvalsh(lyap_lhs)))
        B_P_norm = float(np.linalg.norm(Bs[i].flatten() @ Ps[i]))
        if B_P_norm > 1e-14:
            gamma_candidates.append(sigma_min / (2.0 * B_P_norm))
        else:
            gamma_candidates.append(np.inf)

    gamma_max = float(min(gamma_candidates)) if gamma_candidates else np.inf

    # 격자점 간 최대 ΔK/Δσ 계산
    details = []
    max_dK_dsigma = 0.0
    for i in range(n_pts - 1):
        delta_sigma = abs(dev_angles[i + 1] - dev_angles[i])
        if delta_sigma < 1e-14:
            continue
        delta_K = K_gains[i + 1] - K_gains[i]
        rate = float(np.linalg.norm(delta_K) / delta_sigma)
        details.append({
            'from_deg': float(np.rad2deg(dev_angles[i])),
            'to_deg': float(np.rad2deg(dev_angles[i + 1])),
            'dK_dsigma_norm': rate,
        })
        if rate > max_dK_dsigma:
            max_dK_dsigma = rate

    slow_variation_ok = bool(max_rate < gamma_max)

    return {
        'max_dK_dsigma': float(max_dK_dsigma),
        'gamma_max': float(gamma_max),
        'slow_variation_ok': slow_variation_ok,
        'margin': float(gamma_max - max_rate),
        'details_per_gap': details,
    }


# ---------------------------------------------------------------------------
# W6: 공통 Lyapunov P LMI 검증
# ---------------------------------------------------------------------------

def verify_common_lyapunov(cfg, scheduler) -> dict:
    """모든 격자점에서 공통 Lyapunov 행렬 P ≻ 0 존재 여부를 검증한다.

    공통 P 가 존재하면 임의 전환에도 안정성이 보장된다 (switched-system theory).

    방법
    ----
    1. cvxpy 가 설치된 경우: SDP(반정정프로그래밍) LMI 풀이.
       조건: 모든 i 에 대해 A_cl,i^T P + P A_cl,i ≺ 0, P ≻ 0
    2. cvxpy 없는 경우: 수치 근사 P = average(P_i) 를 공통 후보로 사용하고
       각 격자점에서 A_cl,i^T P + P A_cl,i 의 최대 고유값으로 판단.

    Parameters
    ----------
    cfg : SystemConfig
        시스템 구성.
    scheduler : GainScheduler
        1D cubic Hermite gain scheduler.

    Returns
    -------
    dict with keys:
        'common_P_exists'    : bool   — 공통 P 존재 여부
        'method'             : str    — 'cvxpy_sdp' 또는 'approximate'
        'common_P'           : (8,8) ndarray — 추정된 공통 P (또는 최선 근사)
        'worst_lyap_eig'     : float  — A_cl^T P + P A_cl 최대 고유값 (음이면 안정)
        'all_stable_under_P' : bool   — 공통 P 하에서 모든 격자점 안정 여부
        'n_points'           : int    — 검증된 격자점 수
        'warning'            : str or None
    """
    p = cfg.pack()
    Q = default_Q()
    R = default_R()

    dev_angles = scheduler.deviation_angles
    K_gains = scheduler.K_gains
    n_pts = len(dev_angles)
    q_eq = cfg.equilibrium

    A_cls = []
    P_locals = []

    for i in range(n_pts):
        q_op = q_eq.copy()
        q_op[1] += dev_angles[i]
        A_i, B_i = linearize(q_op, p)
        K_i = K_gains[i].reshape(1, 8)
        A_cls.append(A_i - B_i @ K_i)
        P_i = solve_riccati(A_i, B_i, Q, R)
        P_locals.append(P_i)

    # --- cvxpy SDP ---
    try:
        import cvxpy as cp

        n = 8
        P_var = cp.Variable((n, n), symmetric=True)
        constraints = [P_var >> np.eye(n) * 1e-6]  # P ≻ 0
        for A_cl_i in A_cls:
            # A_cl_i^T P + P A_cl_i ≺ 0  →  -(A_cl^T P + P A_cl) ≻ ε I
            constraints.append(
                A_cl_i.T @ P_var + P_var @ A_cl_i << -1e-6 * np.eye(n)
            )
        prob = cp.Problem(cp.Minimize(cp.trace(P_var)), constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate") and P_var.value is not None:
            P_common = np.array(P_var.value)
            # 검증
            worst = max(
                float(np.max(np.linalg.eigvalsh(A_cl_i.T @ P_common + P_common @ A_cl_i)))
                for A_cl_i in A_cls
            )
            all_stable = bool(worst < 0)
            return {
                'common_P_exists': all_stable,
                'method': 'cvxpy_sdp',
                'common_P': P_common,
                'worst_lyap_eig': worst,
                'all_stable_under_P': all_stable,
                'n_points': n_pts,
                'warning': None if all_stable else
                    "공통 Lyapunov P 미발견: 전환 안정성 이론적 보장 없음",
            }
        else:
            # SDP infeasible → 공통 P 없음
            P_common = np.mean(P_locals, axis=0)
            worst = max(
                float(np.max(np.linalg.eigvalsh(A_cl_i.T @ P_common + P_common @ A_cl_i)))
                for A_cl_i in A_cls
            )
            return {
                'common_P_exists': False,
                'method': 'cvxpy_sdp',
                'common_P': P_common,
                'worst_lyap_eig': worst,
                'all_stable_under_P': False,
                'n_points': n_pts,
                'warning': "SDP infeasible: 공통 Lyapunov P 없음. 전환 안정성 별도 검증 필요.",
            }

    except ImportError:
        pass  # cvxpy 미설치 → 수치 근사로 폴백

    # --- 수치 근사: P = mean(P_i) ---
    P_common = np.mean(P_locals, axis=0)
    # 대칭 강제
    P_common = 0.5 * (P_common + P_common.T)
    worst = max(
        float(np.max(np.linalg.eigvalsh(A_cl_i.T @ P_common + P_common @ A_cl_i)))
        for A_cl_i in A_cls
    )
    all_stable = bool(worst < 0)
    warning = (
        "cvxpy 미설치 — 평균 P 근사 사용. 정확한 LMI 검증을 위해 "
        "'pip install cvxpy' 를 실행하세요."
    )
    if not all_stable:
        warning += " 또한 평균 P 하에서도 일부 격자점이 불안정합니다."

    return {
        'common_P_exists': all_stable,
        'method': 'approximate',
        'common_P': P_common,
        'worst_lyap_eig': worst,
        'all_stable_under_P': all_stable,
        'n_points': n_pts,
        'warning': warning,
    }
