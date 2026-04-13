# 아키텍처 문서

카트-삼중진자 시뮬레이션의 소프트웨어 아키텍처를 서술한다.

---

## 1. 모듈 의존성 다이어그램

```
main.py
  |
  v
pipeline/runner.py  ───────────────────────────────────────────────
  |         |            |            |              |              |
  v         v            v            v              v              v
control/  simulation/  analysis/  visualization/  parameters/    utils/
  |         |            |            |              |              |
  |         v            |            |              v              |
  |    dynamics/         |            |         config.py           |
  |      |               |            |         equilibrium.py     |
  v      v               v            v         derived.py         v
lqr.py  mass_matrix/   energy/     animation/  packing.py     logger.py
ilqr.py gravity/       frequency/  dynamics_plots/
gain_scheduling.py     state/      control_plots/
swing_up/              lqr_verif/  lqr_plots/
supervisor/            region_of_  switching_plots/
robust_stability.py    attraction
channel_rms.py
trim_solver.py
```

### 핵심 의존 관계

- `simulation/` → `dynamics/`: 시뮬레이션 루프가 순동역학(forward dynamics) 호출
- `control/` → `dynamics/`: 선형화(linearization)가 질량행렬, 중력, 코리올리스 호출
- `pipeline/` → 모든 모듈: 오케스트레이터로서 전체 파이프라인 조율
- `core/` → 없음: 순수 유틸리티 (types, state_index, angle_utils)

---

## 2. 데이터 흐름

```
[YAML/CLI 파라미터]
        |
        v
[SystemConfig] ──pack()──> [float64 배열, dim=13]
        |                           |
        v                           v
[equilibrium()] ──> [q_eq]    [@njit 커널]
        |                     (forward_dynamics_fast,
        v                      rk4_step_fast,
[linearize()] ──> [A, B]      time_loop_fast,
        |                      time_loop_switching)
        v                           |
[solve_riccati()] ──> [K, P]        v
        |                    [출력 배열: q, dq, u, mode, energy]
        v                           |
[GainScheduler]                     v
  pack_for_njit() ──>        [analysis/]
  [gs_dev, gs_K, gs_slopes]         |
                                    v
                             [visualization/]
                                    |
                                    v
                             [PNG/GIF 저장]
```

### 핵심 설계 원칙

1. **JIT 경계 최소화**: Python 객체 → flat numpy 배열로 변환 후 JIT 커널에 전달
2. **Pre-allocation**: 출력 배열은 루프 진입 전 `np.empty()`로 단일 할당
3. **Zero-allocation 루프**: 스칼라 상태 변수로 루프 내 힙 할당 완전 제거

---

## 3. JIT 전략

### 3.1 스칼라 상태 접근

4-DOF 시스템의 상태를 배열이 아닌 8개 스칼라 변수로 관리:

```
sq0, sq1, sq2, sq3     ← 일반화 좌표 (x, θ₁, θ₂, θ₃)
sdq0, sdq1, sdq2, sdq3 ← 일반화 속도
```

이 접근의 장점:
- **힙 할당 제거**: `np.array()` 생성 비용 없음
- **레지스터 최적화**: Numba가 스칼라를 CPU 레지스터에 직접 매핑
- **함수 호출 최적화**: 스칼라 인자 전달은 배열 복사보다 빠름

### 3.2 캐시 전략

```python
@njit(cache=True)  # 컴파일 결과를 __pycache__에 저장
def rk4_step_fast(...):
    ...
```

- 첫 실행: JIT 컴파일 (~10–30 ms)
- 이후 실행: 캐시에서 로드 (~1 ms)
- `prebuild_cache.py`: 모든 JIT 함수를 사전 컴파일

### 3.3 Warmup 패턴

```python
# 소규모 호출로 JIT 트리거 (main 시뮬레이션 전)
_run_loop_fast(3, dt, q0, dq0, q_eq, K_flat, p, np.empty(0), u_max)
```

---

## 4. 이중 동역학 경로

| 용도 | 경로 | 특징 |
|------|------|------|
| 시뮬레이션 | `forward_dynamics_fast.py` | 스칼라 입출력, inline cofactor 4×4 풀이 |
| 선형화/분석 | `forward_dynamics.py` | 배열 입출력, `np.linalg.solve` |

두 경로는 동일한 물리 방정식을 구현하지만, 성능 최적화 수준이 다르다:

- **스칼라 경로**: 삼각함수 12회, 곱셈/덧셈 약 150회, 할당 0회
- **배열 경로**: numpy 배열 연산, 가독성 우선, 야코비안 수치미분에 적합

`test_dynamics.py`의 `test_forward_dynamics_consistency`가 두 경로의 일치를 검증한다.

---

## 5. Form-Switching 아키텍처

### 5.1 Supervisor FSM

```
                     ┌─── 전환 명령 ───┐
                     v                  |
            ┌─────────────┐      ┌─────────────┐
            │  SWING_UP   │─────>│  LQR_CATCH  │
            │  (mode=0)   │<─────│  (mode=1)   │
            └─────────────┘      └──────┬──────┘
              u = kₑ(E*-E)ẋ       V<ρ_in│  │V>ρ_out
                                        v  ^
                                 ┌──────────────┐
                                 │  STABILIZED  │
                                 │  (mode=2)    │──> 다음 stage
                                 └──────────────┘
                                   500스텝 수렴 확인
```

### 5.2 데이터 패킹 (Python → JIT)

```python
supervisor.pack_for_njit(path) → {
    'n_stages':    int,          # 전환 단계 수
    'all_q_eq':    (n, 4),       # 각 stage 목표 평형점
    'all_K_flat':  (n, 8),       # 각 stage LQR 게인
    'all_P_flat':  (n, 8, 8),    # 각 stage CARE 해
    'all_E_target': (n,),        # 각 stage 목표 에너지
    'all_rho_in':  (n,),         # LQR catch 진입 임계값
    'all_rho_out': (n,),         # Swing-up 복귀 임계값
}
```

모든 데이터가 float64 numpy 배열로 직렬화되어 `@njit` 루프에 전달된다.

### 5.3 전이 경로 계획

8개 평형점을 3비트 이진수로 매핑 (D=0, U=1):

```
DDD=000, DDU=001, DUD=010, DUU=011
UDD=100, UDU=101, UUD=110, UUU=111
```

인접 구성 = Hamming distance 1 (한 비트 플립). BFS로 최단 경로 탐색:

```
plan_transition("DDD", "UUU")
→ ["DDD", "DDU", "DUU", "UUU"]  (3단계, 각 1링크씩 전환)
```

---

## 6. 신규 모듈 (v3.0)

### 6.1 `control/robust_stability.py` — `compute_robust_stability`

공칭 LQR 게인을 **고정**한 채 질량/길이 파라미터를 랜덤 섭동하여 폐루프 극점이 모두 좌반평면(LHP)에 유지되는지 검증한다. 섭동 범위와 샘플 수는 인수로 지정한다.

```python
result = compute_robust_stability(
    nominal_K, A_nominal, B_nominal,
    param_perturb_frac=0.10,   # ±10% 섭동
    n_samples=200
)
# result: {"pass_rate": float, "min_real_eig": float}
```

### 6.2 `control/channel_rms.py` — `compute_channel_rms`

시뮬레이션 출력 배열에서 각 상태 채널(x, θ₁, θ₂, θ₃, ẋ, θ̇₁, θ̇₂, θ̇₃)의 RMS를 계산한다. 정상상태 구간(전체 시간의 후반 50%)과 과도 구간을 분리하여 보고한다.

### 6.3 `control/trim_solver.py` — `trim_solver`

게인 스케줄링에서 사용하는 **엄밀 운영점(trim point)**을 계산한다. 단순히 각도를 고정하는 대신, $G(\mathbf{q}^*) = 0$을 만족하는 실제 정적 평형점을 뉴턴법으로 탐색한다. 이를 통해 gain scheduling 이론에서 요구하는 평형족(family of equilibria)을 정확히 정의한다.

### 6.4 `control/supervisor/` — `verify_slow_variation` & `verify_common_lyapunov`

**`verify_slow_variation`**: Shamma-Athans(1990) 이론에 기반하여 스케줄링 파라미터가 충분히 느리게 변화하는지 검사한다. 게인 스케줄링 폐루프 시스템의 안정성 보장을 위한 충분 조건이다.

**`verify_common_lyapunov`**: 모든 운영점에서 동시에 성립하는 공통 Lyapunov 함수의 존재 여부를 LMI(선형 행렬 부등식)로 검증한다. `cvxpy`가 설치된 경우에만 활성화된다.

### 6.5 `--all-equilibria` CLI 플래그

8개 평형점 전체를 순차적으로 시뮬레이션하고 결과를 `images/{DDD,DDU,...,UUU}/` 디렉토리에 각각 저장한다. ROA 성공률과 정착시간을 콘솔 테이블로 출력한다.

### 6.5 `analysis/roa_utils.py` — Wilson CI 공통 유틸

`wilson_ci_width`, `adaptive_sample_count`, `get_u_max` 등을 제공하여 `analysis/region_of_attraction.py`와 `control/supervisor/roa_estimation.py` 두 ROA 모듈이 동일한 **적응형 Monte Carlo 샘플링 전략**과 **구동기 포화 정책**을 공유하도록 한다. 이로써 두 모듈의 ROA 추정값이 항상 일관성을 유지한다.

### 6.6 `analysis/performance/rms_error.py` — 채널별 RMS

시뮬레이션 결과로부터 카트 위치, 3 링크 절대각, 카트 속도, 3 링크 각속도의 RMS 오차를 개별 채널 단위로 계산한다. `pipeline/runner.py`에서 자동 호출되며 CHANGELOG/REVIEW 목적의 정량 지표를 제공한다.

### 6.7 `pipeline/multi_equilibrium_runner.py` — 8평형점 일괄 처리

`run_all_equilibria()`가 8개 평형점 각각에 대해 LQR → 시뮬레이션 → 분석 → 시각화를 독립 디렉토리(`images/{NAME}/`)에 저장하고, 마지막에 `summary_grid.png`로 4×2 비교 그리드를 생성한다. CLI `--all-equilibria`, `--equilibria-list "DDD,UUU,…"`로 호출 가능하다.

### 6.8 `--all-equilibria` / `--equilibria-list` CLI

- `--all-equilibria`: 8개 평형점 전체 순회
- `--equilibria-list "UDD,UUU"`: 쉼표 구분 부분 집합만 실행

### 6.9 Korean Font 자동 검출

`visualization/common/korean_font.py`가 플랫폼별 한글 폰트 경로를 탐색하여 matplotlib에 자동 등록한다. Windows(맑은 고딕), macOS(AppleGothic), Linux(NanumGothic) 순서로 시도하며, 미발견 시 `DejaVu Sans`로 폴백한다.

---

## 7. 테스트 구조

```
tests/
├── conftest.py                  # 공유 fixtures (cfg, p, q_eq, lqr_data)
├── test_dynamics.py             # 질량행렬, 중력, 코리올리스, 순동역학
├── test_linearization.py        # 야코비안 차원, 수치 검증
├── test_lqr.py                  # CARE, 게인, 폐루프 안정성, 게인 스케줄링
├── test_simulation.py           # 시뮬레이션 루프, 임펄스, 외란, ROA
├── test_parameters.py           # 파라미터 검증, 유도 상수
├── test_equilibria.py           # 8개 평형점 G=0, PE 순서, 인덱싱
├── test_energy_computation.py   # 에너지 계산, swing-up 제어
├── test_core.py                 # StateIndex, AngleWrap
├── test_multi_equilibrium_lqr.py # 8개 평형점 LQR 일괄 검증
├── test_validation_energy.py    # 에너지 보존, Coriolis 반대칭
└── test_utils.py                # OutputManager, DataLogger
```

**238개 테스트**, 전부 PASS. `pytest tests/ -v`로 실행한다. (기존 180 + W1~W6 수정 검증 40개 + 다중 평형점 시각화 18개)

### 검증 기준

| 항목 | 허용치 |
|------|--------|
| 에너지 보존 (u=0) | drift < 0.01% / 초 |
| G(q_eq) = 0 | < 1e-12 |
| CARE 잔차 | < 1e-6 |
| 폐루프 극점 | Re(λ) < 0 |
| Coriolis 반대칭 | z^T(Ṁ-2C)z < 1e-5 |

---

## 8. 성능 벤치마크

| 항목 | 성능 (환경 의존) |
|------|------|
| 표준 시뮬레이션 (15초) | 약 **2,000~3,200배** 실시간 |
| Switching 시뮬레이션 (30초) | 약 **1,300~2,000배** 실시간 |
| 에너지 계산 단일 호출 | 약 **540~680 ns** |
| ROA 추정 (적응형, 300~2000 샘플) | 약 **0.10~0.13 s** (적응 샘플링 + 병렬 커널 기준) |

`python benchmark.py`로 측정 가능하다. 상세 성능 분석은 [PERFORMANCE.md](PERFORMANCE.md)를 참조한다.
