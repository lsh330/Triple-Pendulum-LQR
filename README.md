# 카트 위 삼중 역진자 시뮬레이터

LQR 최적 제어를 통한 카트 위 삼중 역진자(Cart + Triple Inverted Pendulum) 안정화 시뮬레이터이다. 대역제한 확률적 외란 하에서 **8개 평형 구성(DDD–UUU) 전체**를 지원하며, 에너지 기반 swing-up, FSM form-switching 제어, Lyapunov ROA 기반 모드 전환, 히스테리시스 chattering 방지 기능을 포함한다.

> **벤치마크 시스템**: 모든 물리 파라미터는 강건·최적 제어 분야의 가장 널리 인용되는 실험 벤치마크인 **Medrano-Cerda 삼중 역진자** (University of Salford, UK, 1997) [1]에서 가져왔다.

> **v3.0** — 다중 평형 제어, 에너지 기반 swing-up, FSM form-switching supervisor, Lyapunov ROA 기반 모드 전환, Numba JIT 2,000배 이상 실시간 가속, 238개 테스트. 변경 이력은 [CHANGELOG.md](docs/CHANGELOG.md)를 참조한다.

---

## 빠른 시작 (Quick Start)

```bash
pip install -e ".[test]"
python main.py                          # 기본 Medrano-Cerda (1D 게인 스케줄링)
python main.py --impulse 10 --t-end 20  # 파라미터 직접 지정
python main.py --config config.yaml     # YAML 설정 파일 사용
python main.py --equilibrium UDD        # UDD 평형점 안정화
python main.py --form-switch            # Form-switching: DDD → UUU
python main.py --all-equilibria         # 8개 평형점 일괄 실행
python main.py --gain-scheduler 3d      # 3D 삼선형 게인 스케줄링 (175점)
python main.py --adaptive-q             # 관성 스케일링 Q 행렬 (Bryson 규칙)
python main.py --use-ilqr               # iLQR 궤적 최적화 활성화
python prebuild_cache.py                # JIT 캐시 사전 빌드
pytest tests/ -v                        # 238개 테스트 실행
python benchmark.py                     # 성능 벤치마크
```

---

## 다중 평형점 제어 (Multi-Equilibrium Control)

삼중진자는 각 링크의 방향(Up/Down)에 따라 $2^3 = 8$개의 평형 구성을 가진다.

| 구성 | 링크 방향 | 위치에너지 (J) | 안정성 |
|------|-----------|---------------|--------|
| DDD | ↓↓↓ | −19.6 | 안정 |
| DDU | ↓↓↑ | −13.5 | 불안정 |
| DUD | ↓↑↓ | −9.5 | 불안정 |
| DUU | ↓↑↑ | −3.4 | 불안정 |
| UDD | ↑↓↓ | +3.4 | 불안정 |
| UDU | ↑↓↑ | +9.5 | 불안정 |
| UUD | ↑↑↓ | +13.5 | 불안정 |
| UUU | ↑↑↑ | +19.6 | 불안정 |

### Form-Switching 제어

에너지 기반 swing-up과 LQR catch를 결합하여 평형점 간 전환을 수행한다:

```bash
# DDD → UUU (기본: BFS로 최단 경로 계획)
python main.py --form-switch

# 특정 경로 지정
python main.py --form-switch --switch-source DDD --switch-target UUU

# 에너지 게인 및 시뮬레이션 시간 조정
python main.py --form-switch --k-energy 80 --switch-time 40
```

---

## 8개 평형점 시뮬레이션 결과

모든 결과는 Medrano-Cerda 파라미터, 초기 임펄스 5 N·s, 대역제한 노이즈(RMS = 15 N, $f_c$ = 3 Hz) 조건에서 산출되었다.

| 평형점 | ROA 성공률 | 정착시간 (s) | 결과 이미지 |
|--------|-----------|-------------|------------|
| DDD | 100.0% | 9.58 | [images/DDD/](images/DDD/) |
| DDU | 55.1% | 9.89 | [images/DDU/](images/DDU/) |
| DUD | 50.0% | 9.63 | [images/DUD/](images/DUD/) |
| DUU | 20.6% | 9.88 | [images/DUU/](images/DUU/) |
| UDD | 91.8% | 9.83 | [images/UDD/](images/UDD/) |
| UDU | 48.1% | 9.84 | [images/UDU/](images/UDU/) |
| UUD | 56.9% | 9.81 | [images/UUD/](images/UUD/) |
| UUU | 20.9% | 9.95 | [images/UUU/](images/UUU/) |

DDD(전체 하방)에서 ROA 성공률 100%인 반면, UUU(전체 상방)에서는 불안정성이 가장 심해 20.9%에 그친다. 8개 평형점 일괄 실행은 `--all-equilibria` 플래그로 수행한다.

---

## 설치

**요구 사항**: Python >= 3.9, pip

`pyproject.toml` 방식 (권장):

```bash
pip install -e ".[test]"
```

레거시 requirements 방식:

```bash
pip install -r requirements.txt
```

| 패키지 | 버전 | 용도 |
|--------|------|------|
| numpy | >= 1.24, < 2.1 | 수치 배열 및 선형대수 |
| scipy | >= 1.10, < 1.15 | Riccati 방정식 풀이, 주파수 응답 |
| numba | >= 0.57, < 0.61 | JIT 컴파일 (실시간 동역학) |
| matplotlib | >= 3.6, < 3.10 | 시각화 및 애니메이션 |
| pillow | >= 9.0, < 11.0 | GIF 애니메이션 내보내기 |
| pyyaml | >= 6.0, < 7.0 | YAML 설정 파일 지원 |
| cvxpy | 선택사항 | `verify_common_lyapunov` LMI 검증 |
| pytest | >= 7.0, < 9.0 | 테스트 프레임워크 (`[test]` 추가) |
| pytest-cov | >= 4.0, < 6.0 | 커버리지 리포트 (`[test]` 추가) |

모든 의존성에는 호환성 깨짐을 방지하기 위해 **상한 버전**이 명시되어 있다.

---

## 주요 기능 요약

| 기능 | 설명 |
|------|------|
| 8개 평형점 LQR | DDD~UUU 전체에서 CARE 기반 최적 게인 설계 |
| Form-Switching | BFS 경로 계획 + 에너지 swing-up + LQR catch |
| Gain Scheduling | 1D 단조 3차 Hermite 또는 3D 삼선형 보간 (175점) |
| iLQR | 비선형 궤적을 따른 시변 게인 최적화 |
| ROA 추정 | Halton 준난수 + Wilson 신뢰구간 적응 Monte Carlo |
| Numba JIT | 전체 시뮬레이션 루프 2,000배 이상 실시간 가속 |
| `--all-equilibria` | 8개 평형점 일괄 실행 및 결과 비교 |
| `actuator_saturation` | 구동기 포화 ($u_{\max}$ 기본값 200 N) |
| Korean font 자동 검출 | 플랫폼별 한글 폰트 자동 설정 |
| `compute_robust_stability` | 공칭 게인 고정 강건 안정성 검증 |
| `compute_channel_rms` | 상태 채널별 RMS 계산 |
| `trim_solver` | gain scheduling 엄밀 운영점 계산 |
| `verify_slow_variation` | Shamma-Athans 느린 변동 조건 검증 |
| `verify_common_lyapunov` | LMI 공통 Lyapunov 함수 검증 (cvxpy 옵션) |

---

## 시각화 결과 샘플

### 동역학 분석

![Dynamics Analysis](images/dynamics_analysis.png)

10개 서브플롯으로 카트 위치, 링크 각도, 에너지, 위상도, 제어력을 표시한다. 자세한 해설은 [ANALYSIS.md](docs/ANALYSIS.md#동역학-분석)를 참조한다.

### 제어 분석

![Control Analysis](images/control_analysis.png)

Bode 선도, Nyquist 선도, 민감도 함수, 극점 배치를 포함한 8개 서브플롯. [ANALYSIS.md](docs/ANALYSIS.md#제어-분석)에서 상세 해설을 제공한다.

### LQR 검증

![LQR Verification](images/lqr_verification.png)

Kalman 부등식, Riccati 해 검증, Lyapunov 함수 등 8개 서브플롯. [ANALYSIS.md](docs/ANALYSIS.md#lqr-검증)에서 상세 해설을 제공한다.

### 안정 영역 분석

![ROA Analysis](images/roa_analysis.png)

Monte Carlo 산포도, Wilson CI 성공률, 게인 스케줄링 고유값. [ANALYSIS.md](docs/ANALYSIS.md#안정-영역-분석)에서 상세 해설을 제공한다.

### 평형점 주변 거동

![Animation](images/animation.gif)

30 fps 애니메이션. 카트(회색 직사각형)가 수평 레일 위에서 이동하며, 세 링크(적색: 링크1, 녹색: 링크2, 청색: 링크3)가 안정화되는 과정을 보여준다.

---

## 문서 구조

| 문서 | 내용 | 링크 |
|------|------|------|
| README.md | 프로젝트 소개, 빠른 시작, 결과 요약 | 현재 파일 |
| THEORY.md | 라그랑주 역학, 8개 평형점, LQR, swing-up, FSM supervisor | [docs/THEORY.md](docs/THEORY.md) |
| ARCHITECTURE.md | 모듈 구조, 데이터 흐름, JIT 전략, 테스트 구조 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| USAGE.md | CLI 사용법, YAML 설정, 8평형점 실행 예제 | [docs/USAGE.md](docs/USAGE.md) |
| ANALYSIS.md | 동역학/제어/LQR/ROA/비교 분석 한글 해설 | [docs/ANALYSIS.md](docs/ANALYSIS.md) |
| PERFORMANCE.md | 벤치마크 결과, Numba JIT 전략, 캐시 | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| CHANGELOG.md | v3.x 버전별 변경 이력 | [docs/CHANGELOG.md](docs/CHANGELOG.md) |

---

## 프로젝트 구조

```
Triple-Pendulum-LQR/
├── main.py                              # CLI 진입점 (argparse)
├── pyproject.toml                       # 패키징 설정 (상한 버전 의존성)
├── requirements.txt                     # 레거시 의존성 목록
├── config.example.yaml                  # YAML 설정 템플릿
├── prebuild_cache.py                    # @njit 함수 사전 컴파일
│
├── utils/
│   ├── __init__.py
│   └── logger.py                        # Python logging 설정
│
├── pipeline/
│   ├── runner.py                        # 오케스트레이터: LQR → 시뮬레이션 → 분석 → 플롯 → 저장
│   ├── defaults.py                      # 기본 시뮬레이션 파라미터
│   └── save_outputs.py                  # PNG / GIF 내보내기
│
├── parameters/
│   ├── physical.py                      # 사용자 입력 (질량, 길이, 중력)
│   ├── derived.py                       # 유도 계수 (α, β, γ)
│   ├── packing.py                       # Numba용 평탄 배열 직렬화
│   ├── equilibrium.py                   # 8개 평형점 정의
│   └── config.py                        # SystemConfig 파사드
│
├── dynamics/                            # 전부 @njit 컴파일
│   ├── trigonometry.py                  # 공유 sin/cos 계산
│   ├── mass_matrix/
│   │   ├── cart_link_coupling.py        # 카트-진자 결합항
│   │   ├── pendulum_block.py           # 3×3 관성 서브행렬
│   │   └── assembly.py                 # 4×4 대칭 M 조립
│   ├── coriolis/
│   │   └── christoffel.py              # 해석적 크리스토펠 기호 (희소 폐형)
│   ├── gravity/
│   │   └── gravity_vector.py           # G(q) 계산
│   └── forward_dynamics/
│       ├── tau_assembly.py              # 입력 매핑 τ = (F, 0, 0, 0)
│       ├── solve_acceleration.py        # M 역행렬·rhs
│       ├── forward_dynamics.py          # 전체 q̈ = M⁻¹(τ − Cq̇ − G)
│       └── forward_dynamics_fast.py     # 영할당 단일형 스칼라 동역학 + RK4
│
├── control/
│   ├── linearization/
│   │   ├── analytical_jacobian.py       # 해석적 dG/dq, dM/dq 야코비안
│   │   ├── jit_jacobians.py            # @njit 결합 야코비안 계산
│   │   ├── jacobian_q.py               # ∂f/∂q (수치 폴백)
│   │   ├── jacobian_dq.py              # ∂f/∂q̇
│   │   ├── jacobian_u.py               # ∂f/∂u
│   │   ├── state_space.py              # A(8×8), B(8×1) 조립
│   │   └── linearize.py                # 선형화 파사드
│   ├── cost_matrices/
│   │   ├── default_Q.py                # Q = diag(10, 100, 100, 100, 1, 10, 10, 10)
│   │   └── default_R.py                # R = 0.01
│   ├── riccati/
│   │   └── solve_care.py               # scipy CARE 래퍼
│   ├── gain_computation/
│   │   └── compute_K.py                # K = R⁻¹BᵀP
│   ├── lqr.py                           # 종단 LQR 파사드
│   ├── closed_loop.py                   # A_cl, 고유값, 안정성 검사
│   ├── gain_scheduling.py               # 1D 3차 Hermite + 3D 삼선형 게인 스케줄링
│   ├── ilqr.py                          # 비선형 궤적을 위한 iLQR
│   ├── comparison.py                    # PD 및 극점 배치 비교 제어기
│   ├── robust_stability.py              # compute_robust_stability
│   ├── channel_rms.py                   # compute_channel_rms
│   ├── trim_solver.py                   # gain scheduling 엄밀 운영점
│   └── supervisor/
│       ├── form_switch_supervisor.py    # FSM 슈퍼바이저
│       ├── roa_estimation.py            # ROA Monte Carlo 추정
│       └── transition_graph.py          # BFS 전환 경로 계획
│
├── simulation/
│   ├── warmup.py                        # @njit 사전 컴파일 트리거
│   ├── integrator/
│   │   ├── state_derivative.py          # ż = (q̇, q̈) — @njit
│   │   ├── rk4_step.py                 # 단일 RK4 스텝 — @njit
│   │   └── rk45_step.py                # Dormand-Prince 적응 RK4(5) — @njit
│   ├── disturbance/
│   │   ├── white_noise.py              # 가우시안 백색 잡음 생성
│   │   ├── bandpass_filter.py          # FFT 기반 4차 Butterworth
│   │   ├── normalize.py                # RMS 진폭 스케일링
│   │   └── generate_disturbance.py     # 외란 생성 파이프라인
│   ├── initial_conditions/
│   │   └── impulse_response.py         # M·Δq̇ = (impulse, 0, 0, 0)ᵀ 풀이
│   └── loop/
│       ├── control_law.py              # u = −Kz — @njit
│       ├── time_loop.py                # 시뮬레이션 루프 (빠른 디스패치)
│       └── time_loop_fast.py           # 영할당 스칼라 상태 시뮬레이션 루프
│
├── analysis/
│   ├── state/                           # 절대각, 관절 위치, 편차
│   ├── energy/                          # 운동에너지, 위치에너지, 전체 에너지
│   ├── frequency/                       # 개/폐루프 전달함수, S(jω), T(jω), 여유, 극점, 스텝
│   ├── lqr_verification/               # Lyapunov, Kalman, Nyquist, Monte Carlo
│   ├── region_of_attraction.py          # 적응 Monte Carlo ROA (Wilson 신뢰구간)
│   ├── gain_scheduling_stability.py     # 보간 안정성 검증
│   └── summary/                         # 콘솔 출력 요약
│
├── visualization/
│   ├── common/                          # 공유 색상 및 축 스타일
│   ├── animation/                       # 카트-진자 FuncAnimation
│   ├── dynamics_plots/                  # 5×2 동역학 그리드
│   ├── control_plots/                   # 4×2 제어 그리드 (Nyquist 포함)
│   ├── lqr_plots/                       # 4×2 LQR 검증 그리드
│   ├── roa_plots/                       # 2×2 ROA & 게인 스케줄링 그리드
│   └── comparison_plots/               # LQR vs PD vs 극점 배치 비교
│
├── tests/
│   ├── conftest.py
│   ├── test_dynamics.py
│   ├── test_linearization.py
│   ├── test_lqr.py
│   ├── test_simulation.py
│   ├── test_parameters.py
│   ├── test_equilibria.py
│   ├── test_energy_computation.py
│   ├── test_core.py
│   ├── test_multi_equilibrium_lqr.py
│   ├── test_validation_energy.py
│   └── test_utils.py
│
├── docs/
│   ├── THEORY.md
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   ├── ANALYSIS.md
│   ├── PERFORMANCE.md
│   └── CHANGELOG.md
│
├── images/                              # 자동 생성 출력 플롯
├── LICENSE                              # MIT 라이선스
└── README.md
```

---

## 참고 문헌

1. Medrano-Cerda, G. A. (1997). "Robust stabilization of a triple inverted pendulum-cart." *IEE Proceedings — Control Theory and Applications*, 144(4), 315–325.
2. Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice Hall.
3. Kalman, R. E. (1964). "When is a linear control system optimal?" *ASME Journal of Basic Engineering*, 86(1), 51–60.
4. Glück, T., Eder, A. & Kugi, A. (2013). "Swing-up control of a triple pendulum on a cart with experimental validation." *Automatica*, 49(3), 801–808.
5. Dormand, J. R. & Prince, P. J. (1980). "A family of embedded Runge-Kutta formulae." *Journal of Computational and Applied Mathematics*, 6(1), 19–26.

---

## 라이선스

MIT — [LICENSE](LICENSE) 파일을 참조한다.
