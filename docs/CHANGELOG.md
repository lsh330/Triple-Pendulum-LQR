# 변경 이력 (CHANGELOG)

[Keep a Changelog](https://keepachangelog.com/ko/1.0.0/) 형식을 따른다. 버전 번호는 [Semantic Versioning](https://semver.org/lang/ko/)을 준수한다.

---

## [v3.1.0] — 2026-04-13

### 추가됨 (Added)

- **`analysis/roa_utils.py`**: `wilson_ci_width`, `adaptive_sample_count`, `get_u_max` 공통 유틸 — 두 ROA 모듈이 동일 정책 공유
- **`analysis/performance/rms_error.py`**: `compute_channel_rms` — 8개 상태 채널별 RMS 개별 보고 (`pipeline/runner.py`에 자동 통합)
- **`analysis/lqr_verification/compute_verification.py::compute_robust_stability`**: 공칭 게인 $K_{\text{nom}}$ 고정 상태에서 섭동 플랜트에 대한 polytopic robust stability 검증
- **`analysis/gain_scheduling_stability.py::verify_slow_variation`**: Shamma-Athans 이론 기반 $\|dK/d\sigma\|$ 상한 $\gamma_{\max}$ 계산 및 margin 보고
- **`analysis/gain_scheduling_stability.py::verify_common_lyapunov`**: LMI 기반 공통 Lyapunov 함수 SDP 검증 (cvxpy 옵셔널, 미설치 시 평균 P 폴백)
- **`control/linearization/trim_solver.py`**: 언더액추에이티드 시스템의 최소제곱 trim 해 — gain scheduling 격자 운영점을 heuristic offset 대신 엄밀 평형으로 재정의
- **`control/gain_scheduling.py`**: feedforward $u_{\text{ff}}$ 통합 (`get_gain_and_ff`, `pack_ff_for_njit`, 3D 격자는 `u_ff_grid` 보간)
- **`pipeline/multi_equilibrium_runner.py`**: `run_all_equilibria()` — 8개 평형점에 대한 animation.gif + dynamics/control/lqr_verification/roa/comparison 분석을 `images/{NAME}/` 서브디렉토리에 자동 저장, `summary_grid.png` 비교 그리드 생성
- **`visualization/common/korean_font.py`**: Windows/macOS/Linux 플랫폼별 한글 폰트 자동 탐색 (맑은 고딕 → 나눔고딕 → AppleGothic → DejaVu Sans 폴백)
- **CLI 플래그**: `--all-equilibria`, `--equilibria-list "UDD,UUU"`
- **`parameters/config.py::SystemConfig.actuator_saturation`**: 구동기 포화 필드 (기본 200 N). CLI `--u-max`와 완전히 연동
- **테스트**: `tests/test_warning_fixes.py` (W1~W6 검증 40개), `tests/test_multi_equilibrium_plots.py` (한글 폰트/시각화 18개) — **총 180개 → 238개** (PASS)
- **신규 이미지**: `images/{DDD,DDU,DUD,DUU,UDD,UDU,UUD,UUU}/{animation.gif, dynamics_analysis.png, control_analysis.png, lqr_verification.png, roa_analysis.png, comparison_analysis.png}` + `summary_grid.png` (총 **49개 신규 자산**)

### 변경됨 (Changed)

- **Numba `fastmath=True, boundscheck=False`** 핫패스 일괄 적용: `forward_dynamics_fast`, `_det3`, `rk4_step_fast`, `_run_loop_fast`, `_run_loop_gs_fast`, `_run_loop_switching`, `_lyapunov_value`, `total_energy_scalar`, `target_energy_from_phis`, `_roa_simulate_one`, `_roa_batch`
- **`angle_wrap`/`_angle_wrap`는 fastmath 미적용**: `-π` 경계 denormal 안전성을 위해 정책적으로 제외 (`test_core.py::TestAngleWrap::test_minus_pi` 회귀 방지)
- **ROA 추정 4.2배 가속**: `estimate_lyapunov_roa` 0.429 s → **0.102 s**. `_halton_precompute` 사전 생성 + 신규 `_roa_batch_lyapunov`(`@njit(parallel=True, fastmath=True)`) 병렬 커널 + 배치 단위 Wilson CI 적응 샘플링
- **`u_max` 일관성**: `analysis/region_of_attraction.py` 기본값 `1e30` → **200 N**. `roa_estimation.py`와 동일 정책 적용 (이전에는 분석/감독자 경로가 서로 다른 포화 값 사용)
- **ROA Wilson CI 적응형**: `roa_estimation.py`의 고정 300 샘플 → `n∈[300, 2000]` 적응형 (CI 폭 < 0.05 수렴 시 조기 종료)
- **문서 분할**: README 947줄 → **331줄** + `docs/USAGE.md`, `docs/ANALYSIS.md`, `docs/PERFORMANCE.md`, `docs/CHANGELOG.md` (긴 단일 페이지에서 수식 렌더링이 깨지던 문제 구조적 해결)
- **`pyproject.toml` version**: `2.0.0` → **`3.0.0`** (CHANGELOG와 정합)
- **`pipeline/save_outputs.py`, `visualization/common/korean_font.py`**: `from __future__ import annotations` 추가로 Python 3.9 호환 달성 (`str | None` PEP 604 런타임 이슈 해결)
- **`tests/conftest.py`**: `matplotlib.use("Agg")` 강제 설정 — CI/headless 환경에서 Tcl/Tk 의존 제거
- **`main.py`**: `SystemConfig(actuator_saturation=args.u_max)` 전달로 CLI `--u-max`가 ROA 분석에도 실제 반영되도록 수정

### 수정됨 (Fixed)

- **CLI `--u-max` 무시 버그**: 기존에는 `args.u_max`가 시뮬레이션 루프에만 전달되고 ROA 포화 판정에는 `SystemConfig` 기본값(200 N)이 사용되어 불일치. 이제 `SystemConfig.actuator_saturation`으로 단일 경로 일관
- **Tcl/Tk 의존 테스트 실패**: `test_save_figure_subdir` 등 GUI 백엔드 요구 테스트가 `matplotlib.use("Agg")`로 해결
- **`verify_common_lyapunov` 폴백 검증**: cvxpy 미설치 환경에서 평균 $P$ 근사를 채택할 때 $A_{cl,i}^T P + P A_{cl,i} \prec 0$ 위반 여부를 `warning`/`common_P_exists` 필드로 명시

### 제거됨 (Removed)

- 임시 벤치마크 스크립트(`bench_*.py`, `verify_accuracy.py`)와 C++ 실험 디렉토리(`cpp_ext/`)를 저장소에서 제외 (`.gitignore` 등록)

### 성능 결정: C++ 포팅 불채택

pybind11 + Eigen 기반 `forward_dynamics`/`rk4_step` 구현을 `-O3 -march=native -ffast-math -DNDEBUG`로 빌드하고 실측한 결과:

| 구현 | `_run_loop_fast` step time | 해석 |
|------|--------------------------|------|
| **Numba `@njit(fastmath=True)`** | **285 ns/step** | LLVM이 전체 루프를 단일 컴파일 유닛으로 inline + SIMD 벡터화 |
| C++/GCC -O3 -ffast-math | 935 ns/step | 함수 경계에서 최적화 제한, Python↔C++ dispatch 왕복 |

Numba가 C++보다 **3.3배 빠름**이 실측으로 확인되어 포팅하지 않음. 자세한 수치는 [PERFORMANCE.md](PERFORMANCE.md) 참조.

---

## [v3.0] — 2026-04-13

### 추가됨 (Added)

- **8개 평형점 전체 지원**: DDD~UUU 모든 구성에서 LQR 설계 및 안정화 시뮬레이션
- **`--equilibrium` CLI 플래그**: 목표 평형점 직접 지정 (DDD/DDU/DUD/DUU/UDD/UDU/UUD/UUU)
- **`--all-equilibria` CLI 플래그**: 8개 평형점 일괄 실행 및 비교 테이블 출력
- **Form-Switching Supervisor**: BFS 경로 계획 + 에너지 swing-up + LQR catch FSM
  - `MODE_SWING_UP` (0): 에너지 성형 $u = k_e(E^* - E)\dot{x}$
  - `MODE_LQR_CATCH` (1): LQR + ROA 히스테리시스 모니터링
  - `MODE_STABILIZED` (2): LQR 유지 + 500 스텝 수렴 확인
- **`control/robust_stability.py`**: `compute_robust_stability` — 공칭 게인 고정 강건 안정성 검증 (±10% 섭동 200 샘플)
- **`control/channel_rms.py`**: `compute_channel_rms` — 상태 채널별 RMS (정상상태/과도 구간 분리)
- **`control/trim_solver.py`**: `trim_solver` — 뉴턴법 기반 엄밀 운영점 계산 ($G(\mathbf{q}^*) = 0$ 만족)
- **`verify_slow_variation`**: Shamma-Athans(1990) 이론 기반 느린 변동 조건 검증
- **`verify_common_lyapunov`**: LMI 기반 공통 Lyapunov 함수 검증 (cvxpy 선택 의존성)
- **Korean font 자동 검출**: `visualization/common/font_utils.py` — 플랫폼별 한글 폰트 자동 등록
- **평형점 적응 Q 행렬**: 직립 링크 페널티 100, 하방 링크 페널티 10 자동 설정
- **히스테리시스 chattering 방지**: $\rho_{\text{in}} = 0.5\rho$, $\rho_{\text{out}} = 0.8\rho$ 이중 임계값
- **한글 이론/아키텍처 문서**: `docs/THEORY.md`, `docs/ARCHITECTURE.md` 전면 한글화
- **신규 문서 4종**: `docs/USAGE.md`, `docs/ANALYSIS.md`, `docs/PERFORMANCE.md`, `docs/CHANGELOG.md`

### 변경됨 (Changed)

- **ROA 추정**: Halton 준난수 + Wilson 점수 95% CI 적응 샘플링 (500~2,000 샘플)
- **BFS 전환 경로**: 해밍 거리 1 인접 그래프에서 BFS로 최단 홉 경로 자동 계획
- **JIT 데이터 패킹**: 슈퍼바이저 데이터를 float64 numpy 배열로 직렬화하여 `@njit` 루프 전달
- **문서 분할**: README.md(기존 947줄) → README.md(~400줄) + 6개 서브문서로 분할
- **`actuator_saturation` 필드**: YAML `features` 섹션에 `u_max` 별칭 추가

### 수정됨 (Fixed)

- **THEORY.md LaTeX 오류**: `\text{PE\_code}` → `$E_{\mathrm{code}}$` (MathJax 렌더링 오류 9개소 수정)
- **README.md LaTeX 오류**: `\\\` (3-백슬래시) → `\\` 수정 (6개소), bmatrix 단일행 분할
- **`\text{}` 내 한글 제거**: GitHub MathJax `\text{}` 안의 한글 금지 규칙 적용
- **`$< 0.1$` → `$\lt 0.1$`**: GitHub MathJax 인라인 부등호 안전 처리
- **`\|...\|` → `\lVert...\rVert`**: norm과 절대값 기호 명확화

---

## [v2.5] — 2025-12

### 추가됨

- 원시 제어 추적 (`u_raw_peak` vs `u_max` 비교 출력)
- 전체 8변수 NaN 감지 + 배열 채우기
- NaN 안전 요약 출력
- 입방 Hermite 보간 JIT fast 루프 적용
- 행렬 지수함수(expm) 비교 기능
- Monte Carlo 질량+길이 동시 섭동
- 선형화 자기 검사 (analytical vs numerical 일치 확인)
- 에너지/수렴 테스트 추가
- 대역폭 검증 테스트
- `BadCoefficients` 경고 억제

### 수정됨

- ROA CI 슬라이스 버그 수정

---

## [v2.4] — 2025-10

### 추가됨

- 상대 특이성 허용치 (절대값 대신 상대적 기준)
- RK45 시도 가드 (무한 루프 방지)
- iLQR CARE 종단 비용 설정
- CLI 파라미터 유효성 검사
- 적응 야코비안 스텝 크기 ($h_j = \varepsilon_{\text{mach}}^{1/3} \cdot \max(1, \lvert x_j \rvert)$)
- 극점 여유 테스트 (Re(λ) < -0.1 검증)

---

## [v2.3] — 2025-09

### 변경됨

- 전체 파이프라인 기능 연결: `--gain-scheduler 3d`, `--adaptive-q` CLI 연동
- 데드 코드 완전 제거
- 모든 기능이 파이프라인에서 실제로 호출되는지 검증

---

## [v2.2] — 2025-08

### 추가됨

- `GainScheduler` → `simulate()` 전달 인터페이스
- YAML CLI 재정의 수정 (YAML 값이 CLI 플래그에 의해 올바르게 덮어써짐)
- iLQR 시뮬레이션 루프 연결
- NaN 감지 로직 추가

---

## [v2.1] — 2025-07

### 추가됨

- CARE 제어가능성 검증 (랭크 8 확인)
- Q/R 행렬 유효성 검사 (Q 반양정치, R 양정치)
- 질량행렬 특이성 가드
- ROA 빠른 스칼라 커널 (3배 속도)
- Halton 준난수 샘플링
- ProcessPoolExecutor 병렬 Monte Carlo
- iLQR 행렬 지수함수 이산화
- 3D 게인 스케줄링 그리드 175점 (7×5×5)
- Bryson 규칙 PD 제어기
- 파라미터 패킹 문서화

---

## [v2.0] — 2025-06

대규모 리팩토링 버전.

### 추가됨

- pytest 테스트 스위트 (180개 테스트)
- pyproject.toml 현대적 패키징 (상한 버전 의존성)
- 해석적 야코비안 (analytical dG/dq, dM/dq)
- RK45 Dormand-Prince 적응 적분기
- 다축 게인 스케줄링 (1D/3D)
- iLQR (Iterative LQR)
- 적응 ROA 추정 (Wilson CI)
- CLI (argparse) 및 YAML 설정 파일
- logging 모듈 통합
- 비교 플롯 (LQR vs PD vs 극점 배치)
- JIT 사전 빌드 스크립트

### 변경됨

- 전체 모듈 구조를 도메인 패키지로 재조직 (11개 패키지)

---

## [v1.0] — 2025-01

최초 릴리즈.

### 추가됨

- 라그랑주 역학 기반 삼중진자 동역학 (해석적 크리스토펠 기호)
- LQR 최적 제어 (CARE 풀이)
- 영할당 JIT 시뮬레이션 (스칼라 상태 변수)
- 주파수 분석 (Bode, Nyquist, 민감도)
- ROA 추정 (Monte Carlo)
- 학술지 품질 시각화 (6종 플롯)
- Medrano-Cerda (1997) 벤치마크 파라미터 기준
