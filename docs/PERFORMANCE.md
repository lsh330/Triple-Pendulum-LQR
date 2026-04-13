# 성능 분석 (PERFORMANCE)

카트-삼중진자 시뮬레이터의 Numba JIT 최적화 전략, 벤치마크 결과, 캐시 관리를 서술한다.

---

## 목차

1. [성능 요약](#1-성능-요약)
2. [Numba JIT 전략](#2-numba-jit-전략)
3. [최적화 세부 내역](#3-최적화-세부-내역)
4. [JIT 캐시 관리](#4-jit-캐시-관리)
5. [파이프라인 단계별 시간](#5-파이프라인-단계별-시간)

---

## 1. 성능 요약

`python benchmark.py`로 측정한 결과이다 (Intel Core i7 기준, 실행마다 ±10% 변동 가능).

| 항목 | 성능 (환경 의존) |
|------|------|
| 표준 시뮬레이션 (15초, dt=0.001) | 약 **2,000~3,200배** 실시간 |
| Switching 시뮬레이션 (30초) | 약 **1,300~2,000배** 실시간 |
| 에너지 계산 단일 호출 | 약 **540~680 ns** |
| ROA 추정 (적응형 300~2,000 샘플) | 약 **0.10~0.13 s** (병렬 커널 + Halton 사전 생성) |
| LQR 설계 (CARE + 게인 계산) | **~1 ms** (캐시 적중 시) |
| JIT 웜업 (전체 @njit 함수) | **~2.8 s** (최초 1회) |

> 수치는 CPU 부하, 전원 프로파일, JIT 캐시 상태에 따라 변동한다. v3.0의 `_roa_batch_lyapunov` 병렬 커널과 `_halton_precompute`로 인해 ROA 추정이 이전 0.39 s에서 약 **4.2배** 단축되었다 (자세한 내역은 [CHANGELOG.md](CHANGELOG.md)).

---

## 1.1 fastmath / boundscheck 정책

핫패스(`forward_dynamics_fast`, `rk4_step_fast`, `_run_loop_fast`, `_run_loop_gs_fast`, `_run_loop_switching`, `_lyapunov_value`, `total_energy_scalar`, `_roa_batch`, `_roa_batch_lyapunov`, `_roa_simulate_one`)에는 `@njit(cache=True, fastmath=True, boundscheck=False)`를 적용한다. 이는 LLVM에 -ffast-math에 준하는 재배열과 벡터화를 허용하여 SIMD 성능을 끌어올린다.

단, **각도 래핑 함수**(`core/angle_utils.py::angle_wrap`, `simulation/loop/time_loop_fast.py::_angle_wrap`)는 `fastmath=True`를 적용하지 않는다. `fastmath`는 `floor`/비교 연산의 denormal 처리를 변경하여 `-π` 경계 케이스에서 `test_core.py::TestAngleWrap::test_minus_pi`가 실패한다. 수치 정확도를 위해 이 두 함수만 `@njit(cache=True)`만 사용한다.

제어/분석/선형화 경로(예: `gain_scheduling.py`, `analytical_jacobian.py`, `rk45_step.py`)도 핫패스가 아니므로 기본 `@njit(cache=True)`로 유지하여 정확성을 우선한다.

---

## 2. Numba JIT 전략

### 2.1 스칼라 상태 접근 (Zero-Allocation)

시뮬레이션 핫 패스의 핵심 최적화이다. 4-DOF 시스템의 상태를 NumPy 배열 대신 8개의 스칼라 변수로 직접 관리한다.

```python
# 최적화 전: 배열 할당 발생
def step(q, dq, u, p):
    M = compute_mass_matrix(q)       # np.zeros((4,4)) 할당
    G = compute_gravity(q)           # np.zeros(4) 할당
    acc = np.linalg.solve(M, rhs)    # LAPACK 호출

# 최적화 후: 힙 할당 없음
@njit(cache=True)
def forward_dynamics_fast(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p):
    # 모든 계산이 스칼라 로컬 변수로 처리
    # 행렬식: 코팩터 전개 인라인 구현
    # 해: Cramer의 규칙 직접 계산
```

15초 시뮬레이션(15,001 스텝)에서 약 **570,000건**의 소형 배열 할당이 제거된다.

### 2.2 인라인 4×4 행렬 풀이

$M\ddot{\mathbf{q}} = \mathbf{r}$의 풀이를 LAPACK `np.linalg.solve` 대신 코팩터 전개로 인라인 구현한다.

```python
# 3×3 행렬식 헬퍼 (9개 스칼라 곱)
def _det3(a, b, c, d, e, f, g, h, i):
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

# 16개 코팩터 계산 후 4×4 행렬식
det_M = m00*A00 - m01*A01 + m02*A02 - m03*A03

# Cramer의 규칙으로 직접 풀이
ddq0 = (A00*rhs0 - ...) / det_M
```

특이성 체크: $\lvert\det M\rvert < 10^{-12} \cdot (\max\text{diag})^4$

### 2.3 삼각함수 단일 계산

9개의 삼각함수 값을 한 번만 계산하여 질량행렬, 코리올리스, 중력 계산 전체에서 재사용한다.

```python
# 9개 삼각함수: c1, c12, c123, c2, c23, c3, s1, s12, s123
# forward_dynamics_fast에서 인라인으로 1회 계산
c1 = cos(q1)
c12 = cos(q1 + q2)   # = cos(phi2)
c123 = cos(q1 + q2 + q3)  # = cos(phi3)
# ... 이하 6개 동일
```

배열 기반 경로는 `compute_trig(q)` 함수 1회 호출로 처리한다.

### 2.4 크리스토펠 기호 희소 구현

코리올리스 행렬 $C$를 수치 유한차분 대신 해석적 희소 크리스토펠 기호로 계산한다.

$$h_i = \sum_{j,k} \Gamma_{ijk} \, \dot{q}_j \, \dot{q}_k$$

$\partial M / \partial x = 0$ ($M$은 카트 위치에 무관)이므로 많은 항이 소멸한다. 64회 반복 루프 대신 약 25개의 하드코딩 스칼라 연산으로 구현된다.

### 2.5 전체 루프 JIT 컴파일

시뮬레이션 루프 전체가 단일 `@njit` 함수 내에 있어 Python 인터프리터 오버헤드가 매 타임스텝마다 발생하지 않는다.

```python
@njit(cache=True)
def time_loop_fast(n_steps, dt, q0, dq0, ..., result_arrays):
    for i in range(n_steps):
        # 전체 RK4 단계가 순수 LLVM JIT 코드로 실행
        ddq0, ddq1, ddq2, ddq3 = forward_dynamics_fast(...)
        q0, q1, q2, q3, dq0, dq1, dq2, dq3 = rk4_step_fast(...)
```

---

## 3. 최적화 세부 내역

| 최적화 항목 | 최적화 전 | 최적화 후 | 속도 향상 |
|------------|---------|---------|---------|
| 코리올리스 계산 | 수치 유한차분 (8M 평가/스텝) | 해석적 희소 계산 (0 추가 평가) | **무한대** |
| 4×4 선형 풀이 | np.linalg.solve (LAPACK) | 인라인 Cramer 규칙 (코팩터) | **약 5배** |
| 배열 형태 | (4,1) 인덱싱 [i,0] | (4,) 평탄 벡터화 | **약 2배** |
| 크리스토펠 루프 | 64회 반복 ($4^3$) | 약 25개 하드코딩 스칼라 | **약 3배** |
| 시뮬레이션 루프 | Python for-loop + @njit 호출 | 전체 루프가 단일 @njit | **약 2배** |
| 게인 스케줄링 | Python 보간 | @njit 보간 | 오버헤드 없음 |
| LQR 선형화 | 3개 Python-루프 야코비안 (0.19 s) | 단일 @njit 야코비안 (0.001 s) | **약 190배** |
| JIT 웜업 | 지연 컴파일 (첫 호출 페널티) | startup 시 명시적 warmup_jit() | 예측 가능한 지연 |
| 삼각함수 계산 | forward_dynamics 당 3회 (27 호출) | 단일형 (9 호출) | **약 3배** |
| 배열 할당 | RK4 스텝당 약 38회 (힙) | 스텝당 0회 (전부 스칼라) | **약 2.7배** |
| 상태 패킹 | 매 스텝 8-벡터 pack/unpack | 직접 스칼라 전파 | 오버헤드 없음 |
| 제어 법칙 | 배열 z 생성 + 내적 | 인라인 스칼라 곱셈-덧셈 | 할당 없음 |
| 각도 래핑 | 없음 (드리프트 취약) | 제어 루프 내 매 스텝 atan2-free 래핑 | 미미 |
| ROA 동역학 | 완전 forward_dynamics 매 샘플 | 빠른 스칼라 커널 (45M 삼각함수 호출 제거) | **약 3배** |
| ROA 메모리 | 샘플당 O(N²) 할당 | 사전 할당 O(N) | **약 2배** |
| ROA 샘플링 | 균일 의사난수 | Halton 준난수 (저불일치) | 커버리지 향상 |

**종합 결과**: 15초 시뮬레이션(15,001 스텝)이 약 **15 ms** 내에 완료된다.

---

## 4. JIT 캐시 관리

### 4.1 캐시 저장 위치

Numba `@njit(cache=True)` 함수의 컴파일 결과는 각 모듈의 `__pycache__/` 디렉토리에 `.nbc`, `.nbi` 파일로 저장된다.

```
dynamics/forward_dynamics/__pycache__/
  forward_dynamics_fast.cpython-310-x86_64-linux-gnu.nbc
  forward_dynamics_fast.cpython-310-x86_64-linux-gnu.nbi
```

### 4.2 JIT 캐시 사전 빌드

첫 실행 시 JIT 컴파일에 약 2.8초가 소요된다. 프로덕션 환경이나 CI에서는 사전 빌드를 권장한다.

```bash
python prebuild_cache.py
```

이 스크립트는 모든 `@njit(cache=True)` 함수를 더미 인자로 한 번씩 호출하여 컴파일 결과를 캐시에 저장한다. 이후 실행 시 캐시에서 로드하여 약 1 ms의 초기화 지연만 발생한다.

### 4.3 캐시 무효화

다음 경우에 캐시가 자동으로 무효화된다:
- Numba 버전 변경
- 해당 `.py` 파일 수정
- Python 버전 변경

캐시를 수동으로 지우려면:

```bash
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; echo "캐시 삭제 완료"
```

### 4.4 Warmup 패턴

`simulation/warmup.py`는 메인 시뮬레이션 실행 전에 소규모 더미 실행으로 모든 JIT 함수를 트리거한다.

```python
def warmup_jit(p, q0, dq0, q_eq, K_flat, u_max, dt):
    # 3 스텝 더미 시뮬레이션으로 JIT 트리거
    _run_loop_fast(3, dt, q0[0], q0[1], q0[2], q0[3],
                   dq0[0], dq0[1], dq0[2], dq0[3],
                   q_eq, K_flat, p, np.empty(0), u_max)
```

이로써 메인 시뮬레이션의 첫 호출에서 예측할 수 없는 지연이 발생하지 않는다.

---

## 5. 파이프라인 단계별 시간

| 파이프라인 단계 | 방법 | 소요 시간 |
|--------------|------|---------|
| JIT 웜업 | 전체 @njit 함수 사전 컴파일 | **~2.8 s** (최초 1회) |
| LQR 설계 | @njit 야코비안 + scipy CARE | **~0.001 s** (캐시 적중) |
| 시뮬레이션 (15 s, dt=0.001) | 게인 스케줄링 스칼라 RK4 + 단일형 동역학 | **~0.015 s** |
| Monte Carlo (50 샘플) | ThreadPool 병렬 | **~0.05 s** |
| ROA 추정 (500~2,000 샘플) | JIT 시뮬레이션 샘플별, 적응 CI | **~5~15 s** |
| 주파수 분석 | scipy.signal | **~0.005 s** |
| **전체 (플롯 제외)** | | **~0.23 s** |

ROA 추정이 전체 파이프라인에서 가장 시간이 많이 소요되는 단계이다. 빠른 스칼라 동역학 커널과 조기 이탈 로직으로 약 3배 속도 향상이 달성되었지만, 수백에서 2,000개의 전체 시뮬레이션을 실행해야 하므로 수 초 수준의 시간이 필요하다.
