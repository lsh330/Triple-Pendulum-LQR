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
ilqr.py gravity/       frequency/  dynamics_plots/             output_manager.py
gain_scheduling.py     state/      control_plots/              data_logger.py
swing_up/              lqr_verif/  lqr_plots/
supervisor/            region_of_  switching_plots/
                       attraction
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

- 첫 실행: JIT 컴파일 (~10-30ms)
- 이후 실행: 캐시에서 로드 (~1ms)
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

- **스칼라 경로**: 삼각함수 12회, 곱셈/덧셈 ~150회, 할당 0회
- **배열 경로**: numpy 배열 연산, 가독성 우선, 야코비안 수치미분에 적합

`test_dynamics.py`의 `test_forward_dynamics_consistency`가 두 경로의 일치를 검증.

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

모든 데이터가 float64 numpy 배열로 직렬화되어 `@njit` 루프에 전달됨.

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

## 6. 테스트 구조

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

**180개 테스트**, 전부 PASS. `pytest tests/ -v`로 실행.

### 검증 기준

| 항목 | 허용치 |
|------|--------|
| 에너지 보존 (u=0) | drift < 0.01% / 초 |
| G(q_eq) = 0 | < 1e-12 |
| CARE 잔차 | < 1e-6 |
| 폐루프 극점 | Re(λ) < 0 |
| Coriolis 반대칭 | z^T(Ṁ-2C)z < 1e-5 |

---

## 7. 성능 벤치마크

| 항목 | 성능 |
|------|------|
| 표준 시뮬레이션 (15초) | 2,674× 실시간 |
| Switching 시뮬레이션 (30초) | 2,010× 실시간 |
| 에너지 계산 단일 호출 | 536 ns |
| ROA 추정 (300 샘플) | 0.39 s |

`python benchmark.py`로 측정 가능.
