# 사용 안내 (USAGE)

카트-삼중진자 시뮬레이터의 CLI 사용법, YAML 설정, 전체 8평형점 실행 방법을 서술한다.

---

## 목차

1. [CLI 사용법](#1-cli-사용법)
2. [YAML 설정 파일](#2-yaml-설정-파일)
3. [평형점 지정 실행](#3-평형점-지정-실행)
4. [8개 평형점 일괄 실행](#4-8개-평형점-일괄-실행)
5. [Form-Switching 실행](#5-form-switching-실행)
6. [고급 기능 조합 예제](#6-고급-기능-조합-예제)
7. [출력 파일 목록](#7-출력-파일-목록)

---

## 1. CLI 사용법

시뮬레이터는 `argparse` 기반 CLI로 완전히 제어된다. 모든 물리 파라미터와 시뮬레이션 파라미터를 명령행에서 재정의할 수 있다.

```bash
# 전체 도움말 출력
python main.py --help

# 사용자 지정 물리 파라미터
python main.py --mc 3.0 --m1 1.5 --L3 0.8

# 시뮬레이션 시간 및 입력 조정
python main.py --t-end 20 --dt 0.0005 --impulse 10 --dist-amplitude 20

# iLQR 궤적 최적화 활성화
python main.py --use-ilqr --ilqr-horizon 500 --ilqr-iterations 10

# 3D 삼선형 게인 스케줄링 (175 운영점)
python main.py --gain-scheduler 3d

# 관성 스케일링 적응 Q 행렬 (Bryson 규칙)
python main.py --adaptive-q

# 기능 조합
python main.py --gain-scheduler 3d --adaptive-q --use-ilqr

# headless 모드 (matplotlib 표시 없음)
python main.py --no-display

# 로그 레벨 조정
python main.py --log-level DEBUG
```

### 전체 CLI 플래그 목록

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--mc` | float | 2.4 | 카트 질량 [kg] |
| `--m1`, `--m2`, `--m3` | float | 1.323, 1.389, 0.8655 | 링크 질량 [kg] |
| `--L1`, `--L2`, `--L3` | float | 0.402, 0.332, 0.720 | 링크 길이 [m] |
| `--g` | float | 9.81 | 중력 가속도 [m/s²] |
| `--t-end` | float | 15.0 | 시뮬레이션 시간 [s] |
| `--dt` | float | 0.001 | 적분 타임스텝 [s] |
| `--impulse` | float | 5.0 | 초기 카트 임펄스 [N·s] |
| `--dist-amplitude` | float | 15.0 | 외란 RMS 진폭 [N] |
| `--dist-bandwidth` | float | 3.0 | 외란 차단 주파수 [Hz] |
| `--u-max` | float | 200.0 | 구동기 포화 한계 [N] |
| `--seed` | int | 42 | 랜덤 시드 |
| `--equilibrium` | str | UUU | 목표 평형점 (DDD/DDU/DUD/DUU/UDD/UDU/UUD/UUU) |
| `--all-equilibria` | 플래그 | off | 8개 평형점 일괄 실행 |
| `--equilibria-list` | str | None | `--all-equilibria`와 함께 사용. 실행할 평형점을 쉼표로 구분 (예: `"UDD,UUU"`). 미지정 시 8개 전체 |
| `--use-ilqr` | 플래그 | off | iLQR 궤적 최적화 활성화 |
| `--ilqr-horizon` | int | 500 | iLQR 계획 수평선 스텝 수 |
| `--ilqr-iterations` | int | 10 | iLQR 반복 횟수 |
| `--gain-scheduler` | 선택 | 1d | 게인 스케줄러: `1d` (3차 Hermite) 또는 `3d` (삼선형 7×5×5) |
| `--adaptive-q` | 플래그 | off | 관성 스케일링 Q 행렬 사용 (Bryson 규칙) |
| `--form-switch` | 플래그 | off | Form-switching 활성화 |
| `--switch-source` | str | DDD | Form-switching 시작 평형점 |
| `--switch-target` | str | UUU | Form-switching 목표 평형점 |
| `--switch-time` | float | 30.0 | Form-switching 총 시뮬레이션 시간 [s] |
| `--k-energy` | float | 50.0 | 에너지 성형 게인 $k_e$ |
| `--config` | str | None | YAML 설정 파일 경로 |
| `--no-display` | 플래그 | off | matplotlib 화면 표시 생략 |
| `--log-level` | 선택 | INFO | 로그 레벨: DEBUG, INFO, WARNING |

---

## 2. YAML 설정 파일

재현 가능한 실험을 위해 파라미터를 YAML 파일로 저장하여 사용한다.

```bash
cp config.example.yaml config.yaml
# config.yaml 편집 후:
python main.py --config config.yaml
```

**`config.yaml` 전체 예제**:

```yaml
system:
  mc: 2.4              # 카트 질량 [kg]
  m1: 1.323            # 링크 1 질량 [kg]
  m2: 1.389            # 링크 2 질량 [kg]
  m3: 0.8655           # 링크 3 질량 [kg]
  L1: 0.402            # 링크 1 길이 [m]
  L2: 0.332            # 링크 2 길이 [m]
  L3: 0.720            # 링크 3 길이 [m]
  g: 9.81              # 중력 가속도 [m/s²]

simulation:
  t_end: 15.0          # 시뮬레이션 시간 [s]
  dt: 0.001            # 적분 타임스텝 [s]
  impulse: 5.0         # 초기 카트 임펄스 [N·s]
  dist_amplitude: 15.0 # 외란 RMS [N]
  dist_bandwidth: 3.0  # 외란 차단 주파수 [Hz]
  u_max: 200.0         # 구동기 포화 한계 [N]
  seed: 42             # 랜덤 시드

features:
  equilibrium: "UUU"        # 목표 평형점
  use_ilqr: false           # iLQR 활성화
  ilqr_horizon: 500         # iLQR 계획 수평선 스텝
  ilqr_iterations: 10       # iLQR 반복 횟수
  gain_scheduler: "1d"      # "1d" (3차 Hermite) 또는 "3d" (삼선형 7×5×5)
  adaptive_q: false         # 관성 스케일링 Q 행렬
  actuator_saturation: 200.0 # 구동기 포화 [N] (u_max와 동일)

form_switch:
  enabled: false
  source: "DDD"
  target: "UUU"
  switch_time: 30.0
  k_energy: 50.0
```

**우선순위**: CLI 플래그 > YAML 값 > 내장 기본값 순서로 적용된다.

### `actuator_saturation` 필드

`features.actuator_saturation`은 `simulation.u_max`와 동일한 효과를 가지며 YAML에서 직접 지정 가능한 별칭이다. 두 필드가 모두 지정된 경우 `u_max`가 우선한다.

```yaml
features:
  actuator_saturation: 150.0  # 150 N으로 포화 한계 축소
```

---

## 3. 평형점 지정 실행

`--equilibrium` 플래그로 8개 평형점 중 하나를 선택하여 안정화 시뮬레이션을 실행한다.

```bash
# 전체 상방 평형점 (UUU) — 기본값
python main.py --equilibrium UUU

# 링크1만 상방 (UDD)
python main.py --equilibrium UDD --t-end 20

# 전체 하방 (DDD) — 자연 안정 평형점
python main.py --equilibrium DDD

# 링크1,2 상방 (UUD)
python main.py --equilibrium UUD --impulse 3 --dist-amplitude 10
```

각 평형점에서 자동으로 해당 구성에 맞는 LQR 게인이 재계산된다. 직립 링크(Up 방향)에는 각도 페널티 100, 하방 링크에는 10이 부여된 Q 행렬이 사용된다.

---

## 4. 8개 평형점 일괄 실행

`--all-equilibria` 플래그를 사용하면 8개 평형점 전체를 순차적으로 시뮬레이션하고 결과를 비교한다.

```bash
# 기본 설정으로 전체 실행
python main.py --all-equilibria

# 빠른 테스트 (시간 단축)
python main.py --all-equilibria --t-end 10 --no-display

# 고정밀 설정
python main.py --all-equilibria --t-end 20 --gain-scheduler 3d

# 부분 집합만 실행 (--equilibria-list)
python main.py --all-equilibria --equilibria-list "UUU,DDD" --no-display
python main.py --all-equilibria --equilibria-list "UDD,UUD,UUU"
```

**출력 구조**:

```
images/
├── DDD/
│   ├── animation.gif
│   ├── dynamics_analysis.png
│   ├── control_analysis.png
│   ├── lqr_verification.png
│   └── roa_analysis.png
├── DDU/
│   └── ...
└── UUU/
    └── ...
```

콘솔에는 다음과 같은 요약 테이블이 출력된다:

```
평형점  ROA성공률  정착시간(s)  최대Re(λ)
DDD     100.0%     9.58        -1.45
DDU      55.1%     9.89        -0.87
...
UUU      20.9%     9.95        -0.62
```

---

## 5. Form-Switching 실행

에너지 기반 swing-up과 LQR catch를 결합하여 한 평형점에서 다른 평형점으로 전환한다.

```bash
# DDD → UUU 기본 전환 (BFS 최단 경로 자동 계획)
python main.py --form-switch

# 특정 경로 지정
python main.py --form-switch --switch-source DDD --switch-target UUU

# 에너지 게인 및 시뮬레이션 시간 조정
python main.py --form-switch --k-energy 80 --switch-time 40

# 부분 전환 (UDD → UUU)
python main.py --form-switch --switch-source UDD --switch-target UUU --switch-time 25
```

**BFS 경로 계획 예시**:

| 출발 | 도착 | 최단 경로 | 단계 수 |
|------|------|-----------|---------|
| DDD | UUU | DDD→UDD→UUD→UUU | 3 |
| DDD | DDU | DDD→DDU | 1 |
| UDD | UUU | UDD→UUD→UUU | 2 |
| DUD | UUU | DUD→UUD→UUU | 2 |

FSM 모드 전환 조건:

- **Swing-Up → LQR Catch**: Lyapunov 값 $V < \rho_{\text{in}}$
- **LQR Catch → Swing-Up** (이탈): $V > \rho_{\text{out}}$
- **LQR Catch → Stabilized**: 편차 합계 < 0.1이 500 스텝 연속

---

## 6. 고급 기능 조합 예제

### 게인 스케줄링 + 적응 Q 행렬

```bash
python main.py \
  --gain-scheduler 3d \
  --adaptive-q \
  --equilibrium UUU \
  --t-end 20
```

`--gain-scheduler 3d`는 (θ₁, θ₂, θ₃) 3축에 걸쳐 7×5×5 = 175개 운영점에서 LQR 게인을 사전 계산하고, 런타임에 삼선형 보간으로 게인을 적용한다. `--adaptive-q`는 링크 관성 텐서에 비례하는 Q 행렬을 자동 산출한다(Bryson 규칙).

### iLQR + 3D 게인 스케줄링

```bash
python main.py \
  --use-ilqr --ilqr-horizon 500 --ilqr-iterations 10 \
  --gain-scheduler 3d \
  --t-end 30
```

iLQR은 실제 비선형 궤적을 따라 시변(time-varying) 게인을 산출하므로, 초기 편차가 큰 경우 고정점 LQR 대비 성능이 향상된다. backward pass에서 행렬 지수함수(expm) 기반 이산화를 사용한다.

### 강건 안정성 검증

```bash
python main.py --equilibrium UUU --log-level DEBUG
```

`--log-level DEBUG`로 실행하면 `compute_robust_stability` 결과가 콘솔에 출력된다: 공칭 게인 고정 상태에서 ±10% 질량/길이 섭동 200 샘플에 대한 LHP 유지율이 보고된다.

### 채널별 RMS 분석

시뮬레이션 후 `compute_channel_rms`가 자동 호출되어 각 상태 채널(x, θ₁, θ₂, θ₃, ẋ, θ̇₁, θ̇₂, θ̇₃)의 정상상태 RMS를 요약 테이블로 출력한다.

### JIT 캐시 사전 빌드

첫 실행 시 JIT 컴파일에 약 3초가 소요된다. 프로덕션 환경이나 CI 환경에서는 사전 빌드를 권장한다.

```bash
python prebuild_cache.py
```

이후 실행 시 캐시에서 로드하여 컴파일 지연이 없다.

---

## 7. 출력 파일 목록

각 실행 시 `images/` 디렉토리에 자동으로 다음 파일들이 저장된다.

| 파일 | 내용 |
|------|------|
| `animation.gif` | 30 fps 카트-진자 애니메이션 |
| `dynamics_analysis.png` | 10개 서브플롯: 위치, 속도, 가속도, 에너지, 위상도, 제어력, 관절 반력 |
| `control_analysis.png` | 8개 서브플롯: Bode, Nyquist, 민감도, 극점, 스텝 응답, 주파수 스펙트럼 |
| `lqr_verification.png` | 8개 서브플롯: Lyapunov, Riccati 고유값, LQR 비용, Kalman 부등식, Monte Carlo |
| `roa_analysis.png` | 4개 서브플롯: ROA 산포도, 성공률, GS 고유값, P 조건수 |
| `comparison_analysis.png` | LQR vs PD vs 극점 배치 비교 (시간영역, 주파수영역) |

`--all-equilibria` 실행 시에는 각 평형점별로 `images/{구성명}/` 하위 디렉토리에 저장된다.
