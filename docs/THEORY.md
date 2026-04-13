# 삼중진자 시뮬레이션 이론 문서

카트 위 삼중 역진자(Cart + Triple Inverted Pendulum) LQR 최적 안정화 시뮬레이션의 수학적 기반과 제어 이론을 서술한다.

---

## 목차

1. [시스템 기술](#1-시스템-기술)
2. [라그랑주 역학](#2-라그랑주-역학)
3. [질량행렬](#3-질량행렬)
4. [코리올리스 및 원심력 벡터](#4-코리올리스-및-원심력-벡터)
5. [중력 벡터](#5-중력-벡터)
6. [8개 평형 구성](#6-8개-평형-구성)
7. [선형화 및 제어가능성](#7-선형화-및-제어가능성)
8. [LQR 설계](#8-lqr-설계)
9. [에너지 기반 Swing-Up 제어](#9-에너지-기반-swing-up-제어)
10. [Form-Switching Supervisor](#10-form-switching-supervisor)
11. [수치 방법](#11-수치-방법)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 시스템 기술

### 1.1 물리 구성

질량 $m_c$인 카트가 수평 레일 위를 이동하며, 카트에 3개의 균일한 강체 링크가 직렬로 연결되어 있다. 각 링크의 파라미터는 다음과 같다.

| 링크 | 질량 | 길이 | 기준값 (Medrano-Cerda) |
|------|------|------|------------------------|
| 1    | $m_1$ | $L_1$ | 1.323 kg, 0.402 m |
| 2    | $m_2$ | $L_2$ | 1.389 kg, 0.332 m |
| 3    | $m_3$ | $L_3$ | 0.8655 kg, 0.720 m |
| 카트 | $m_c$ | —     | 2.4 kg               |

링크 3은 가장 길지만(0.72 m) 가장 가벼워(0.87 kg) 선단이 외란에 매우 취약하다.

시스템은 **4-DOF, 1-입력** 언더액추에이티드(underactuated) 구조다. 카트에 가해지는 수평력 $F$ 하나만으로 4개의 자유도를 제어해야 한다.

### 1.2 좌표 규약

일반화 좌표 벡터는 다음과 같이 정의된다.

$$\mathbf{q} = \begin{bmatrix} x \\ \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix}$$

<!-- source: parameters/equilibrium.py:1-15 -->

- $x$: 카트의 수평 위치 [m]
- $\theta_1$: 링크 1의 **절대각** (수직 하방 기준)
- $\theta_2$: 링크 2의 **상대각** (링크 1 기준)
- $\theta_3$: 링크 3의 **상대각** (링크 2 기준)

**절대각(absolute angle)** $\phi_k$는 상대각의 누적합이다.

$$\phi_k = \sum_{i=1}^{k} \theta_i \qquad (k = 1, 2, 3)$$

따라서 $\phi_1 = \theta_1$, $\phi_2 = \theta_1 + \theta_2$, $\phi_3 = \theta_1 + \theta_2 + \theta_3$.

**평형 기준**: $\phi = 0$은 링크가 수직 **하방**을 향하는 안정 위치(Down), $\phi = \pi$는 수직 **상방**을 향하는 불안정 위치(Up)를 의미한다.

<!-- source: parameters/equilibrium.py:6-9 -->

코드 구현에서 삼각 함수는 다음 9개 값으로 집약된다.

<!-- source: dynamics/trigonometry.py:6-19 -->
```python
c1, c12, c123, c2, c23, c3, s1, s12, s123 = compute_trig(q)
```

여기서 `c12 = cos(φ2)`, `c123 = cos(φ3)`, `c2 = cos(θ2)`, `c23 = cos(θ2+θ3)`.

---

## 2. 라그랑주 역학

### 2.1 운동에너지

각 링크의 질량중심 위치(카트 원점 기준):

$$x_{cm,k} = x + \sum_{i=1}^{k-1} L_i \sin\phi_i + \frac{L_k}{2} \sin\phi_k$$

$$y_{cm,k} = -\sum_{i=1}^{k-1} L_i \cos\phi_i - \frac{L_k}{2} \cos\phi_k$$

균일 봉의 관성 모멘트 $I_k = m_k L_k^2 / 12$를 포함한 전체 운동에너지:

$$T = \frac{1}{2} m_c \dot{x}^2 + \sum_{k=1}^{3} \left[ \frac{1}{2} m_k \left( \dot{x}_{cm,k}^2 + \dot{y}_{cm,k}^2 \right) + \frac{1}{2} I_k \dot{\phi}_k^2 \right]$$

이를 전개하면 $T = \frac{1}{2} \dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}}$ 형태가 된다.

### 2.2 위치에너지와 PE 부호 규약

중력 위치에너지의 물리적 정의(카트 높이를 기준점 0으로 설정):

$$V_{\text{physical}} = \sum_{k=1}^{3} m_k g \, y_{cm,k}$$

$y_{cm,k}$가 코사인항을 포함하므로, 코드에서는 아래와 같은 $E_{\mathrm{code}}$를 내부 계산에 활용한다.

$$E_{\mathrm{code}} = \sum_{k=1}^{3} gg_k \cos\phi_k$$

**부호 관계**: $V_{\text{physical}} = -E_{\mathrm{code}}$

따라서 물리적 전체 에너지는:

$$E = T + V_{\text{physical}} = T - E_{\mathrm{code}} = \mathrm{KE} - E_{\mathrm{code}}$$

<!-- source: control/swing_up/energy_computation.py:3-18 -->

Down 평형($\phi_k = 0$)에서 $E_{\mathrm{code}} > 0$ (양수)이며 $V_{\text{physical}} < 0$.

Up 평형($\phi_k = \pi$)에서 $\cos\phi_k = -1$이므로 $E_{\mathrm{code}} < 0$이며 $V_{\text{physical}} > 0$.

### 2.3 오일러-라그랑주 방정식

라그랑지안 $\mathcal{L} = T - V_{\text{physical}}$에서 유도되는 운동방정식:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i} = \tau_i$$

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + G(\mathbf{q}) = \boldsymbol{\tau}$$

<!-- source: dynamics/forward_dynamics/forward_dynamics_fast.py:119-124 -->

일반화 힘 벡터 $\boldsymbol{\tau} = [F, 0, 0, 0]^T$: 카트에만 수평력 $F$가 작용하고, 관절에는 외부 토크가 없다.

---

## 3. 질량행렬 $M(\mathbf{q})$

### 3.1 유도 파라미터 (13개)

물리 파라미터로부터 13개의 유도 파라미터가 계산된다.

<!-- source: parameters/derived.py:4-19 -->

| 인덱스 | 심볼 | 정의 | 물리적 의미 |
|--------|------|------|-------------|
| `p[0]` | $M_t$ | $m_c + m_1 + m_2 + m_3$ | 전체 질량 |
| `p[1]` | $gx_1$ | $(m_1/2 + m_2 + m_3)L_1$ | 카트-링크1 결합 계수 |
| `p[2]` | $gx_2$ | $(m_2/2 + m_3)L_2$ | 카트-링크2 결합 계수 |
| `p[3]` | $gx_3$ | $(m_3/2)L_3$ | 카트-링크3 결합 계수 |
| `p[4]` | $a_1$ | $(m_1/3 + m_2 + m_3)L_1^2$ | 링크1 회전 관성 계수 |
| `p[5]` | $a_2$ | $(m_2/3 + m_3)L_2^2$ | 링크2 회전 관성 계수 |
| `p[6]` | $a_3$ | $(m_3/3)L_3^2$ | 링크3 회전 관성 계수 |
| `p[7]` | $b_1$ | $(m_2/2 + m_3)L_1 L_2$ | 링크1-링크2 결합 계수 |
| `p[8]` | $b_2$ | $(m_3/2)L_1 L_3$ | 링크1-링크3 결합 계수 |
| `p[9]` | $b_3$ | $(m_3/2)L_2 L_3$ | 링크2-링크3 결합 계수 |
| `p[10]` | $gg_1$ | $gx_1 \cdot g$ | 링크1 중력 상수 |
| `p[11]` | $gg_2$ | $gx_2 \cdot g$ | 링크2 중력 상수 |
| `p[12]` | $gg_3$ | $gx_3 \cdot g$ | 링크3 중력 상수 |

### 3.2 카트-링크 결합항 $m_{xi}$

카트 병진운동과 링크 회전운동의 관성 결합항은 절대각의 코사인으로 표현된다.

<!-- source: dynamics/mass_matrix/cart_link_coupling.py:6-9 -->

$$m_{x1} = gx_1 \cos\phi_1 + gx_2 \cos\phi_2 + gx_3 \cos\phi_3$$

$$m_{x2} = gx_2 \cos\phi_2 + gx_3 \cos\phi_3$$

$$m_{x3} = gx_3 \cos\phi_3$$

### 3.3 진자 블록 $M_{ij}$

진자 블록은 **상대각** $\theta_2$, $\theta_3$의 코사인만으로 표현된다.

<!-- source: dynamics/mass_matrix/pendulum_block.py:13-20 -->

$$M_{11} = a_1 + a_2 + a_3 + 2b_1 \cos\theta_2 + 2b_2 \cos(\theta_2+\theta_3) + 2b_3 \cos\theta_3$$

$$M_{12} = a_2 + a_3 + b_1 \cos\theta_2 + b_2 \cos(\theta_2+\theta_3) + 2b_3 \cos\theta_3$$

$$M_{13} = a_3 + b_2 \cos(\theta_2+\theta_3) + b_3 \cos\theta_3$$

$$M_{22} = a_2 + a_3 + 2b_3 \cos\theta_3$$

$$M_{23} = a_3 + b_3 \cos\theta_3$$

$$M_{33} = a_3$$

### 3.4 전체 질량행렬

$$M(\mathbf{q}) = \begin{bmatrix}
M_t & m_{x1} & m_{x2} & m_{x3} \\
m_{x1} & M_{11} & M_{12} & M_{13} \\
m_{x2} & M_{12} & M_{22} & M_{23} \\
m_{x3} & M_{13} & M_{23} & M_{33}
\end{bmatrix}$$

<!-- source: dynamics/mass_matrix/assembly.py:10-35 -->

$M(\mathbf{q})$는 대칭 양정치(symmetric positive definite) 행렬이다. $M$은 $x$(카트 위치)에 무관하며, $\theta_2$, $\theta_3$에만 의존한다(상단 행/열은 $\phi_1, \phi_2, \phi_3$에도 의존).

---

## 4. 코리올리스 및 원심력 벡터 $C(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}$

### 4.1 크리스토펠 기호

코리올리스·원심력 벡터 $\mathbf{h} = C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$의 각 성분은 제1종 크리스토펠 기호(Christoffel symbols of the first kind)로 계산된다.

<!-- source: dynamics/coriolis/christoffel.py:49-50 -->

$$h_i = \sum_{j,k} \Gamma_{ijk} \, \dot{q}_j \, \dot{q}_k$$

$$\Gamma_{ijk} = \frac{1}{2} \left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right)$$

### 4.2 희소성(Sparsity) 활용

$\partial M / \partial q_0 = 0$ ($M$은 카트 위치 $x$에 무관)이므로, 많은 크리스토펠 항이 소멸한다. 또한 $\partial M^{\text{pend}} / \partial \theta_1 = 0$ (진자 블록이 $\theta_1$에 무관)이다.

**주요 편미분** (스칼라 구현과 동일한 형태):

<!-- source: dynamics/coriolis/christoffel.py:27-46 -->

$k=1$ ($\partial M / \partial \theta_1$): 카트-링크 결합행만 비영(non-zero)

$$\frac{\partial m_{x1}}{\partial \theta_1} = -gx_1 s_1 - gx_2 s_{12} - gx_3 s_{123}$$

$k=2$ ($\partial M / \partial \theta_2$): 카트-링크 결합행 + 진자 블록 일부

$$\frac{\partial M_{11}}{\partial \theta_2} = -2b_1 s_2 - 2b_2 s_{23}, \quad \frac{\partial M_{12}}{\partial \theta_2} = -b_1 s_2 - b_2 s_{23}$$

$k=3$ ($\partial M / \partial \theta_3$): 카트-링크 결합행 + 진자 블록 전체

$$\frac{\partial M_{22}}{\partial \theta_3} = -2b_3 s_3, \quad \frac{\partial M_{23}}{\partial \theta_3} = -b_3 s_3$$

### 4.3 $\dot{M} - 2C$ 반대칭 성질

행렬 $N = \dot{M} - 2C$는 반대칭(skew-symmetric)이다. 즉 $N + N^T = 0$. 이 성질은 에너지 보존 확인과 Lyapunov 안정성 분석에 활용된다.

### 4.4 스칼라 구현

코드에서 `h[0]`~`h[3]`은 배열 할당 없이 순수 스칼라 연산으로 계산된다.

<!-- source: dynamics/forward_dynamics/forward_dynamics_fast.py:89-113 -->

```python
h0 = ((dmx1_d1 * d1 + dmx2_d1 * d2 + dmx3_d1 * d3) * d1
     + (dmx1_d2 * d1 + dmx2_d2 * d2 + dmx3_d2 * d3) * d2
     + (dmx1_d3 * d1 + dmx2_d3 * d2 + dmx3_d3 * d3) * d3)
```

---

## 5. 중력 벡터 $G(\mathbf{q})$

중력 벡터는 절대각의 사인함수로만 표현된다.

<!-- source: dynamics/gravity/gravity_vector.py:14-19 -->

$$G(\mathbf{q}) = \begin{bmatrix}
0 \\
gg_1 \sin\phi_1 + gg_2 \sin\phi_2 + gg_3 \sin\phi_3 \\
gg_2 \sin\phi_2 + gg_3 \sin\phi_3 \\
gg_3 \sin\phi_3
\end{bmatrix}$$

$G_0 = 0$: 카트의 수평 방향에는 중력이 작용하지 않는다.

**평형 조건**: $G(\mathbf{q}^*) = 0$이 되려면 $\sin\phi_k^* = 0$이어야 하므로 $\phi_k^* \in \{0, \pi\}$. 이것이 8개 평형점이 존재하는 근거이다.

---

## 6. 8개 평형 구성

### 6.1 평형점 분류

각 링크의 절대 방향이 Down($\phi=0$) 또는 Up($\phi=\pi$)인 경우의 조합이므로 $2^3 = 8$개의 평형점이 존재한다.

<!-- source: parameters/equilibrium.py:21-30 -->

| 이름 | $\phi_1$ | $\phi_2$ | $\phi_3$ | 비트(D=0,U=1) | 물리적 의미 |
|------|----------|----------|----------|---------------|-------------|
| DDD  | 0        | 0        | 0        | 000           | 전체 하방 (가장 안정) |
| DDU  | 0        | 0        | $\pi$    | 001           | 링크3만 상방 |
| DUD  | 0        | $\pi$    | 0        | 010           | 링크2만 상방 |
| DUU  | 0        | $\pi$    | $\pi$    | 011           | 링크2,3 상방 |
| UDD  | $\pi$    | 0        | 0        | 100           | 링크1만 상방 |
| UDU  | $\pi$    | 0        | $\pi$    | 101           | 링크1,3 상방 |
| UUD  | $\pi$    | $\pi$    | 0        | 110           | 링크1,2 상방 |
| UUU  | $\pi$    | $\pi$    | $\pi$    | 111           | 전체 상방 (가장 불안정) |

### 6.2 상대각으로의 변환

시뮬레이션 좌표 $\theta_i$는 상대각이므로 변환이 필요하다.

<!-- source: parameters/equilibrium.py:53-58 -->

$$\theta_1 = \phi_1, \quad \theta_2 = \phi_2 - \phi_1, \quad \theta_3 = \phi_3 - \phi_2$$

예: UUU 구성은 $(\phi_1, \phi_2, \phi_3) = (\pi, \pi, \pi)$이므로 $(\theta_1, \theta_2, \theta_3) = (\pi, 0, 0)$.

### 6.3 위치에너지 레벨

각 평형점에서의 물리적 위치에너지 $V_{\text{physical}} = -E_{\mathrm{code}}$:

<!-- source: parameters/equilibrium.py:67-92 -->

$$V_{\text{physical}}(\text{config}) = -\left( m_1 g h_1 + m_2 g h_2 + m_3 g h_3 \right)$$

여기서 $h_k = L_1 \cos\phi_1 + \cdots + (L_k/2)\cos\phi_k$는 링크 $k$ 질량중심의 높이.

DDD가 가장 낮은 에너지(전체 하방), UUU가 가장 높은 에너지(전체 상방)를 가진다.

---

## 7. 선형화 및 제어가능성

### 7.1 상태공간 선형화

상태 편차 $\mathbf{z} = (\delta\mathbf{q}, \delta\dot{\mathbf{q}})^T \in \mathbb{R}^8$에 대한 선형 근사:

$$\dot{\mathbf{z}} = A\mathbf{z} + Bu$$

$$A = \begin{bmatrix} \mathbf{0}_{4\times4} & I_{4\times4} \\ A_q & A_{\dot{q}} \end{bmatrix}, \qquad B = \begin{bmatrix} \mathbf{0}_{4\times1} \\ B_u \end{bmatrix}$$

<!-- source: control/linearization/analytical_jacobian.py:255-274 -->

### 7.2 해석적 야코비안 $A_q$

평형점에서 $\dot{\mathbf{q}} = 0$이면 코리올리스항이 소멸하고 $\ddot{\mathbf{q}} = M^{-1}(\boldsymbol{\tau} - G)$이다.

$$A_q = \frac{\partial \ddot{\mathbf{q}}}{\partial \mathbf{q}}\bigg|_{\mathbf{q}^*, \dot{\mathbf{q}}^*} = -M^{-1} \frac{\partial G}{\partial \mathbf{q}} - M^{-1} \left(\frac{\partial M}{\partial \mathbf{q}} M^{-1}(\boldsymbol{\tau}-G)\right)$$

행렬 역함수의 편미분 항등식 $\partial M^{-1}/\partial q_k = -M^{-1}(\partial M/\partial q_k)M^{-1}$을 적용하면:

<!-- source: control/linearization/analytical_jacobian.py:107-161 -->

```python
col = M_inv @ (-dGdq[:, k_col] - dM @ (M_inv @ rhs))
```

**중력 야코비안** $\partial G / \partial \mathbf{q}$: 절대각 $\phi_k = \sum_{i=1}^k \theta_i$에 의존하므로 $\partial\phi_k/\partial\theta_j = 1$($j \le k$), $0$($j > k$)의 하삼각 구조를 가진다.

### 7.3 $B_u$ 해석적 계산

카트에만 입력이 가해지므로:

$$B_u = M^{-1} \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

<!-- source: control/linearization/analytical_jacobian.py:165-206 -->

### 7.4 $A_{\dot{q}}$ 수치 미분

코리올리스 야코비안 $\partial(C\dot{\mathbf{q}})/\partial\dot{\mathbf{q}}|_{\dot{\mathbf{q}}=0}$은 크리스토펠 기호 전체를 필요로 하므로, 중앙차분 수치 미분을 사용한다.

<!-- source: control/linearization/analytical_jacobian.py:209-237 -->

적응 스텝: $h_j = \varepsilon_{\text{mach}}^{1/3} \cdot \max(1, \lvert dq_j \rvert) \approx 6.055 \times 10^{-6}$

### 7.5 Hautus 제어가능성 판별

시스템 $(A, B)$의 제어가능성 행렬:

$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \cdots & A^7B \end{bmatrix} \in \mathbb{R}^{8 \times 8}$$

$\text{rank}(\mathcal{C}) = 8$이면 제어가능. 코드에서 CARE 풀기 전에 이 조건을 검증한다.

<!-- source: control/riccati/solve_care.py:11-18 -->

8개 평형점 모두에서 $(A, B)$가 제어가능함이 수치적으로 확인된다.

---

## 8. LQR 설계

### 8.1 LQR 비용 함수

무한 수평선 이차 비용을 최소화한다.

$$J = \int_0^\infty \left( \mathbf{z}^T Q \mathbf{z} + u^T R u \right) dt$$

기본값:

<!-- source: control/cost_matrices/default_Q.py:4-6 -->
<!-- source: control/cost_matrices/default_R.py:4-6 -->

$$Q = \mathrm{diag}(10,\, 100,\, 100,\, 100,\, 1,\, 10,\, 10,\, 10)$$

$$R = 0.01$$

직립 링크에는 각도 페널티 100, 하방 링크에는 10을 부여하는 평형점 적응 $Q$ 행렬도 제공된다.

<!-- source: control/cost_matrices/default_Q.py:41-76 -->

### 8.2 CARE 풀이

연속 대수 리카티 방정식(Continuous Algebraic Riccati Equation):

$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$

<!-- source: control/riccati/solve_care.py:23-48 -->

SciPy `solve_continuous_are`를 사용하여 양정치 해 $P \succ 0$을 구한다. 풀이 후 $P$의 최소 고유값으로 양정치를 검증한다.

### 8.3 최적 게인 행렬

$$K = R^{-1} B^T P, \qquad u = -K\mathbf{z}$$

$K \in \mathbb{R}^{1 \times 8}$은 8개의 스칼라 게인으로 구성된다.

### 8.4 Lyapunov 안정성 보장

CARE 해 $P \succ 0$은 폐루프 시스템의 Lyapunov 함수 $V(\mathbf{z}) = \mathbf{z}^T P \mathbf{z}$를 직접 제공한다.

$$\dot{V} = -\mathbf{z}^T (Q + K^T R K) \mathbf{z} \lt 0 \quad \forall \, \mathbf{z} \neq \mathbf{0}$$

이는 선형화된 폐루프 시스템의 전역 점근 안정성을 보장한다.

---

## 9. 에너지 기반 Swing-Up 제어

### 9.1 목표 에너지 $E^*$

목표 평형점에서 운동에너지(KE) $= 0$이므로:

$$E^* = V_{\text{physical}}^* = -E_{\mathrm{code}}^* = -(gg_1\cos\phi_1^* + gg_2\cos\phi_2^* + gg_3\cos\phi_3^*)$$

<!-- source: control/swing_up/energy_computation.py:100-121 -->

### 9.2 현재 에너지 계산

전체 역학 에너지의 스칼라 계산 (배열 할당 없음):

<!-- source: control/swing_up/energy_computation.py:24-97 -->

$$E = \underbrace{\frac{1}{2}\dot{\mathbf{q}}^T M(\mathbf{q})\dot{\mathbf{q}}}_{\mathrm{KE}} - \underbrace{(gg_1\cos\phi_1 + gg_2\cos\phi_2 + gg_3\cos\phi_3)}_{E_{\mathrm{code}}}$$

### 9.3 에너지 성형 제어법칙

$$u = k_e (E^* - E) \dot{x}$$

<!-- source: control/swing_up/energy_controller.py:46-57 -->
<!-- source: simulation/loop/time_loop_switching.py:239-241 -->

$k_e > 0$는 에너지 성형 게인, $\dot{x}$는 카트 속도이다.

### 9.4 Lyapunov 안정성 증명

Lyapunov 후보 함수 $V_{\text{lyap}} = \frac{1}{2}(E - E^*)^2 \ge 0$에 대해:

$$\dot{V}_{\text{lyap}} = (E - E^*)\dot{E}$$

에너지 변화율 $\dot{E} = F \cdot \dot{x} = u \cdot \dot{x}$ (카트력이 유일한 비보존력)를 대입하면:

<!-- source: control/swing_up/energy_controller.py:6-14 -->

$$\dot{V}_{\text{lyap}} = (E - E^*) \cdot k_e(E^* - E)\dot{x}^2 = -k_e(E - E^*)^2 \dot{x}^2 \le 0$$

따라서 $V_{\text{lyap}}$는 단조감소(non-increasing)하며 $E \to E^*$으로 수렴하는 경향이 있다.

($\dot{x} = 0$에서 $\dot{V}_{\text{lyap}} = 0$: LaSalle 불변집합 원리로 분석 가능)

---

## 10. Form-Switching Supervisor

### 10.1 FSM 3-상태 구조

한 평형점에서 다른 평형점으로의 전환을 위해 유한상태기계(FSM)가 동작한다.

<!-- source: simulation/loop/time_loop_switching.py:1-16 -->

| 모드 상수 | 값 | 동작 |
|-----------|---|------|
| `MODE_SWING_UP` | 0 | 에너지 성형: $u = k_e(E^* - E)\dot{x}$ |
| `MODE_LQR_CATCH` | 1 | LQR: $u = -K\mathbf{z}$, ROA 히스테리시스 모니터링 |
| `MODE_STABILIZED` | 2 | LQR 유지; 500 스텝 연속 수렴 후 다음 단계로 전진 |

### 10.2 Lyapunov 레벨셋 전환 조건

현재 상태의 Lyapunov 값:

$$V(\mathbf{z}) = \mathbf{z}^T P \mathbf{z}$$

<!-- source: simulation/loop/time_loop_switching.py:32-114 -->

전환 조건 (히스테리시스 포함):

- **Swing-Up → LQR Catch**: $V \lt \rho_{\text{in}}$
- **LQR Catch → Swing-Up** (이탈): $V \gt \rho_{\text{out}}$
- **LQR Catch → Stabilized**: 편차 합계가 0.1 미만인 상태가 500 스텝 연속 지속

### 10.3 ROA 추정 및 임계값

Monte Carlo 전방 시뮬레이션(기본값: 300 샘플, 3초 수평선)으로 수렴 초기조건들의 최대 Lyapunov 값 $\rho_{\max}$를 추정한다.

<!-- source: control/supervisor/roa_estimation.py:99-222 -->

$$\rho = \alpha \cdot \rho_{\max}, \quad \alpha = 0.8 \quad \text{(safety factor)}$$

$$\rho_{\text{in}} = 0.5\rho, \quad \rho_{\text{out}} = 0.8\rho$$

### 10.4 BFS 전환 경로 계획

8개 평형점은 3비트 정수로 인코딩된다(D=0, U=1).

| 구성 | 비트 패턴 | 정수 |
|------|-----------|------|
| DDD | 000 | 0 |
| DDU | 001 | 1 |
| DUD | 010 | 2 |
| DUU | 011 | 3 |
| UDD | 100 | 4 |
| UDU | 101 | 5 |
| UUD | 110 | 6 |
| UUU | 111 | 7 |

두 구성의 **해밍 거리**(Hamming distance)가 1인 경우 인접(adjacent)하다: 링크 방향 하나만 바뀐다. BFS(너비우선탐색)로 최소 홉(hop) 전환 경로를 계획한다.

<!-- source: control/supervisor/transition_graph.py:1-124 -->

```
DDD → UUU 최단 경로 예시: DDD -> UDD -> UUD -> UUU (3 단계)
```

### 10.5 JIT 루프를 위한 데이터 패킹

Python 오버헤드 없이 `@njit` 루프에서 동작하도록 모든 슈퍼바이저 데이터를 평탄한 NumPy 배열로 직렬화한다.

<!-- source: control/supervisor/form_switch_supervisor.py:86-150 -->

```python
{
    "n_stages": int,          # 전환 단계 수
    "all_q_eq":  (n, 4),      # 각 단계의 목표 평형점
    "all_K_flat": (n, 8),     # LQR 게인 벡터 (평탄화)
    "all_P_flat": (n, 8, 8),  # Lyapunov 행렬
    "all_E_target": (n,),     # 목표 에너지
    "all_rho_in":  (n,),      # ROA 진입 임계값
    "all_rho_out": (n,),      # ROA 이탈 임계값
}
```

---

## 11. 수치 방법

### 11.1 RK4 고정 스텝 적분

4차 룽게-쿠타(Runge-Kutta) 방법, 기본 타임스텝 $\Delta t = 0.001$ s:

<!-- source: dynamics/forward_dynamics/forward_dynamics_fast.py:163-212 -->

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

스칼라 상태(scalar state) 인터페이스로 구현되어 RK4 4단계 전체에서 힙 할당이 발생하지 않는다.

### 11.2 4×4 행렬식/역행렬 인라인 풀이

$M\ddot{\mathbf{q}} = \mathbf{r}$의 풀이는 코팩터 전개(cofactor expansion)를 직접 인라인으로 구현하여 LAPACK 호출을 제거한다.

<!-- source: dynamics/forward_dynamics/forward_dynamics_fast.py:127-158 -->

- 3×3 행렬식 헬퍼 `_det3`: 9개의 스칼라 곱으로 계산
- 16개의 코팩터 $A_{ij}$ 계산 후 $\det M = m_{00}A_{00} + \cdots$
- 특이성 체크: $\lvert\det M\rvert \lt 10^{-12} \cdot (\max \text{diag})^4$

### 11.3 스칼라 상태 접근 (Zero-Allocation)

`forward_dynamics_fast`는 배열 대신 8개의 스칼라를 직접 인수로 전달받는다.

<!-- source: dynamics/forward_dynamics/forward_dynamics_fast.py:20-21 -->

```python
def forward_dynamics_fast(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p):
```

이 설계로 15초 시뮬레이션(15,001 스텝)에서 약 570,000건의 소형 배열 할당이 제거된다.

### 11.4 삼각함수 단일 계산

9개의 삼각함수 값(`c1, c12, c123, c2, c23, c3, s1, s12, s123`)을 한 번만 계산하여 질량행렬, 코리올리스, 중력 계산 모두에서 재사용한다.

<!-- source: dynamics/trigonometry.py:6-19 -->

배열 기반 경로는 `compute_trig`를 호출하고, 스칼라 기반 경로(`forward_dynamics_fast`)는 동일한 값을 인라인으로 직접 계산한다.

### 11.5 NaN 발산 감지

RK4 적분 후 상태에 NaN이 발생하면 잔여 배열을 NaN으로 채우고 루프를 즉시 종료한다.

<!-- source: simulation/loop/time_loop_switching.py:304-314 -->

---

## 12. 참고 문헌

1. **Medrano-Cerda, G. A.** (1997). *Robust stabilization of a triple inverted pendulum-cart*. IEE Proceedings — Control Theory and Applications, 144(4), 315–325. [Medrano-Cerda 시스템 파라미터 출처]

2. **Åström, K. J. & Furuta, K.** (2000). Swinging up a pendulum by energy control. *Automatica*, 36(2), 287–295. [에너지 기반 swing-up 제어 이론]

3. **Glück, T., Eder, A. & Kugi, A.** (2013). Swing-up control of a triple pendulum on a cart with experimental validation. *Automatica*, 49(3), 801–808. [삼중 진자 swing-up 실험 검증]

4. **Tedrake, R.** (2022). *Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation*. MIT Press. [ROA 및 Lyapunov 방법론]

5. **Khalil, H. K.** (2002). *Nonlinear Systems* (3rd ed.). Prentice Hall. [LaSalle 불변집합 원리, Lyapunov 안정성]

6. **Anderson, B. D. O. & Moore, J. B.** (1990). *Optimal Control: Linear Quadratic Methods*. Prentice Hall. [CARE, LQR 이론]
