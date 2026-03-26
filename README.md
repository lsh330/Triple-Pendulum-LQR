# Cart + Triple Inverted Pendulum Simulator

LQR-optimal stabilization of a triple inverted pendulum on a cart under band-limited stochastic disturbance. Provides comprehensive dynamics analysis, frequency-domain control analysis, and formal LQR verification with publication-quality visualizations.

> **Benchmark system**: All physical parameters are taken from the **Medrano-Cerda triple inverted pendulum** (University of Salford, UK, 1997), one of the most widely cited experimental benchmarks in robust and optimal control literature [1].

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Installation

**Prerequisites**: Python >= 3.9 and pip.

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Numerical arrays and linear algebra |
| scipy | >= 1.10 | Riccati equation solver, frequency response |
| numba | >= 0.57 | JIT compilation for real-time dynamics |
| matplotlib | >= 3.6 | All plots and animation |
| pillow | >= 9.0 | GIF animation export |

## Configuration

`main.py` contains only the physical parameters and a single `run()` call:

```python
from parameters.config import SystemConfig
from pipeline.runner import run

# Medrano-Cerda benchmark (University of Salford, 1997)
cfg = SystemConfig(
    mc=2.4,                          # cart mass [kg]
    m1=1.323, m2=1.389, m3=0.8655,  # link masses [kg]
    L1=0.402, L2=0.332, L3=0.720,   # link lengths [m]
)

run(cfg)
```

The `run()` function accepts optional overrides:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t_end` | 15.0 | Simulation duration [s] |
| `dt` | 0.002 | Integration time step [s] |
| `impulse` | 5.0 | Initial cart impulse [N$\cdot$s] |
| `dist_amplitude` | 15.0 | Disturbance RMS amplitude [N] |
| `dist_bandwidth` | 3.0 | Disturbance cutoff frequency [Hz] |

## Benchmark Parameters

The Medrano-Cerda system [1] has been used extensively in control research since 1997. Key characteristics:

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Cart mass | $m_c$ | 2.4 | kg |
| Link 1 mass | $m_1$ | 1.323 | kg |
| Link 2 mass | $m_2$ | 1.389 | kg |
| Link 3 mass | $m_3$ | 0.8655 | kg |
| Link 1 length | $L_1$ | 0.402 | m |
| Link 2 length | $L_2$ | 0.332 | m |
| Link 3 length | $L_3$ | 0.720 | m |
| Gravity | $g$ | 9.81 | m/s² |

Notable feature: $L_3$ is the longest link (0.72 m) but lightest (0.87 kg), making the tip highly susceptible to disturbances and the system challenging to stabilize.

---

## Theory

### 1. System Description

A cart of mass $m_c$ translates along a horizontal rail. Three uniform rigid links ($m_1, L_1$), ($m_2, L_2$), ($m_3, L_3$) form a serial chain attached to the cart by revolute joints.

**Generalized coordinates**:

$$\mathbf{q} = \begin{pmatrix} x \\ \theta_1 \\ \theta_2 \\ \theta_3 \end{pmatrix}$$

- $x$ : cart horizontal position
- $\theta_1$ : absolute angle of link 1 from the downward vertical
- $\theta_2$ : relative angle of link 2 with respect to link 1
- $\theta_3$ : relative angle of link 3 with respect to link 2

**Control input**: a single horizontal force $F$ on the cart. This makes the system **underactuated** (4 DOF, 1 input).

### 2. Lagrangian Dynamics

#### 2.1 Kinematics

Absolute angles $\phi_k$ accumulate from the relative coordinates:

$$\phi_k = \sum_{i=1}^{k} \theta_i, \qquad k = 1,2,3$$

Center-of-mass position of the $k$-th link:

$$x_{cm,k} = x + \sum_{i=1}^{k-1} L_i \sin\phi_i + \frac{L_k}{2}\sin\phi_k$$

$$y_{cm,k} = -\sum_{i=1}^{k-1} L_i \cos\phi_i - \frac{L_k}{2}\cos\phi_k$$

#### 2.2 Energy

The kinetic energy includes cart translation, link CoM translation, and link rotation ($I_k = m_k L_k^2 / 12$ for uniform rods):

$$T = \frac{1}{2}m_c\dot{x}^2 + \sum_{k=1}^{3}\left[\frac{1}{2}m_k\!\left(\dot{x}_{cm,k}^2 + \dot{y}_{cm,k}^2\right) + \frac{1}{2}I_k\,\dot{\phi}_k^2\right]$$

The gravitational potential energy:

$$V = \sum_{k=1}^{3} m_k \, g \, y_{cm,k}$$

#### 2.3 Mass Matrix

The transformation from absolute to relative angular velocities uses:

$$\dot{\boldsymbol{\phi}} = J\,\dot{\boldsymbol{\theta}}, \qquad J = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{pmatrix}$$

The resulting $4 \times 4$ symmetric mass matrix $M(\mathbf{q})$:

$$M = \begin{pmatrix} M_t & m_{x1} & m_{x2} & m_{x3} \\[4pt] m_{x1} & M_{11} & M_{12} & M_{13} \\[4pt] m_{x2} & M_{12} & M_{22} & M_{23} \\[4pt] m_{x3} & M_{13} & M_{23} & M_{33} \end{pmatrix}$$

- $M_t = m_c + m_1 + m_2 + m_3$ is the total system mass
- $m_{x1}, m_{x2}, m_{x3}$ are cart-link coupling terms (functions of $\cos\phi_1, \cos\phi_2, \cos\phi_3$)
- $M_{ij}$ form the $3 \times 3$ pendulum inertia block (functions of $\cos\theta_2, \cos\theta_3, \cos(\theta_2 + \theta_3)$)

Built from three families of derived constants:

$$\alpha_i = \left(\frac{m_i}{3} + \sum_{j>i}m_j\right)L_i^2, \qquad \beta_{ij} = \left(\frac{m_j}{2} + \sum_{k>j}m_k\right)L_i L_j, \qquad \gamma_i = \left(\frac{m_i}{2} + \sum_{j>i}m_j\right)L_i$$

#### 2.4 Coriolis and Gravity

The Coriolis/centrifugal vector $\mathbf{h}$ is computed via Christoffel symbols of the first kind:

$$h_i = \sum_{j,k} \Gamma_{ijk}\,\dot{q}_j\,\dot{q}_k$$

$$\Gamma_{ijk} = \frac{1}{2}\!\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)$$

The gravity vector $\mathbf{G} = \partial V / \partial \mathbf{q}$, with $g_i = \gamma_i \cdot g$:

$$G(\mathbf{q}) = \begin{pmatrix} 0 \\[4pt] g_1\sin\phi_1 + g_2\sin\phi_2 + g_3\sin\phi_3 \\[4pt] g_2\sin\phi_2 + g_3\sin\phi_3 \\[4pt] g_3\sin\phi_3 \end{pmatrix}$$

#### 2.5 Equations of Motion

$$M(\mathbf{q})\,\ddot{\mathbf{q}} + C(\mathbf{q},\dot{\mathbf{q}})\,\dot{\mathbf{q}} + G(\mathbf{q}) = \begin{pmatrix} F \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

### 3. LQR Control Design

#### 3.1 Linearization

The system is linearized around the upright equilibrium $\mathbf{q}^* = (0, \pi, 0, 0)^T$, $\dot{\mathbf{q}}^* = \mathbf{0}$ using numerical central differences to compute the Jacobians $A_q = \partial \ddot{\mathbf{q}} / \partial \mathbf{q}$, $A_{\dot{q}} = \partial \ddot{\mathbf{q}} / \partial \dot{\mathbf{q}}$, $B_u = \partial \ddot{\mathbf{q}} / \partial u$.

The state $\mathbf{z} = (\delta\mathbf{q},\; \delta\dot{\mathbf{q}})^T \in \mathbb{R}^8$ yields:

$$\dot{\mathbf{z}} = A\,\mathbf{z} + B\,u$$

$$A = \begin{pmatrix} \mathbf{0}_{4 \times 4} & I_{4 \times 4} \\[4pt] A_q & A_{\dot{q}} \end{pmatrix}, \qquad B = \begin{pmatrix} \mathbf{0}_{4 \times 1} \\[4pt] B_u \end{pmatrix}$$

#### 3.2 LQR Optimal Gain

The LQR minimizes the infinite-horizon quadratic cost by solving the Continuous Algebraic Riccati Equation (CARE):

$$J = \int_0^\infty \!\left(\mathbf{z}^T Q\,\mathbf{z} + u^T R\,u\right) dt$$

$$A^T\!P + PA - PBR^{-1}B^T\!P + Q = 0$$

$$K = R^{-1}B^T\!P, \qquad u = -K\,\mathbf{z}$$

#### 3.3 Default Cost Weights

| State | Symbol | $Q_{ii}$ | Rationale |
|-------|--------|----------|-----------|
| Cart position | $x$ | 10 | Moderate regulation |
| Base link angle | $\theta_1$ | 100 | Primary stabilization target |
| Middle link angle | $\theta_2$ | 100 | Secondary stabilization |
| Tip link angle | $\theta_3$ | 100 | Tertiary stabilization |
| Cart velocity | $\dot{x}$ | 1 | Low penalty |
| Angular velocities | $\dot{\theta}_{1,2,3}$ | 10 | Moderate damping |

Control weight: $R = 0.01$ (permits aggressive actuation).

### 4. LQR Verification

#### 4.1 Lyapunov Stability

The CARE solution $P \succ 0$ defines a Lyapunov candidate $V(\mathbf{z}) = \mathbf{z}^T P\,\mathbf{z}$:

$$\dot{V} = -\mathbf{z}^T\!\left(Q + K^T\!RK\right)\mathbf{z} < 0 \quad \forall\; \mathbf{z} \neq \mathbf{0}$$

This guarantees **global asymptotic stability** of the linearized closed-loop system.

#### 4.2 Kalman Frequency-Domain Inequality

For SISO LQR with loop transfer function $L(s) = K(sI - A)^{-1}B$, the **return difference condition** holds:

$$\left|1 + L(j\omega)\right| \geq 1 \quad \forall\; \omega$$

This guarantees:
- **Gain margin**: $(-6\text{ dB},\; +\infty)$, i.e., the system remains stable under gain variations from 0.5 to $\infty$
- **Phase margin**: $\geq 60°$

#### 4.3 Nyquist Criterion

The open-loop plant has $n_u$ unstable poles (right half-plane eigenvalues of $A$). Closed-loop stability requires the Nyquist contour of $L(j\omega)$ to make exactly $n_u$ clockwise encirclements of $(-1 + 0j)$:

$$N_{\text{CW}} = n_u$$

### 5. Disturbance Model

Band-limited white noise, generated by FFT-filtering Gaussian noise through a 4th-order Butterworth lowpass:

$$d(t) = \mathcal{F}^{-1}\!\Big[W(j\omega) \cdot H(j\omega)\Big]$$

$$H(j\omega) = \frac{1}{1 + \left(\omega / \omega_c\right)^4}$$

where $W(j\omega)$ is the white noise spectrum and $\omega_c = 2\pi f_c$ is the cutoff angular frequency.

### 6. Numerical Integration

Classical 4th-order Runge-Kutta with fixed step $\Delta t$:

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}\!\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)$$

All dynamics functions ($M$, $C$, $G$, forward dynamics) are compiled to native code via Numba `@njit(cache=True)`.

---

## Analysis Results

All results below use the Medrano-Cerda parameters with initial impulse $= 5$ N$\cdot$s and band-limited noise (RMS $= 15$ N, $f_c = 3$ Hz).

### Dynamics Analysis

![Dynamics Analysis](images/dynamics_analysis.png)

**Cart Position** (top-left): The initial impulse displaces the cart to approximately $-0.38$ m. The LQR controller drives it back toward $x = 0$, but continuous noise causes persistent small oscillations around the origin. The cart never diverges, confirming closed-loop stability.

**Cart Velocity** (top-right): Peak velocity of $\sim 1.8$ m/s occurs at $t = 0$ due to the impulse. The controller rapidly damps this, with subsequent velocity fluctuations of $\pm 0.3$ m/s driven by the noise.

**Angle Deviations** (middle-left): Maximum deviation is $\sim 5°$ for $\theta_1$ (base link), $\sim 3°$ for $\theta_2$, and $\sim 1.2°$ for $\theta_3$. The base link shows the largest deviation because it directly couples to the cart motion. The decreasing deviation from base to tip reflects the LQR's heavy weighting ($Q = 100$) on all angles.

**Angular Velocities** (middle-right): Peak angular rates reach $\sim 50°$/s at the impulse. The controller suppresses these within $\sim 1$ second. The steady-state angular velocity noise is $\sim \pm 10°$/s, driven by the external disturbance.

**Cart Acceleration** (second-from-bottom left): Shows the control effort in acceleration space. The initial spike corresponds to the impulsive control response ($\sim 200$ m/s²), then settles to $\pm 20$ m/s² during noise rejection.

**Angular Accelerations** (second-from-bottom right): Link accelerations peak at $\sim 500°$/s² during the initial transient. $\theta_3$ (tip) shows the largest accelerations due to its low inertia and long moment arm.

**Energy** (bottom-left): Kinetic energy (orange) spikes at $t = 0$ from the impulse. Potential energy (cyan) is nearly constant near $V \approx (m_1 + m_2 + m_3) \cdot g \cdot L_{\text{total}} / 2$ since the pendulum stays near upright. Total energy (black dashed) fluctuates as the controller continuously injects and dissipates energy.

**Phase Portrait** (bottom-right): All three link trajectories ($\theta_i$ vs $\dot{\theta}_i$) spiral inward toward the origin $(0, 0)$, providing visual confirmation of **asymptotic stability**. The spiraling pattern indicates underdamped oscillatory convergence.

### Control Analysis

![Control Analysis](images/control_analysis.png)

**Force Comparison** (top-left): The control force (blue) peaks at $\sim -500$ N at $t = 0$ -- the LQR applies maximum effort in the $-x$ direction to counteract the $+x$ impulse and prevent the pendulum from toppling. Subsequent control is $\sim \pm 50$ N, tracking the disturbance (red) to maintain balance.

**Frequency Spectrum** (top-right): FFT of the control signal shows dominant energy below 5 Hz, matching the pendulum's natural frequencies. The disturbance spectrum rolls off at 3 Hz (the Butterworth cutoff). The controller bandwidth extends to $\sim 10$ Hz.

**Bode Plot -- Open Loop** (middle-left): The magnitude plot of $|L(j\omega)|$ shows the gain crossover frequency $\omega_{gc}$ where $|L| = 0$ dB. The slope at crossover indicates the stability margin quality. Phase margin (PM) and gain margin (GM) are annotated. The -3 dB bandwidth of the closed-loop system indicates the disturbance rejection bandwidth.

**Nyquist Diagram** (middle-right): The Nyquist contour of $L(j\omega)$ for $\omega > 0$ (solid blue) and its reflection for $\omega < 0$ (dashed). The critical point $(-1 + 0j)$ is marked with a red cross. Direction arrows show increasing $\omega$. The minimum distance from the contour to $(-1, 0)$ quantifies the robustness margin. The number of clockwise encirclements must equal the number of unstable open-loop poles ($n_u = 3$ for the triple inverted pendulum).

**Sensitivity $S(j\omega)$ and $T(j\omega)$** (second-from-bottom left): $|S(j\omega)|$ (blue) is the disturbance-to-output transfer function -- it should remain below 6 dB to avoid noise amplification. $|T(j\omega)|$ (red) is the complementary sensitivity. The peak values $M_s$ and $M_t$ are key robustness indicators. The crossover frequency where $|S| = |T|$ roughly separates the disturbance rejection band (low frequency) from the noise attenuation band (high frequency).

**Pole Map** (second-from-bottom right): Open-loop poles (red crosses) include 3 in the right half-plane (RHP), confirming the system is **open-loop unstable** -- inherent to the inverted pendulum. Closed-loop poles (blue circles) are all in the left half-plane (LHP), confirming the LQR successfully stabilizes the system. Each pole is annotated with its **damping ratio** $\zeta = -\text{Re}(p) / |p|$. The dominant pole (closest to the imaginary axis) determines the settling time.

**Bode Plot -- Closed Loop** (bottom-left): Transfer function from disturbance $d$ to cart position $x$. The -3 dB bandwidth indicates the frequency range over which disturbances effectively reach the output. Resonance peaks indicate frequencies where the disturbance is amplified.

**Step Response** (bottom-right): Response of the closed-loop system to a unit step disturbance force. Key metrics annotated:
- **Overshoot**: percentage by which the response exceeds the steady-state value
- **Settling time** $T_s$ (2% band): time for the response to remain within 2% of steady-state
- **Rise time** $T_r$: time from 10% to 90% of steady-state

### LQR Verification

![LQR Verification](images/lqr_verification.png)

**Lyapunov Function** $V(t) = \mathbf{z}^T P\,\mathbf{z}$ (top-left): The Lyapunov function is plotted on a logarithmic scale. Under ideal LQR (no disturbance), $V(t)$ would decrease monotonically. With noise, small increases occur but the overall trend is strongly decreasing. The percentage of timesteps where $\dot{V} < 0$ is displayed -- values above 95% confirm robust stability despite persistent disturbance.

**Riccati $P$ Eigenvalues** (top-right): All 8 eigenvalues of $P$ must be strictly positive for $P$ to be positive definite. This is a **necessary condition** for the CARE solution to be valid and for $V = \mathbf{z}^T P\,\mathbf{z}$ to be a proper Lyapunov function. Green bars indicate positive eigenvalues.

**LQR Cost Breakdown** (middle-left): The instantaneous state cost $\mathbf{z}^T Q\,\mathbf{z}$ (blue) and control cost $u^T\!Ru$ (red) on a log scale. The state cost dominates initially (large deviations from equilibrium), while the control cost spikes at $t = 0$ (peak force). Both decay exponentially, confirming the cost is being minimized.

**Cumulative Cost** $J(t) = \int_0^t (\mathbf{z}^T Q\,\mathbf{z} + u^T\!Ru)\,d\tau$ (middle-right): The running integral of the LQR objective. Convergence to a finite value confirms the infinite-horizon cost is bounded -- a fundamental requirement of LQR optimality. The final value represents the total "price" paid for stabilization.

**Return Difference** $|1 + L(j\omega)|$ (bottom-left): The Kalman inequality requires this to be $\geq 0$ dB (i.e., $\geq 1$ in linear scale) at all frequencies. This is the **signature guarantee** of SISO LQR, ensuring robustness margins of at least $\pm 6$ dB gain margin and $60°$ phase margin. The minimum value and its frequency are annotated.

**Nyquist Encirclement Verification** (bottom-right): The Nyquist contour is shown with the computed number of clockwise encirclements $N_{\text{CW}}$ of $(-1, 0)$. For closed-loop stability of a system with $n_u$ unstable open-loop poles, the Nyquist criterion demands $N_{\text{CW}} = n_u$. A **PASS/FAIL** indicator confirms whether this condition is met.

### Animation

![Animation](images/animation.gif)

Real-time animation of the cart-pendulum system. Each link is color-coded:
- **Red**: Link 1 ($L_1 = 0.402$ m, $m_1 = 1.323$ kg)
- **Green**: Link 2 ($L_2 = 0.332$ m, $m_2 = 1.389$ kg)
- **Blue**: Link 3 ($L_3 = 0.720$ m, $m_3 = 0.8655$ kg)

The faint red trace shows the trajectory of the tip (end of link 3). The cart (gray rectangle) oscillates about $x = 0$ while keeping all three links balanced upright against continuous random forcing.

---

## Outputs

All automatically saved to `images/` on each run:

| File | Content |
|------|---------|
| `dynamics_analysis.png` | 8 subplots: position, velocity, acceleration, energy, phase portrait |
| `control_analysis.png` | 8 subplots: Bode, Nyquist, sensitivity, poles, step response, spectrum |
| `lqr_verification.png` | 6 subplots: Lyapunov, Riccati, cost, Kalman inequality, Nyquist check |
| `animation.gif` | Cart-pendulum animation at 30 fps |

---

## Project Structure

```
Triple-Pendulum-LQR/
├── main.py                              # Entry point (physical parameters only)
│
├── pipeline/
│   ├── runner.py                        # Orchestrator: LQR → simulate → analyze → plot → save
│   ├── defaults.py                      # Default simulation parameters
│   └── save_outputs.py                  # PNG / GIF export
│
├── parameters/
│   ├── physical.py                      # Raw user inputs (masses, lengths, gravity)
│   ├── derived.py                       # Derived coefficients (α, β, γ)
│   ├── packing.py                       # Flat array serialization for Numba
│   ├── equilibrium.py                   # Upright equilibrium q* = (0, π, 0, 0)
│   └── config.py                        # SystemConfig facade
│
├── dynamics/                            # All @njit compiled
│   ├── trigonometry.py                  # Shared sin/cos computation
│   ├── mass_matrix/
│   │   ├── cart_link_coupling.py        # Cart-pendulum coupling m_xi
│   │   ├── pendulum_block.py           # 3×3 inertia sub-matrix M_ij
│   │   └── assembly.py                 # 4×4 symmetric M assembly
│   ├── coriolis/
│   │   └── christoffel.py              # Christoffel symbols via central differences
│   ├── gravity/
│   │   └── gravity_vector.py           # G(q) computation
│   └── forward_dynamics/
│       ├── tau_assembly.py              # Input mapping τ = (F, 0, 0, 0)
│       ├── solve_acceleration.py        # M⁻¹ · rhs via np.linalg.solve
│       └── forward_dynamics.py          # Full q̈ = M⁻¹(τ − Cq̇ − G)
│
├── control/
│   ├── linearization/
│   │   ├── jacobian_q.py               # ∂f/∂q via central differences
│   │   ├── jacobian_dq.py              # ∂f/∂q̇
│   │   ├── jacobian_u.py               # ∂f/∂u
│   │   ├── state_space.py              # Assemble A(8×8), B(8×1)
│   │   └── linearize.py                # Linearization facade
│   ├── cost_matrices/
│   │   ├── default_Q.py                # Q = diag(10, 100, 100, 100, 1, 10, 10, 10)
│   │   └── default_R.py                # R = 0.01
│   ├── riccati/
│   │   └── solve_care.py               # scipy CARE wrapper
│   ├── gain_computation/
│   │   └── compute_K.py                # K = R⁻¹BᵀP
│   ├── lqr.py                           # End-to-end LQR facade
│   └── closed_loop.py                   # A_cl, eigenvalues, stability check
│
├── simulation/
│   ├── integrator/
│   │   ├── state_derivative.py          # ż = (q̇, q̈) — @njit
│   │   └── rk4_step.py                 # Single RK4 step — @njit
│   ├── disturbance/
│   │   ├── white_noise.py              # Gaussian white noise generator
│   │   ├── bandpass_filter.py          # FFT-based 4th-order Butterworth
│   │   ├── normalize.py                # RMS amplitude scaling
│   │   └── generate_disturbance.py     # Disturbance generation pipeline
│   ├── initial_conditions/
│   │   └── impulse_response.py         # M⁻¹ · (impulse, 0, 0, 0)ᵀ
│   └── loop/
│       ├── control_law.py              # u = −Kz — @njit
│       └── time_loop.py                # Main simulation loop
│
├── analysis/
│   ├── state/                           # Absolute angles, joint positions, deviations
│   ├── energy/                          # Kinetic, potential, total energy
│   ├── frequency/                       # Open/closed-loop TF, S(jω), T(jω), margins, poles, step
│   ├── lqr_verification/               # Lyapunov, Kalman, Nyquist checks
│   └── summary/                         # Console output
│
├── visualization/
│   ├── common/                          # Shared colors and axis styling
│   ├── animation/                       # Cart-pendulum FuncAnimation
│   ├── dynamics_plots/                  # 4×2 dynamics grid
│   ├── control_plots/                   # 4×2 control grid
│   └── lqr_plots/                       # 3×2 LQR verification grid
│
├── images/                              # Auto-generated output plots
├── requirements.txt
├── LICENSE                              # MIT License
└── README.md
```

**60 source files** organized into 8 domain packages.

---

## References

1. Medrano-Cerda, G.A. (1997). "Robust stabilization of a triple inverted pendulum-cart." *International Journal of Control*, 68(4), 849–865.
2. Anderson, B.D.O. & Moore, J.B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice Hall.
3. Kalman, R.E. (1964). "When is a linear control system optimal?" *ASME Journal of Basic Engineering*, 86(1), 51–60.
4. Gluck, T., Eder, A. & Kugi, A. (2013). "Swing-up control of a triple pendulum on a cart with experimental validation." *Automatica*, 49(3), 801–808.
5. Tsachouridis, V.A. (1999). "Robust control of a triple inverted pendulum." *IEEE Conference on Decision and Control*.

## License

MIT -- see [LICENSE](LICENSE).
