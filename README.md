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
| `t_end` | 15.0 | Simulation duration in seconds |
| `dt` | 0.002 | Integration time step in seconds |
| `impulse` | 5.0 | Initial cart impulse in N·s |
| `dist_amplitude` | 15.0 | Disturbance RMS amplitude in N |
| `dist_bandwidth` | 3.0 | Disturbance cutoff frequency in Hz |

## Benchmark Parameters

The Medrano-Cerda system [1] has been used extensively in control research since 1997. Key characteristics:

| Parameter | Value | Unit |
|-----------|-------|------|
| Cart mass m_c | 2.4 | kg |
| Link 1 mass m₁ | 1.323 | kg |
| Link 2 mass m₂ | 1.389 | kg |
| Link 3 mass m₃ | 0.8655 | kg |
| Link 1 length L₁ | 0.402 | m |
| Link 2 length L₂ | 0.332 | m |
| Link 3 length L₃ | 0.720 | m |
| Gravity g | 9.81 | m/s² |

Notable feature: L₃ is the longest link (0.72 m) but lightest (0.87 kg), making the tip highly susceptible to disturbances and the system challenging to stabilize.

---

## Theory

### 1. System Description

A cart of mass m_c translates along a horizontal rail. Three uniform rigid links form a serial chain attached to the cart by revolute joints. The generalized coordinates are:

$$\mathbf{q} = \begin{pmatrix} x \\ \theta_1 \\ \theta_2 \\ \theta_3 \end{pmatrix}$$

where:
- x is the cart horizontal position
- θ₁ is the absolute angle of link 1 from the downward vertical
- θ₂ is the relative angle of link 2 w.r.t. link 1
- θ₃ is the relative angle of link 3 w.r.t. link 2

The only control input is a single horizontal force F on the cart. This makes the system **underactuated** (4 DOF, 1 input).

### 2. Lagrangian Dynamics

#### 2.1 Kinematics

Absolute angles accumulate from the relative coordinates:

$$\phi_k = \sum_{i=1}^{k} \theta_i \qquad (k = 1, 2, 3)$$

Center-of-mass position of the k-th link:

$$x_{cm,k} = x + \sum_{i=1}^{k-1} L_i \sin \phi_i + \frac{L_k}{2} \sin \phi_k$$

$$y_{cm,k} = -\sum_{i=1}^{k-1} L_i \cos \phi_i - \frac{L_k}{2} \cos \phi_k$$

#### 2.2 Energy

Kinetic energy (with moment of inertia I_k = m_k L_k² / 12 for uniform rods):

$$T = \frac{1}{2} m_c \dot{x}^2 + \sum_{k=1}^{3} \left[ \frac{1}{2} m_k \left( \dot{x}_{cm,k}^2 + \dot{y}_{cm,k}^2 \right) + \frac{1}{2} I_k \dot{\phi}_k^2 \right]$$

Gravitational potential energy:

$$V = \sum_{k=1}^{3} m_k \, g \, y_{cm,k}$$

#### 2.3 Mass Matrix

The transformation from absolute to relative angular velocities:

$$\dot{\boldsymbol{\phi}} = J \dot{\boldsymbol{\theta}}$$

$$J = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{pmatrix}$$

The resulting 4×4 symmetric mass matrix:

$$M(\mathbf{q}) = \begin{pmatrix} M_t & m_{x1} & m_{x2} & m_{x3} \\ m_{x1} & M_{11} & M_{12} & M_{13} \\ m_{x2} & M_{12} & M_{22} & M_{23} \\ m_{x3} & M_{13} & M_{23} & M_{33} \end{pmatrix}$$

where:
- M_t = m_c + m₁ + m₂ + m₃ is the total system mass
- m_{x1}, m_{x2}, m_{x3} are cart-link coupling terms (functions of cos φ₁, cos φ₂, cos φ₃)
- M_{ij} form the 3×3 pendulum inertia block (functions of cos θ₂, cos θ₃, cos(θ₂ + θ₃))

Built from three families of derived constants:

$$\alpha_i = \left( \frac{m_i}{3} + \sum_{j>i} m_j \right) L_i^2$$

$$\beta_{ij} = \left( \frac{m_j}{2} + \sum_{k>j} m_k \right) L_i L_j$$

$$\gamma_i = \left( \frac{m_i}{2} + \sum_{j>i} m_j \right) L_i$$

#### 2.4 Coriolis and Gravity

The Coriolis/centrifugal vector is computed via Christoffel symbols:

$$h_i = \sum_{j,k} \Gamma_{ijk} \, \dot{q}_j \, \dot{q}_k$$

$$\Gamma_{ijk} = \frac{1}{2} \left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right)$$

The gravity vector (with gravity constants g_i = γ_i · g):

$$G(\mathbf{q}) = \begin{pmatrix} 0 \\ g_1 \sin \phi_1 + g_2 \sin \phi_2 + g_3 \sin \phi_3 \\ g_2 \sin \phi_2 + g_3 \sin \phi_3 \\ g_3 \sin \phi_3 \end{pmatrix}$$

#### 2.5 Equations of Motion

$$M(\mathbf{q}) \, \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}}) \, \dot{\mathbf{q}} + G(\mathbf{q}) = \begin{pmatrix} F \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

### 3. LQR Control Design

#### 3.1 Linearization

The system is linearized around the upright equilibrium **q**\* = (0, π, 0, 0)ᵀ, **q̇**\* = **0** using numerical central differences to obtain the Jacobians A_q, A_q̇, B_u.

The 8-dimensional state-space form with **z** = (δ**q**, δ**q̇**)ᵀ:

$$\dot{\mathbf{z}} = A \, \mathbf{z} + B \, u$$

$$A = \begin{pmatrix} \mathbf{0} & I \\ A_q & A_{\dot{q}} \end{pmatrix}, \qquad B = \begin{pmatrix} \mathbf{0} \\ B_u \end{pmatrix}$$

where the top-left and bottom-right blocks are 4×4, I is the 4×4 identity, and B is 8×1.

#### 3.2 LQR Optimal Gain

The LQR minimizes the infinite-horizon quadratic cost by solving the Continuous Algebraic Riccati Equation (CARE):

$$J = \int_0^\infty \left( \mathbf{z}^T Q \, \mathbf{z} + u^T R \, u \right) dt$$

$$A^T P + P A - P B R^{-1} B^T P + Q = 0$$

$$K = R^{-1} B^T P, \qquad u = -K \mathbf{z}$$

#### 3.3 Default Cost Weights

| State | Q weight | Rationale |
|-------|----------|-----------|
| x (cart position) | 10 | Moderate regulation |
| θ₁ (base link) | 100 | Primary stabilization target |
| θ₂ (middle link) | 100 | Secondary stabilization |
| θ₃ (tip link) | 100 | Tertiary stabilization |
| ẋ (cart velocity) | 1 | Low penalty |
| θ̇₁, θ̇₂, θ̇₃ | 10 | Moderate damping |
| R (control weight) | 0.01 | Permits aggressive actuation |

### 4. LQR Verification

#### 4.1 Lyapunov Stability

The CARE solution P ≻ 0 defines a Lyapunov candidate V(**z**) = **z**ᵀP**z**:

$$\dot{V} = -\mathbf{z}^T (Q + K^T R K) \mathbf{z} < 0 \quad \forall \, \mathbf{z} \neq \mathbf{0}$$

This guarantees **global asymptotic stability** of the linearized closed-loop system.

#### 4.2 Kalman Frequency-Domain Inequality

For SISO LQR with loop transfer function L(s) = K(sI − A)⁻¹B, the **return difference condition** holds:

$$|1 + L(j\omega)| \geq 1 \quad \forall \, \omega$$

This guarantees:
- **Gain margin**: (−6 dB, +∞), i.e., stable under gain variations from 0.5× to ∞×
- **Phase margin**: ≥ 60°

#### 4.3 Nyquist Criterion

The open-loop plant has n_u unstable poles (right half-plane eigenvalues of A). Closed-loop stability requires the Nyquist contour of L(jω) to make exactly n_u clockwise encirclements of (−1 + 0j):

$$N_{\text{CW}} = n_u$$

### 5. Disturbance Model

Band-limited white noise, generated by FFT-filtering Gaussian noise through a 4th-order Butterworth lowpass:

$$d(t) = \mathcal{F}^{-1} \left[ W(j\omega) \cdot H(j\omega) \right]$$

$$H(j\omega) = \frac{1}{1 + (\omega / \omega_c)^4}$$

where W(jω) is the white noise spectrum and ω_c = 2πf_c is the cutoff angular frequency.

### 6. Numerical Integration

Classical 4th-order Runge-Kutta with fixed step Δt:

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6} \left( \mathbf{k}_1 + 2 \mathbf{k}_2 + 2 \mathbf{k}_3 + \mathbf{k}_4 \right)$$

All dynamics functions (M, C, G, forward dynamics) are compiled to native machine code via Numba `@njit(cache=True)`.

---

## Analysis Results

All results below use the Medrano-Cerda parameters with initial impulse = 5 N·s and band-limited noise (RMS = 15 N, f_c = 3 Hz).

### Dynamics Analysis

![Dynamics Analysis](images/dynamics_analysis.png)

**Cart Position** (top-left): The initial impulse displaces the cart to approximately −0.38 m. The LQR controller drives it back toward x = 0, but continuous noise causes persistent small oscillations around the origin. The cart never diverges, confirming closed-loop stability.

**Cart Velocity** (top-right): Peak velocity of ~1.8 m/s occurs at t = 0 due to the impulse. The controller rapidly damps this, with subsequent velocity fluctuations of ±0.3 m/s driven by the noise.

**Angle Deviations** (middle-left): Maximum deviation is ~5° for θ₁ (base link), ~3° for θ₂, and ~1.2° for θ₃. The base link shows the largest deviation because it directly couples to the cart motion. The decreasing deviation from base to tip reflects the LQR's heavy weighting (Q = 100) on all angles.

**Angular Velocities** (middle-right): Peak angular rates reach ~50°/s at the impulse. The controller suppresses these within ~1 second. The steady-state angular velocity noise is ~±10°/s, driven by the external disturbance.

**Cart Acceleration** (second-from-bottom left): Shows the control effort in acceleration space. The initial spike corresponds to the impulsive control response (~200 m/s²), then settles to ±20 m/s² during noise rejection.

**Angular Accelerations** (second-from-bottom right): Link accelerations peak at ~500°/s² during the initial transient. θ₃ (tip) shows the largest accelerations due to its low inertia and long moment arm.

**Energy** (bottom-left): Kinetic energy (orange) spikes at t = 0 from the impulse. Potential energy (cyan) is nearly constant since the pendulum stays near upright. Total energy (dashed black) fluctuates as the controller continuously injects and dissipates energy.

**Phase Portrait** (bottom-right): All three link trajectories (θᵢ vs θ̇ᵢ) spiral inward toward the origin (0, 0), providing visual confirmation of **asymptotic stability**. The spiraling pattern indicates underdamped oscillatory convergence.

### Control Analysis

![Control Analysis](images/control_analysis.png)

**Force Comparison** (top-left): The control force peaks at ~−500 N at t = 0 — the LQR applies maximum effort in the −x direction to counteract the +x impulse and prevent toppling. Subsequent control is ~±50 N, tracking the disturbance to maintain balance.

**Frequency Spectrum** (top-right): FFT of the control signal shows dominant energy below 5 Hz, matching the pendulum's natural frequencies. The disturbance spectrum rolls off at 3 Hz (the Butterworth cutoff). The controller bandwidth extends to ~10 Hz.

**Bode Plot — Open Loop** (middle-left): Magnitude plot of |L(jω)| with the gain crossover frequency ω_gc annotated. Phase margin (PM) and gain margin (GM) are displayed. The slope at crossover (in dB/decade) indicates stability margin quality. The −3 dB closed-loop bandwidth is also shown.

**Nyquist Diagram** (middle-right): The Nyquist contour of L(jω) for ω > 0 (solid blue) and its reflection for ω < 0 (dashed). The critical point (−1 + 0j) is marked. Direction arrows show increasing ω. The minimum distance from the contour to (−1, 0) quantifies the robustness margin. The number of clockwise encirclements must equal n_u = 3 (the number of unstable open-loop poles).

**Sensitivity S(jω) and T(jω)** (second-from-bottom left): |S(jω)| (blue) should remain below 6 dB to avoid noise amplification. |T(jω)| (red) is the complementary sensitivity. Peak values M_s and M_t are annotated. The crossover frequency where |S| = |T| separates the disturbance rejection band from the noise attenuation band.

**Pole Map** (second-from-bottom right): Open-loop poles (red crosses) include 3 in the right half-plane, confirming the system is **open-loop unstable**. Closed-loop poles (blue circles) are all in the left half-plane, confirming stabilization. Each pole is annotated with its damping ratio ζ = −Re(p) / |p|.

**Bode Plot — Closed Loop** (bottom-left): Transfer function from disturbance d to cart position x. The −3 dB bandwidth and any resonance peaks are annotated.

**Step Response** (bottom-right): Response to a unit step disturbance force. Annotated metrics: overshoot (%), settling time T_s (2% band), rise time T_r (10%→90%).

### LQR Verification

![LQR Verification](images/lqr_verification.png)

**Lyapunov Function V(t) = zᵀPz** (top-left): Plotted on log scale. Under ideal LQR (no disturbance), V(t) decreases monotonically. With noise, small increases occur but the overall trend is strongly decreasing. The percentage of timesteps where V̇ < 0 is displayed — values above 95% confirm robust stability.

**Riccati P Eigenvalues** (top-right): All 8 eigenvalues of P must be strictly positive for P to be positive definite. This is a **necessary condition** for the CARE solution to be valid and for V = zᵀPz to be a proper Lyapunov function. Green bars = positive.

**LQR Cost Breakdown** (middle-left): Instantaneous state cost zᵀQz (blue) and control cost uᵀRu (red) on log scale. State cost dominates initially; control cost spikes at t = 0. Both decay exponentially.

**Cumulative Cost J(t)** (middle-right): Running integral of the LQR objective. Convergence to a finite value confirms the infinite-horizon cost is bounded — a fundamental requirement of LQR optimality.

**Return Difference |1 + L(jω)|** (bottom-left): The Kalman inequality requires this to be ≥ 0 dB at all frequencies. This is the **signature guarantee** of SISO LQR. The minimum value and frequency are annotated. Violation would indicate a non-optimal or incorrect LQR design.

**Nyquist Encirclement Verification** (bottom-right): Nyquist contour with computed number of clockwise encirclements N_CW of (−1, 0). For stability: N_CW must equal the number of unstable open-loop poles n_u. A PASS/FAIL indicator is displayed.

### Animation

![Animation](images/animation.gif)

Real-time animation of the cart-pendulum system. Each link is color-coded:
- **Red**: Link 1 (L₁ = 0.402 m, m₁ = 1.323 kg)
- **Green**: Link 2 (L₂ = 0.332 m, m₂ = 1.389 kg)
- **Blue**: Link 3 (L₃ = 0.720 m, m₃ = 0.8655 kg)

The faint red trace shows the trajectory of the tip (end of link 3). The gray cart oscillates about x = 0 while keeping all three links balanced upright against continuous random forcing.

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
│   │   ├── cart_link_coupling.py        # Cart-pendulum coupling terms
│   │   ├── pendulum_block.py           # 3×3 inertia sub-matrix
│   │   └── assembly.py                 # 4×4 symmetric M assembly
│   ├── coriolis/
│   │   └── christoffel.py              # Christoffel symbols via central differences
│   ├── gravity/
│   │   └── gravity_vector.py           # G(q) computation
│   └── forward_dynamics/
│       ├── tau_assembly.py              # Input mapping τ = (F, 0, 0, 0)
│       ├── solve_acceleration.py        # M⁻¹ · rhs
│       └── forward_dynamics.py          # Full q̈ = M⁻¹(τ − Cq̇ − G)
│
├── control/
│   ├── linearization/
│   │   ├── jacobian_q.py               # ∂f/∂q
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
│   │   └── impulse_response.py         # Solve M·Δq̇ = (impulse, 0, 0, 0)ᵀ
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

MIT — see [LICENSE](LICENSE).
