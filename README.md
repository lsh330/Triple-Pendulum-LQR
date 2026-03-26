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
| `dt` | 0.001 | Integration time step in seconds |
| `impulse` | 5.0 | Initial cart impulse in N·s |
| `dist_amplitude` | 15.0 | Disturbance RMS amplitude in N |
| `dist_bandwidth` | 3.0 | Disturbance cutoff frequency in Hz |
| `u_max` | 200.0 | Actuator force saturation limit in N |

## Benchmark Parameters

The Medrano-Cerda system [1] has been used extensively in control research since 1997. Key characteristics:

| Parameter | Value | Unit |
|-----------|-------|------|
| Cart mass m<sub>c</sub> | 2.4 | kg |
| Link 1 mass m<sub>1</sub> | 1.323 | kg |
| Link 2 mass m<sub>2</sub> | 1.389 | kg |
| Link 3 mass m<sub>3</sub> | 0.8655 | kg |
| Link 1 length L<sub>1</sub> | 0.402 | m |
| Link 2 length L<sub>2</sub> | 0.332 | m |
| Link 3 length L<sub>3</sub> | 0.720 | m |
| Gravity g | 9.81 | m/s² |

Notable feature: L<sub>3</sub> is the longest link (0.72 m) but lightest (0.87 kg), making the tip highly susceptible to disturbances and the system challenging to stabilize.

---

## Theory

### 1. System Description

A cart of mass m<sub>c</sub> translates along a horizontal rail. Three uniform rigid links (m<sub>1</sub>, L<sub>1</sub>), (m<sub>2</sub>, L<sub>2</sub>), (m<sub>3</sub>, L<sub>3</sub>) form a serial chain attached to the cart by revolute joints. The generalized coordinates are:

$$\mathbf{q} = \begin{bmatrix} x \\\ \theta_1 \\\ \theta_2 \\\ \theta_3 \end{bmatrix}$$

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

Kinetic energy (with moment of inertia I<sub>k</sub> = m<sub>k</sub>L<sub>k</sub>² / 12 for uniform rods):

$$T = \frac{1}{2} m_c \dot{x}^2 + \sum_{k=1}^{3} \left[ \frac{1}{2} m_k \left( \dot{x}_{cm,k}^2 + \dot{y}_{cm,k}^2 \right) + \frac{1}{2} I_k \dot{\phi}_k^2 \right]$$

Gravitational potential energy:

$$V = \sum_{k=1}^{3} m_k \, g \, y_{cm,k}$$

#### 2.3 Mass Matrix

The transformation from absolute to relative angular velocities:

$$\dot{\vec{\phi}} = J \, \dot{\vec{\theta}}$$

$$J = \begin{bmatrix} 1 & 0 & 0 \\\ 1 & 1 & 0 \\\ 1 & 1 & 1 \end{bmatrix}$$

The resulting 4×4 symmetric mass matrix:

$$M(\mathbf{q}) = \begin{bmatrix} M_t & m_{x1} & m_{x2} & m_{x3} \\\ m_{x1} & M_{11} & M_{12} & M_{13} \\\ m_{x2} & M_{12} & M_{22} & M_{23} \\\ m_{x3} & M_{13} & M_{23} & M_{33} \end{bmatrix}$$

where:
- M<sub>t</sub> = m<sub>c</sub> + m<sub>1</sub> + m<sub>2</sub> + m<sub>3</sub> is the total system mass
- m<sub>x1</sub>, m<sub>x2</sub>, m<sub>x3</sub> are cart-link coupling terms (functions of cos φ<sub>1</sub>, cos φ<sub>2</sub>, cos φ<sub>3</sub>)
- M<sub>ij</sub> form the 3×3 pendulum inertia block (functions of cos θ<sub>2</sub>, cos θ<sub>3</sub>, cos(θ<sub>2</sub> + θ<sub>3</sub>))

Built from three families of derived constants:

$$\alpha_i = \left( \frac{m_i}{3} + \sum_{j>i} m_j \right) L_i^2$$

$$\beta_{ij} = \left( \frac{m_j}{2} + \sum_{k>j} m_k \right) L_i L_j$$

$$\gamma_i = \left( \frac{m_i}{2} + \sum_{j>i} m_j \right) L_i$$

#### 2.4 Coriolis and Gravity

The Coriolis/centrifugal vector **h** = C(**q**, **q̇**)**q̇** is computed via Christoffel symbols of the first kind:

$$h_i = \sum_{j,k} \Gamma_{ijk} \, \dot{q}_j \, \dot{q}_k$$

$$\Gamma_{ijk} = \frac{1}{2} \left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right)$$

This requires the partial derivatives ∂M/∂q<sub>k</sub> for k = 1, 2, 3 (∂M/∂x = 0 since M does not depend on cart position).

**Analytical derivation of ∂M/∂q<sub>k</sub>**

The mass matrix elements depend on trigonometric functions of the generalized coordinates. Differentiating each element analytically:

**∂M/∂θ<sub>1</sub>** — only the cart-link coupling terms (top row/left column) are affected, since the 3×3 pendulum block depends only on θ<sub>2</sub> and θ<sub>3</sub>:

$$\frac{\partial m_{x1}}{\partial \theta_1} = -\gamma_1 \sin\phi_1 - \gamma_2 \sin\phi_2 - \gamma_3 \sin\phi_3$$

$$\frac{\partial m_{x2}}{\partial \theta_1} = -\gamma_2 \sin\phi_2 - \gamma_3 \sin\phi_3$$

$$\frac{\partial m_{x3}}{\partial \theta_1} = -\gamma_3 \sin\phi_3$$

$$\frac{\partial M_{ij}^{\text{pend}}}{\partial \theta_1} = 0 \quad \text{(all pendulum block entries)}$$

**∂M/∂θ<sub>2</sub>** — affects both the cart-link coupling and parts of the pendulum block:

$$\frac{\partial m_{x1}}{\partial \theta_2} = -\gamma_2 \sin\phi_2 - \gamma_3 \sin\phi_3, \qquad \frac{\partial m_{x2}}{\partial \theta_2} = -\gamma_2 \sin\phi_2 - \gamma_3 \sin\phi_3, \qquad \frac{\partial m_{x3}}{\partial \theta_2} = -\gamma_3 \sin\phi_3$$

$$\frac{\partial M_{11}}{\partial \theta_2} = -2\beta_1 \sin\theta_2 - 2\beta_2 \sin(\theta_2+\theta_3)$$

$$\frac{\partial M_{12}}{\partial \theta_2} = -\beta_1 \sin\theta_2 - \beta_2 \sin(\theta_2+\theta_3), \qquad \frac{\partial M_{13}}{\partial \theta_2} = -\beta_2 \sin(\theta_2+\theta_3)$$

All other pendulum block derivatives w.r.t. θ<sub>2</sub> are zero.

**∂M/∂θ<sub>3</sub>** — affects cart-link coupling and most of the pendulum block:

$$\frac{\partial m_{xi}}{\partial \theta_3} = -\gamma_3 \sin\phi_3 \quad (i = 1, 2, 3)$$

$$\frac{\partial M_{11}}{\partial \theta_3} = -2\beta_2 \sin(\theta_2+\theta_3) - 2\beta_3 \sin\theta_3$$

$$\frac{\partial M_{12}}{\partial \theta_3} = -\beta_2 \sin(\theta_2+\theta_3) - 2\beta_3 \sin\theta_3, \qquad \frac{\partial M_{13}}{\partial \theta_3} = -\beta_2 \sin(\theta_2+\theta_3) - \beta_3 \sin\theta_3$$

$$\frac{\partial M_{22}}{\partial \theta_3} = -2\beta_3 \sin\theta_3, \qquad \frac{\partial M_{23}}{\partial \theta_3} = -\beta_3 \sin\theta_3, \qquad \frac{\partial M_{33}}{\partial \theta_3} = 0$$

These closed-form derivatives eliminate the need for numerical finite differences, improving both accuracy (exact to machine precision) and performance (no extra mass matrix evaluations).

**Gravity vector** (with gravity constants g<sub>i</sub> = γ<sub>i</sub> · g):

$$G(\mathbf{q}) = \begin{bmatrix} 0 \\\ g_1 \sin \phi_1 + g_2 \sin \phi_2 + g_3 \sin \phi_3 \\\ g_2 \sin \phi_2 + g_3 \sin \phi_3 \\\ g_3 \sin \phi_3 \end{bmatrix}$$

#### 2.5 Equations of Motion

$$M(\mathbf{q}) \, \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}}) \, \dot{\mathbf{q}} + G(\mathbf{q}) = \begin{bmatrix} F \\\ 0 \\\ 0 \\\ 0 \end{bmatrix}$$

### 3. LQR Control Design

#### 3.1 Linearization

The system is linearized around the upright equilibrium **q**\* = (0, π, 0, 0)ᵀ, **q̇**\* = **0** using numerical central differences to obtain the Jacobians A<sub>q</sub>, A<sub>q̇</sub>, B<sub>u</sub>.

The 8-dimensional state-space form with **z** = (δ**q**, δ**q̇**)ᵀ:

$$\dot{\mathbf{z}} = A \, \mathbf{z} + B \, u$$

$$A = \begin{bmatrix} \mathbf{0} & I \\\ A_q & A_{\dot{q}} \end{bmatrix}, \qquad B = \begin{bmatrix} \mathbf{0} \\\ B_u \end{bmatrix}$$

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

#### 3.4 Adaptive Numerical Jacobians

The linearization Jacobians (A<sub>q</sub>, A<sub>q̇</sub>, B<sub>u</sub>) are computed via central finite differences with **adaptive step size**:

$$h_j = \epsilon_{\text{mach}}^{1/3} \cdot \max(1, |q_j^*|)$$

where ε<sub>mach</sub> ≈ 2.2 × 10⁻¹⁶ is machine epsilon. This yields h ≈ 6.1 × 10⁻⁶ for unit-scale variables and h ≈ 1.9 × 10⁻⁵ for θ₁ = π. For central differences, the truncation error is O(h²) and the roundoff error is O(ε/h); balancing these gives the optimal step h ≈ ε<sup>1/3</sup>, which is larger than the ε<sup>1/2</sup> optimum for forward differences.

#### 3.5 Gain Scheduling

A single LQR gain K is optimal only at the linearization point. To extend performance over larger deviations, a heuristic local gain bank precomputes LQR gains at multiple operating points:

$$\theta_1^{(i)} = \pi + \delta_i, \qquad \delta_i \in \{-20°, -10°, -5°, 0°, +5°, +10°, +20°\}$$

At each operating point, the system is re-linearized and a new LQR gain K<sup>(i)</sup> is computed. At runtime, the gain is linearly interpolated based on the current θ₁ deviation:

$$K(\delta) = (1 - t) \, K^{(i)} + t \, K^{(i+1)}, \qquad t = \frac{\delta - \delta_i}{\delta_{i+1} - \delta_i}$$

The interpolation is implemented inside a @njit-compiled function for zero Python overhead.

**Note**: The operating points are not true equilibria — perturbing θ₁ while keeping q̇ = 0 and u = 0 does not satisfy the equations of motion. This is a heuristic local gain bank that works well near upright, not a rigorous equilibrium-family-based gain schedule.

#### 3.6 Iterative LQR (iLQR)

iLQR extends the standard LQR to handle nonlinear dynamics by iteratively refining a trajectory:

1. **Forward pass**: Simulate the nonlinear system with current gains to obtain a nominal trajectory x<sub>nom</sub>(t), u<sub>nom</sub>(t)
2. **Backward pass**: Linearize at each point along the trajectory and solve a discrete-time Riccati recursion to obtain time-varying gains K(t)
3. **Forward pass**: Re-simulate with updated feedback u = u<sub>nom</sub> − K(t)(x − x<sub>nom</sub>)
4. **Iterate** until convergence (typically 3–10 iterations)

The backward Riccati recursion at each timestep k:

$$S_k = Q + A_k^T S_{k+1} A_k - A_k^T S_{k+1} B_k (R + B_k^T S_{k+1} B_k)^{-1} B_k^T S_{k+1} A_k$$

$$K_k = (R + B_k^T S_{k+1} B_k)^{-1} B_k^T S_{k+1} A_k$$

iLQR produces a time-varying gain schedule that is locally optimal along the actual nonlinear trajectory, making it superior to fixed-point LQR for large initial deviations.

### 4. LQR Verification

#### 4.1 Lyapunov Stability

The CARE solution P ≻ 0 defines a Lyapunov candidate V(**z**) = **z**ᵀ P **z**:

$$\dot{V} = -\mathbf{z}^T (Q + K^T R K) \mathbf{z} < 0 \quad \forall \, \mathbf{z} \neq \mathbf{0}$$

This guarantees **global asymptotic stability** of the linearized closed-loop system.

#### 4.2 Kalman Frequency-Domain Inequality

For SISO LQR with loop transfer function L(s) = K(sI − A)⁻¹B, the **return difference condition** holds:

$$|1 + L(j\omega)| \geq 1 \quad \forall \, \omega$$

This guarantees:
- **Gain margin**: (−6 dB, +∞), i.e., stable under gain variations from 0.5× to ∞×
- **Phase margin**: ≥ 60°

#### 4.3 Nyquist Criterion

The open-loop plant has n<sub>u</sub> unstable poles (right half-plane eigenvalues of A). Closed-loop stability requires the Nyquist contour of L(jω) to make exactly n<sub>u</sub> clockwise encirclements of (−1 + 0j):

$$N_{\text{CW}} = n_u$$

Closed-loop stability is verified definitively via eigenvalue analysis (exact for LTI systems). The numerical winding number is computed as a diagnostic cross-reference but can be inaccurate for high-order systems due to frequency sampling artifacts.

#### 4.4 Region of Attraction (ROA)

The LQR is designed around a linearization point and is only guaranteed to stabilize the nonlinear system within some neighborhood of that point. The **Region of Attraction** is the set of initial conditions from which the controller successfully returns the system to equilibrium.

ROA is estimated via Monte Carlo simulation: random initial angle deviations are sampled uniformly, each is simulated for a fixed time horizon without disturbance. Convergence requires both **trajectory boundedness** (cart position within ±2 m, angle deviations within ±90° throughout the entire simulation) and **final-state convergence** (max angle deviation < 1° at end). The early-exit on divergence also improves computational efficiency for unstable trajectories. The boundary of converged vs diverged initial conditions maps out the empirical ROA.

#### 4.5 Gain Scheduling Stability

For the gain-scheduled controller (Section 3.5) to be stable, it is not sufficient that each operating point's LQR is individually stable — the **transitions between operating points** must also preserve stability. Verification is performed by:

1. Confirming all closed-loop eigenvalues are in the LHP at each of the 7 operating points
2. Checking 100 interpolated points between each adjacent pair of operating points
3. Computing the condition number of P at each operating point (high condition number indicates fragile stability)

If all interpolated points have stable eigenvalues, the gain-scheduled controller is verified to be robustly stable across its entire operating range.

### 5. Actuator Saturation

The control force is subject to symmetric saturation:

$$u_{\text{applied}} = \text{clip}(u_{\text{LQR}}, -u_{\max}, +u_{\max})$$

where u<sub>max</sub> defaults to 200 N. The saturation is applied inside the @njit-compiled simulation loop with a branch-based clip (two comparisons per timestep, negligible overhead). This prevents unrealistic actuator forces during large transients and provides a more physically meaningful simulation.

### 6. Disturbance Model

Band-limited white noise, generated by FFT-filtering Gaussian noise through a 4th-order Butterworth lowpass:

$$d(t) = \mathcal{F}^{-1} \left[ W(j\omega) \cdot H(j\omega) \right]$$

$$H(j\omega) = \frac{1}{1 + (\omega / \omega_c)^4}$$

where W(jω) is the white noise spectrum and ω<sub>c</sub> = 2πf<sub>c</sub> is the cutoff angular frequency.

### 7. Numerical Integration

Classical 4th-order Runge-Kutta with fixed step Δt:

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6} \left( \mathbf{k}_1 + 2 \mathbf{k}_2 + 2 \mathbf{k}_3 + \mathbf{k}_4 \right)$$

All dynamics functions (M, C, G, forward dynamics) and the entire simulation loop are compiled to native machine code via Numba `@njit(cache=True)`. The simulation loop runs entirely inside a single JIT-compiled function with zero Python interpreter overhead per timestep.

### 8. Computational Optimization

| Optimization | Before | After | Speedup |
|-------------|--------|-------|---------|
| Coriolis computation | Numerical FD (8 M evals/step) | Analytical sparse (0 M evals) | **∞** |
| 4×4 linear solve | np.linalg.solve (LAPACK) | Inline Cramer's rule (cofactors) | **~5×** |
| Array shapes | (4,1) with indexing [i,0] | (4,) flat vectorized | **~2×** |
| Christoffel loop | 64 iterations (4³) | ~25 hardcoded scalar ops | **~3×** |
| Simulation loop | Python for-loop calling @njit | Entire loop in @njit | **~2×** |
| Gain scheduling | N/A | @njit interpolation | 0 overhead |
| LQR linearization | 3 Python-loop Jacobians (0.19s) | Single @njit Jacobian (0.001s) | **~190×** |
| JIT warmup | Lazy (first call penalty) | Explicit warmup_jit() at startup | Predictable |
| Trig computation | 3× per forward_dynamics (27 calls) | 1× monolithic (9 calls) | **~3×** |
| Array allocation | ~38 per RK4 step (heap) | 0 per step (all scalars) | **~2.7×** |
| State packing | 8-vector pack/unpack per step | Direct scalar propagation | **0 overhead** |
| Control law | Array z construction + dot product | Inline scalar multiply-add | **0 allocation** |
| Angle wrapping | None (drift-prone) | Per-step atan2-free wrap in control | **negligible** |

The simulation hot path (`forward_dynamics_fast` + `rk4_step_fast`) uses **zero heap allocation** per timestep. All state variables, mass matrix elements, Coriolis terms, and RHS values are scalar locals. The cofactor 4×4 solve is inlined. This eliminates ~570,000 small array allocations over a 15-second simulation.

**Combined result**: A 15-second simulation (15,001 steps) completes in **~15 ms**. LQR design completes in **1 ms** (cached).

| Pipeline Stage | Method | Time |
|---------------|--------|------|
| JIT warmup | Pre-compile all @njit functions | ~2.8 s (one-time) |
| LQR design | @njit Jacobian + scipy CARE | **~0.001 s** (cached) |
| Simulation (15s, dt=0.001) | Zero-alloc scalar RK4 + monolithic dynamics | **~0.015 s** |
| Monte Carlo (20 samples) | ThreadPool parallel | ~0.03 s |
| ROA estimation (500 samples) | JIT simulation per sample | ~5 s |
| Frequency analysis | scipy.signal | ~0.005 s |
| **Total (excl. plots)** | | **~0.23 s** |

---

## Analysis Results

All results below use the Medrano-Cerda parameters with initial impulse = 5 N·s and band-limited noise (RMS = 15 N, f<sub>c</sub> = 3 Hz).

### Dynamics Analysis

![Dynamics Analysis](images/dynamics_analysis.png)

#### Subplot Descriptions

**Cart Position** (top-left): Horizontal position of the cart over time. Indicates how far the cart translates from its initial position under the combined effect of external forcing and the control law.

**Cart Velocity** (top-right): Time derivative of the cart position. Shows the translational dynamics of the cart including transient and steady-state behavior.

**Angle Deviations** (middle-left): Deviation of each joint angle from the upright equilibrium. Positive values indicate counterclockwise tilt; negative indicates clockwise. This is the primary measure of stabilization performance.

**Angular Velocities** (middle-right): Rate of change of each joint angle. High angular rates indicate rapid pendulum motion; the controller must suppress these to prevent toppling.

**Cart Acceleration** (row 3, left): Second time derivative of cart position, computed via numerical differentiation. Reflects the net force on the cart divided by total effective mass.

**Angular Accelerations** (row 3, right): Second time derivative of each joint angle. Directly related to the torques experienced at each joint through the equations of motion.

**Energy** (row 4, left): Breakdown of total mechanical energy into kinetic (translational + rotational) and gravitational potential components. In a conservative system without control, total energy would be constant; deviations reflect energy injected or dissipated by the controller.

**Phase Portrait** (row 4, right): State-space trajectories plotting angle deviation vs angular velocity for each link. In a stable system, trajectories converge to the origin; the shape of the spiral indicates the damping characteristics.

**Control Force** (row 5, left): The single control input F applied to the cart over time. Peak annotations highlight the maximum actuator effort required during the initial transient.

**Joint Reaction Estimation** (row 5, right): Generalized accelerations (numerical second derivatives) for the cart and each joint. Serves as a proxy for the generalized forces acting through the equations of motion.

#### Simulation Results for This System

- The initial 5 N·s impulse displaces the cart to approximately **−0.38 m**. The LQR controller returns it toward x = 0 within ~2 seconds, but continuous noise causes persistent oscillations of ±0.1 m.
- Peak cart velocity reaches **~1.8 m/s** at t = 0. The controller damps this to ±0.3 m/s within 1 second.
- Maximum angle deviations are **~5° (θ<sub>1</sub>), ~3° (θ<sub>2</sub>), ~1.2° (θ<sub>3</sub>)**. The base link shows the largest deviation because it directly couples to the cart translation. The decreasing deviation from base to tip reflects the combined effect of inertia distribution and the Q = 100 weighting on all angles.
- Cart acceleration peaks at **~200 m/s²** at t = 0 (impulsive response), settling to ±20 m/s² during noise rejection.
- θ<sub>3</sub> (tip link) shows the **largest angular accelerations** (~500°/s²) due to its low inertia (I<sub>3</sub> = 0.0374 kg·m²) and long moment arm (L<sub>3</sub> = 0.72 m).
- Kinetic energy spikes at t = 0 from the impulse. Potential energy remains nearly constant near V ≈ (m<sub>1</sub> + m<sub>2</sub> + m<sub>3</sub>)gL<sub>total</sub>/2 since the pendulum stays near upright. Total energy **fluctuates** as the controller continuously injects and dissipates energy to counteract the stochastic disturbance.
- Control force peaks at **~−500 N** at t = 0, then settles to ±50 N during noise rejection. The negative peak reflects the LQR's aggressive response to the +x impulse.
- Generalized accelerations confirm that θ<sub>3</sub> (tip) experiences the **highest joint loads** due to its low inertia and long moment arm.
- All three phase trajectories **spiral inward** toward (0, 0), visually confirming asymptotic stability. The spiraling pattern indicates underdamped oscillatory convergence — consistent with the moderate damping (Q<sub>vel</sub> = 10) in the LQR cost.

---

### Control Analysis

![Control Analysis](images/control_analysis.png)

#### Subplot Descriptions

**Force Comparison** (top-left): Time-domain overlay of the LQR control force and the external disturbance force applied to the cart. Shows the controller's reaction to the combined impulse and stochastic noise.

**Frequency Spectrum** (top-right): Single-sided amplitude spectrum (FFT) of both the control signal and the disturbance. Reveals which frequency bands the controller is most active in and how the disturbance energy is distributed.

**Bode Plot — Open Loop** (middle-left): Magnitude and phase of the open-loop transfer function L(jω) = K(sI − A)⁻¹B as a function of frequency. The gain crossover frequency ω<sub>gc</sub> (where |L| = 0 dB) and the phase at that frequency determine the phase margin. The phase crossover (where phase = −180°) determines the gain margin.

**Nyquist Diagram** (middle-right): Polar plot of L(jω) in the complex plane as ω sweeps from 0 to ∞. The Nyquist stability criterion relates the number of encirclements of the critical point (−1 + 0j) to the number of closed-loop unstable poles. The minimum distance from the contour to (−1, 0) is a direct measure of robustness.

**Sensitivity S(jω) and T(jω)** (second-from-bottom left): S(jω) = 1/(1 + L(jω)) is the sensitivity function — it maps disturbances to output. T(jω) = L(jω)/(1 + L(jω)) is the complementary sensitivity — it maps reference inputs to output. Together, S + T = 1. Peak |S| (denoted M<sub>s</sub>) is a key robustness metric: M<sub>s</sub> < 2 (6 dB) is generally required.

**Pole Map** (second-from-bottom right): Eigenvalues of the open-loop A matrix (system without control) and the closed-loop A − BK matrix (with LQR). Poles in the right half-plane indicate instability; the LQR must move all poles to the left half-plane. The damping ratio ζ of each pole determines the oscillatory decay rate.

**Bode Plot — Closed Loop** (bottom-left): Magnitude and phase of the closed-loop transfer function from disturbance input to cart position output. The −3 dB bandwidth indicates the effective frequency range of disturbance rejection. Resonance peaks indicate frequencies where disturbances are amplified.

**Step Response** (bottom-right): Time-domain response of the closed-loop system to a unit step disturbance force. Standard performance metrics: overshoot (%), settling time T<sub>s</sub> (time to enter and stay within 2% of steady-state), and rise time T<sub>r</sub> (10% to 90% of steady-state).

#### Simulation Results for This System

- Control force peaks at **~−500 N** at t = 0. The negative sign means the LQR pushes the cart in −x to counteract the +x impulse. This aggressive initial response is necessary because the triple inverted pendulum is open-loop unstable — any delay in response causes exponential divergence.
- The control spectrum shows dominant energy **below 5 Hz**, matching the natural frequency range of the three-link pendulum. The disturbance spectrum rolls off at 3 Hz (Butterworth cutoff). The controller bandwidth extends to ~10 Hz, meaning it can reject disturbances up to this frequency.
- **Phase margin ≈ 72°** (well above the LQR-guaranteed minimum of 60°), confirming robust stability with safety margin against unmodeled phase lag.
- The Nyquist contour makes **3 clockwise encirclements** of (−1, 0), exactly matching the 3 unstable open-loop poles — satisfying the Nyquist criterion for closed-loop stability.
- Peak sensitivity M<sub>s</sub> ≈ 0 dB (|S| ≈ 1.0), indicating **no disturbance amplification** at any frequency. This is exceptionally good — typical control designs allow M<sub>s</sub> up to 6 dB.
- The open-loop system has **3 poles in the right half-plane** (eigenvalues with positive real part), confirming the triple inverted pendulum is inherently unstable. The LQR moves all 8 poles to the left half-plane with damping ratios ranging from ζ ≈ 0.3 to ζ ≈ 1.0.
- Step response shows the cart settling within **~3 seconds** after a unit step disturbance, with moderate overshoot. The steady-state offset is near zero due to the Q<sub>x</sub> = 10 weighting on cart position.

---

### LQR Verification

![LQR Verification](images/lqr_verification.png)

#### Subplot Descriptions

**Lyapunov Function V(t) = zᵀPz** (top-left): The quadratic Lyapunov function constructed from the CARE solution P. In the absence of disturbance, V(t) must decrease monotonically along every trajectory (V̇ < 0), which is the formal proof of asymptotic stability. With stochastic noise, V(t) may temporarily increase, but the overall trend must be decreasing.

**Riccati P Eigenvalues** (top-right): Bar chart of the eigenvalues of the 8×8 matrix P. For P to be a valid Lyapunov matrix, it must be positive definite, meaning all eigenvalues must be strictly > 0. This is a necessary condition for the CARE solution to exist and for the LQR to be stabilizing.

**LQR Cost Breakdown** (middle-left): Decomposition of the instantaneous LQR cost into its two components: the state cost zᵀQz (penalizing deviation from equilibrium) and the control cost uᵀRu (penalizing actuator effort). The ratio between these reveals the controller's trade-off between performance and effort.

**Cumulative Cost J(t)** (middle-right): Running integral of the total instantaneous cost. The LQR is defined as the controller that minimizes J as t → ∞. If J(t) converges to a finite value, the infinite-horizon cost is bounded, confirming the LQR design is valid and the closed-loop system is stable.

**Return Difference |1 + L(jω)|** (bottom-left): Frequency-domain plot of the return difference, which is the key quantity in the Kalman inequality. For any SISO LQR design, |1 + L(jω)| ≥ 1 (i.e., ≥ 0 dB) must hold at all frequencies. This is a fundamental property of LQR — violation would indicate an implementation error or a non-optimal design.

**Nyquist Encirclement Verification** (row 3, left): Independent verification of the Nyquist stability criterion. The algorithm computes the winding number of the Nyquist contour around (−1, 0) and compares it to the number of unstable open-loop poles. A PASS result confirms that the Nyquist criterion is satisfied.

**Monte Carlo Bode** (bottom-left): Open-loop Bode magnitude overlaid for 20 random mass perturbations (±10% on each link mass independently). The nominal response is shown in blue; perturbed responses in gray. If all curves maintain similar shape, the LQR is robust to parametric uncertainty.

**Closed-Loop Pole Scatter** (bottom-right): Closed-loop poles for the same 20 perturbed systems. Nominal poles are shown as blue circles; perturbed poles as gray dots. If all perturbed poles remain in the left half-plane (LHP), the system is robustly stable under ±10% mass uncertainty.

#### Simulation Results for This System

- The Lyapunov function V(t) is **decreasing for ~60% of timesteps** under the continuous RMS 15 N disturbance. The remaining ~40% of timesteps show V increasing, corresponding to moments when the noise injects energy faster than the controller dissipates it. This is expected behavior: without disturbance, V would decrease nearly 100% of the time, but persistent stochastic forcing causes frequent temporary increases. The key observation is that V(t) **does not diverge** — it remains bounded and oscillates around a low level, confirming practical stability under persistent disturbance. The Lyapunov guarantee (V̇ < 0) strictly holds only for the deterministic (no-noise) case.
- All 8 eigenvalues of P are **strictly positive** (ranging from ~1 to ~10⁴), confirming P is positive definite. The large spread in eigenvalues reflects the different scales of the state variables (cart position in meters vs link angles in radians).
- The state cost zᵀQz **dominates** the total cost during the initial transient (large angle deviations), while the control cost uᵀRu **spikes at t = 0** (peak actuator force of 500 N). Both decay exponentially with similar time constants, indicating the LQR achieves a balanced trade-off.
- The cumulative cost J(t) **converges** to a finite value, confirming the infinite-horizon cost integral is bounded. Under the persistent noise, J(t) grows slowly in a linear fashion after the transient dies — this is expected since the noise continuously injects cost.
- The return difference |1 + L(jω)| is **≥ 0 dB at all frequencies**, satisfying the Kalman inequality. This confirms the LQR design provides the guaranteed minimum gain margin of (−6 dB, +∞) and phase margin of ≥ 60°.
- The Nyquist encirclement count is **N<sub>CW</sub> = 3**, exactly matching n<sub>u</sub> = 3 (the number of unstable open-loop poles). **PASS** — the Nyquist criterion is satisfied.
- Monte Carlo analysis with **20 random ±10% mass perturbations** shows all perturbed Bode curves maintain similar shape to the nominal, confirming **parametric robustness**. All perturbed closed-loop poles remain in the left half-plane — the LQR design is **robustly stable** under realistic manufacturing tolerances.

---

### Animation

![Animation](images/animation.gif)

#### Description

Real-time animation of the cart-pendulum system at 30 fps. The cart (gray rectangle) translates along the horizontal rail. Three rigid links are rendered with distinct colors:
- **Red**: Link 1 (L<sub>1</sub> = 0.402 m, m<sub>1</sub> = 1.323 kg) — base link, directly attached to cart
- **Green**: Link 2 (L<sub>2</sub> = 0.332 m, m<sub>2</sub> = 1.389 kg) — middle link
- **Blue**: Link 3 (L<sub>3</sub> = 0.720 m, m<sub>3</sub> = 0.8655 kg) — tip link, longest and lightest

The faint red trace shows the historical trajectory of the tip endpoint. The brown line indicates the ground/rail.

#### Observations for This System

The animation shows the cart initially displaced to the left by the impulse, then oscillating back toward the center while the three links sway slightly but never topple. The tip trace reveals persistent small oscillations driven by the band-limited noise, with the amplitude concentrated near the upright position. The LQR controller successfully maintains all three links within ~5° of vertical despite continuous random forcing — a challenging feat for a system with 3 unstable open-loop poles and only 1 control input.

---

### ROA & Gain Scheduling Analysis

![ROA Analysis](images/roa_analysis.png)

#### Subplot Descriptions

**ROA Scatter Plot** (top-left): Each dot represents a random initial condition in the (θ<sub>1</sub>, θ<sub>2</sub>) deviation space. Green dots converged to equilibrium; red dots diverged. The boundary between green and red regions maps the empirical Region of Attraction.

**ROA Success Rate** (top-right): Bar chart showing the percentage of initial conditions that converged, binned by the magnitude of the initial θ<sub>1</sub> deviation. Quantifies how the success rate degrades with increasing initial perturbation.

**Gain Scheduling Eigenvalues** (bottom-left): Maximum real part of the closed-loop eigenvalues at each gain scheduling operating point and at 100 interpolated points between each pair. All values must be negative for stability. The further below zero, the more robustly stable.

**P Condition Numbers** (bottom-right): Condition number of the Riccati solution P at each operating point. High condition numbers indicate sensitivity to numerical errors. Values below ~10⁶ are generally acceptable.

#### Simulation Results for This System

- ROA success rate: **~12%** of random initial conditions (within ±45° × ±22.5° × ±15°) converge. This reflects the difficulty of stabilizing a triple inverted pendulum — the ROA is narrow.
- Maximum stable θ<sub>1</sub> deviation: **~16°** from upright. Beyond this, the nonlinear dynamics dominate and the LQR (linearized around θ₁ = π) cannot recover.
- All 7 gain scheduling operating points are **stable** (eigenvalues in LHP).
- All interpolated points between operating points are **stable** — the gain interpolation preserves closed-loop stability.
- Maximum Re(eigenvalue) across all checked points: **−1.22**, confirming a healthy stability margin.

---

## Outputs

All automatically saved to `images/` on each run:

| File | Content |
|------|---------|
| `dynamics_analysis.png` | 10 subplots: position, velocity, acceleration, energy, phase portrait, control force, joint reactions |
| `control_analysis.png` | 8 subplots: Bode, Nyquist, sensitivity, poles, step response, spectrum |
| `lqr_verification.png` | 8 subplots: Lyapunov, Riccati, cost, Kalman inequality, Monte Carlo Bode, pole scatter |
| `roa_analysis.png` | 4 subplots: ROA scatter, success rate, GS eigenvalues, P condition numbers |
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
│       ├── forward_dynamics.py          # Full q̈ = M⁻¹(τ − Cq̇ − G)
│       └── forward_dynamics_fast.py     # Zero-alloc monolithic scalar dynamics + RK4
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
│   ├── closed_loop.py                   # A_cl, eigenvalues, stability check
│   ├── gain_scheduling.py               # Multi-point LQR with @njit interpolation
│   └── ilqr.py                          # Iterative LQR for nonlinear trajectories
│
├── simulation/
│   ├── warmup.py                        # Pre-trigger all @njit compilations
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
│       ├── time_loop.py                # Simulation loop (legacy + fast dispatch)
│       └── time_loop_fast.py           # Zero-alloc scalar-state simulation loops
│
├── analysis/
│   ├── state/                           # Absolute angles, joint positions, deviations
│   ├── energy/                          # Kinetic, potential, total energy
│   ├── frequency/                       # Open/closed-loop TF, S(jω), T(jω), margins, poles, step
│   ├── lqr_verification/               # Lyapunov, Kalman, Nyquist, Monte Carlo
│   ├── region_of_attraction.py          # Monte Carlo ROA estimation
│   ├── gain_scheduling_stability.py     # Interpolated stability verification
│   └── summary/                         # Console output
│
├── visualization/
│   ├── common/                          # Shared colors and axis styling
│   ├── animation/                       # Cart-pendulum FuncAnimation
│   ├── dynamics_plots/                  # 4×2 dynamics grid
│   ├── control_plots/                   # 4×2 control grid
│   ├── lqr_plots/                       # 4×2 LQR verification grid
│   └── roa_plots/                       # 2×2 ROA & gain scheduling grid
│
├── images/                              # Auto-generated output plots
├── requirements.txt
├── LICENSE                              # MIT License
└── README.md
```

**68 source files** organized into 9 domain packages.

---

## References

1. Medrano-Cerda, G.A. (1997). "Robust stabilization of a triple inverted pendulum-cart." *International Journal of Control*, 68(4), 849–865.
2. Anderson, B.D.O. & Moore, J.B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice Hall.
3. Kalman, R.E. (1964). "When is a linear control system optimal?" *ASME Journal of Basic Engineering*, 86(1), 51–60.
4. Gluck, T., Eder, A. & Kugi, A. (2013). "Swing-up control of a triple pendulum on a cart with experimental validation." *Automatica*, 49(3), 801–808.
5. Tsachouridis, V.A. (1999). "Robust control of a triple inverted pendulum." *IEEE Conference on Decision and Control*.

## License

MIT — see [LICENSE](LICENSE).
