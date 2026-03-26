# Cart + Triple Inverted Pendulum Simulator

LQR-optimal stabilization of a triple inverted pendulum on a cart, subject to band-limited stochastic disturbances. This simulator provides comprehensive dynamics, control, and LQR verification analysis with publication-quality visualizations.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run simulation
python main.py
```

Four windows will appear:
1. **Animation** -- Real-time pendulum motion with link color coding
2. **Dynamics Analysis** -- 8 subplots: position, velocity, acceleration, energy, phase portrait
3. **Control Analysis** -- 8 subplots: Bode, Nyquist, sensitivity, poles, step response
4. **LQR Verification** -- 6 subplots: Lyapunov decay, Riccati validation, Kalman inequality

All plots are automatically saved to `images/`.

## Installation

### Prerequisites
- Python >= 3.9
- VS Code (recommended IDE)

### Dependencies

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Numerical computation |
| scipy | >= 1.10 | LQR (Riccati solver), frequency analysis |
| numba | >= 0.57 | JIT compilation for simulation speed |
| matplotlib | >= 3.6 | Visualization |
| pillow | >= 9.0 | GIF animation export |

## Configuration

Edit `main.py` to change physical parameters:

```python
from parameters.config import SystemConfig
from pipeline.runner import run

cfg = SystemConfig(
    mc=2.4,                    # cart mass [kg]
    m1=1.323, m2=1.389, m3=0.8655,  # link masses [kg]
    L1=0.402, L2=0.332, L3=0.720,   # link lengths [m]
)

run(cfg)                       # default: 15s, impulse=5 N*s, noise RMS=15 N
run(cfg, t_end=20, impulse=10) # override simulation parameters
```

Default parameters are from the Medrano-Cerda system (University of Salford, 1997), a widely-used benchmark in robust control literature.

## Outputs

All saved to `images/`:

| File | Content |
|------|---------|
| `dynamics_analysis.png` | Cart position/velocity/acceleration, angles, energy, phase portrait |
| `control_analysis.png` | Force comparison, Bode, Nyquist, sensitivity, poles, step response |
| `lqr_verification.png` | Lyapunov function, Riccati eigenvalues, cost, Kalman inequality |
| `animation.gif` | Animated pendulum motion |

---

## Theory

### 1. System Description

A cart moves along a horizontal rail. Three rigid uniform links are connected in series by revolute joints, forming a triple pendulum attached to the cart. The generalized coordinates are:

$$\mathbf{q} = \begin{pmatrix} x \\ \theta_1 \\ \theta_2 \\ \theta_3 \end{pmatrix}$$

where x is the cart position, theta\_1 is the absolute angle of link 1 from the downward vertical, theta\_2 is the relative angle of link 2 w.r.t. link 1, and theta\_3 is the relative angle of link 3 w.r.t. link 2. The only control input is a horizontal force F applied to the cart.

### 2. Lagrangian Dynamics

#### 2.1 Kinematics

The absolute angles and center-of-mass positions of each link are:

$$\phi_k = \sum_{i=1}^{k} \theta_i \qquad (k = 1, 2, 3)$$

$$x_{cm,k} = x + \sum_{i=1}^{k-1} L_i \sin\phi_i + \frac{L_k}{2}\sin\phi_k$$

$$y_{cm,k} = -\sum_{i=1}^{k-1} L_i \cos\phi_i - \frac{L_k}{2}\cos\phi_k$$

#### 2.2 Energy

$$T = \frac{1}{2}m_c\dot{x}^2 + \sum_{k=1}^{3}\left[\frac{1}{2}m_k\!\left(\dot{x}_{cm,k}^2 + \dot{y}_{cm,k}^2\right) + \frac{1}{2}\cdot\frac{m_k L_k^2}{12}\cdot\dot{\phi}_k^2\right]$$

$$V = \sum_{k=1}^{3} m_k \, g \, y_{cm,k}$$

#### 2.3 Mass Matrix

The coordinate transformation from absolute to relative angular velocities is:

$$\dot{\phi} = J\,\dot{\theta}, \qquad J = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{pmatrix}$$

After this transformation, the 4x4 mass matrix in the full coordinates q = (x, theta\_1, theta\_2, theta\_3) is:

$$M(\mathbf{q}) = \begin{pmatrix} M_t    & m_{x1} & m_{x2} & m_{x3} \\ m_{x1} & M_{11} & M_{12} & M_{13} \\ m_{x2} & M_{12} & M_{22} & M_{23} \\ m_{x3} & M_{13} & M_{23} & M_{33} \end{pmatrix}$$

The top-left element is the total system mass. The top row/left column contains cart-link coupling terms. The bottom-right 3x3 block is the pendulum inertia sub-matrix. These are built from three families of constants:

$$\alpha_i = \left(\frac{m_i}{3} + \sum_{j>i}m_j\right)L_i^2, \qquad \beta_{ij} = \left(\frac{m_j}{2} + \sum_{k>j}m_k\right)L_i L_j, \qquad \gamma_{xi} = \left(\frac{m_i}{2} + \sum_{j>i}m_j\right)L_i$$

#### 2.4 Coriolis and Gravity

The Coriolis/centrifugal vector is computed via Christoffel symbols of the first kind:

$$h_i = \sum_{j,k} \Gamma_{ijk}\,\dot{q}_j\dot{q}_k$$

$$\Gamma_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)$$

The gravity vector, with gravity constants defined as g\_i = gamma\_{xi} times g:

$$G(\mathbf{q}) = \begin{pmatrix} 0 \\ g_1\sin\phi_1 + g_2\sin\phi_2 + g_3\sin\phi_3 \\ g_2\sin\phi_2 + g_3\sin\phi_3 \\ g_3\sin\phi_3 \end{pmatrix}$$

#### 2.5 Equations of Motion

$$M(\mathbf{q})\,\ddot{\mathbf{q}} + C(\mathbf{q},\dot{\mathbf{q}})\,\dot{\mathbf{q}} + G(\mathbf{q}) = \begin{pmatrix} F \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

This is an **underactuated** system: 4 degrees of freedom controlled by a single input F.

### 3. LQR Control Design

#### 3.1 Linearization

The nonlinear system is linearized around the upright equilibrium:

$$\mathbf{q}^* = \begin{pmatrix} 0 \\ \pi \\ 0 \\ 0 \end{pmatrix}, \qquad \dot{\mathbf{q}}^* = \mathbf{0}$$

The linearized acceleration is:

$$\delta\ddot{\mathbf{q}} = A_q\,\delta\mathbf{q} + A_{\dot{q}}\,\delta\dot{\mathbf{q}} + B_u\,\delta u$$

where the Jacobians are computed numerically via central differences. The 8-dimensional state-space form is:

$$\dot{\mathbf{z}} = A\mathbf{z} + B\,u$$

$$A = \begin{pmatrix} \mathbf{0}_{4\times4} & I_{4\times4} \\ A_q & A_{\dot{q}} \end{pmatrix}, \qquad B = \begin{pmatrix} \mathbf{0}_{4\times1} \\ B_u \end{pmatrix}$$

#### 3.2 LQR Optimal Gain

The LQR minimizes the infinite-horizon quadratic cost, solved via the Continuous Algebraic Riccati Equation (CARE):

$$J = \int_0^\infty \!\left(\mathbf{z}^T Q \, \mathbf{z} + u^T R \, u\right) dt$$

$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$

$$K = R^{-1}B^T P, \qquad u = -K\mathbf{z}$$

#### 3.3 Default Cost Weights

| State | Weight | Rationale |
|-------|--------|-----------|
| x | 10 | Moderate cart position regulation |
| theta\_1 | 100 | Base link -- primary stabilization |
| theta\_2 | 100 | Middle link |
| theta\_3 | 100 | Tip link |
| dx/dt | 1 | Low velocity penalty |
| dtheta/dt | 10 | Moderate angular rate damping |
| R | 0.01 | Allows aggressive control force |

### 4. LQR Verification

#### 4.1 Lyapunov Stability

The Riccati solution P defines a Lyapunov function. Its time derivative under LQR is strictly negative, guaranteeing asymptotic stability:

$$V(\mathbf{z}) = \mathbf{z}^T P \, \mathbf{z}$$

$$\dot{V} = -\mathbf{z}^T(Q + K^T R K)\mathbf{z} < 0 \quad \forall\; \mathbf{z} \neq 0$$

#### 4.2 Kalman Frequency-Domain Inequality

The return difference of the SISO LQR loop transfer function must satisfy:

$$|1 + L(j\omega)| \geq 1 \quad \forall\; \omega$$

$$L(s) = K(sI - A)^{-1}B$$

This guarantees gain margin of at least (-6 dB, +inf) and phase margin of at least 60 degrees.

#### 4.3 Nyquist Criterion

For an open-loop system with n\_u unstable poles, closed-loop stability requires:

$$N_{\text{CW}} = n_u$$

where N\_CW is the number of clockwise encirclements of (-1, 0) by the Nyquist contour.

### 5. Disturbance Model

The external disturbance is band-limited white noise, generated by filtering Gaussian white noise through a 4th-order Butterworth lowpass:

$$d(t) = \mathcal{F}^{-1}\!\left[W(j\omega) \cdot H(j\omega)\right]$$

$$H(j\omega) = \frac{1}{1 + (\omega / \omega_c)^4}$$

where W is the white noise spectrum and omega\_c is the cutoff frequency.

### 6. Numerical Integration

The 4th-order Runge-Kutta method with fixed time step:

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}\!\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

All dynamics computations are JIT-compiled via Numba.

---

## Analysis Results (Medrano-Cerda System)

The following results are from the default configuration with impulse = 5 N*s and band-limited noise (RMS = 15 N, bandwidth = 3 Hz).

### Dynamics

![Dynamics Analysis](images/dynamics_analysis.png)

- **Cart displacement**: Peaks at ~0.38 m after initial impulse, then oscillates around origin under noise
- **Angle deviations**: Maximum ~5 deg (theta1), demonstrating effective stabilization of the inherently unstable equilibrium
- **Energy**: Total energy fluctuates as the controller injects/removes energy to counteract disturbances
- **Phase portrait**: Trajectories spiral toward the origin, confirming asymptotic stability

### Control

![Control Analysis](images/control_analysis.png)

- **Control force**: Peak ~500 N at t=0 (impulse response), then ~50 N RMS during noise rejection
- **Bode plot**: Shows the open-loop gain crossover slope and bandwidth
- **Nyquist diagram**: Properly encircles (-1,0) the required number of times for stability
- **Sensitivity**: Peak |S| < 6 dB indicates good disturbance rejection without amplification
- **Pole map**: All closed-loop poles in the left half-plane with annotated damping ratios

### LQR Verification

![LQR Verification](images/lqr_verification.png)

- **Lyapunov V(t)**: Monotonically decreasing (>95%), proving stability in the Lyapunov sense
- **Riccati P eigenvalues**: All strictly positive, confirming P is positive definite
- **LQR cost J(t)**: Converges to a finite value, confirming the infinite-horizon cost is bounded
- **Return difference**: |1 + L(jw)| >= 0 dB at all frequencies, satisfying the Kalman inequality
- **Nyquist encirclements**: Number of CW encirclements equals the number of unstable OL poles

### Animation

![Animation](images/animation.gif)

---

## Project Structure

```
Triple_pendulum_simulation/
|-- main.py                                  # Entry point (config only)
|-- requirements.txt
|-- README.md
|-- images/                                  # Auto-generated outputs
|
|-- pipeline/                                # Orchestration
|   |-- runner.py                            # Main pipeline
|   |-- defaults.py                          # Default simulation parameters
|   |-- save_outputs.py                      # PNG/GIF export
|
|-- parameters/                              # Physical system definition
|   |-- physical.py                          # Raw inputs: masses, lengths, gravity
|   |-- derived.py                           # Derived coefficients
|   |-- packing.py                           # Flat array for JIT
|   |-- equilibrium.py                       # Upright equilibrium point
|   |-- config.py                            # SystemConfig facade
|
|-- dynamics/                                # Lagrangian mechanics (all @njit)
|   |-- trigonometry.py                      # Shared sin/cos
|   |-- mass_matrix/
|   |   |-- cart_link_coupling.py            # Cart-pendulum coupling
|   |   |-- pendulum_block.py               # 3x3 inertia sub-matrix
|   |   |-- assembly.py                     # 4x4 assembly
|   |-- coriolis/
|   |   |-- christoffel.py                  # Christoffel symbols
|   |-- gravity/
|   |   |-- gravity_vector.py               # Gravity vector
|   |-- forward_dynamics/
|       |-- tau_assembly.py                  # Input mapping
|       |-- solve_acceleration.py            # Linear solve
|       |-- forward_dynamics.py              # M, C, G -> ddq
|
|-- control/                                 # LQR design
|   |-- linearization/
|   |   |-- jacobian_q.py, jacobian_dq.py, jacobian_u.py
|   |   |-- state_space.py, linearize.py
|   |-- cost_matrices/
|   |   |-- default_Q.py, default_R.py
|   |-- riccati/
|   |   |-- solve_care.py
|   |-- gain_computation/
|   |   |-- compute_K.py
|   |-- lqr.py, closed_loop.py
|
|-- simulation/                              # Time-domain simulation
|   |-- integrator/
|   |   |-- state_derivative.py (@njit)
|   |   |-- rk4_step.py (@njit)
|   |-- disturbance/
|   |   |-- white_noise.py, bandpass_filter.py, normalize.py, generate_disturbance.py
|   |-- initial_conditions/
|   |   |-- impulse_response.py
|   |-- loop/
|       |-- control_law.py (@njit), time_loop.py
|
|-- analysis/                                # Post-simulation
|   |-- state/, energy/, frequency/, lqr_verification/, summary/
|
|-- visualization/                           # Matplotlib
    |-- common/, animation/, dynamics_plots/, control_plots/, lqr_plots/
```

**60 source files** across 8 packages.

## References

1. Medrano-Cerda, G.A. (1997). "Robust stabilization of a triple inverted pendulum-cart." *Int. J. Control*, 68(4).
2. Anderson, B.D.O. & Moore, J.B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice Hall.
3. Kalman, R.E. (1964). "When is a linear control system optimal?" *ASME J. Basic Engineering*, 86(1).
4. Gluck, T., Eder, A. & Kugi, A. (2013). "Swing-up control of a triple pendulum on a cart." *Automatica*, 49(3).

## License

MIT
