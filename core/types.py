"""Type aliases for the triple pendulum simulation."""
from numpy.typing import NDArray
import numpy as np

# 상태 벡터 (dim 4: x, theta1, theta2, theta3)
ConfigVec = NDArray[np.float64]
# 속도 벡터 (dim 4: dx, dtheta1, dtheta2, dtheta3)
VelocityVec = NDArray[np.float64]
# 전체 상태 (dim 8: q, dq)
StateVec = NDArray[np.float64]
# 파라미터 벡터 (dim 13)
ParamVec = NDArray[np.float64]
# LQR 게인 (dim 8)
GainVec = NDArray[np.float64]
