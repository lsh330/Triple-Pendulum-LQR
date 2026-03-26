from parameters.config import SystemConfig
from pipeline.runner import run

# Medrano-Cerda system (University of Salford, 1997)
cfg = SystemConfig(
    mc=2.4,
    m1=1.323, m2=1.389, m3=0.8655,
    L1=0.402, L2=0.332, L3=0.720,
)

run(cfg)
