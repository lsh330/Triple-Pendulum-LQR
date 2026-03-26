"""Cart + Triple Inverted Pendulum: LQR optimal stabilization.

Usage:
    python main.py                          # Default Medrano-Cerda
    python main.py --impulse 10 --t-end 20  # Custom parameters
    python main.py --config config.yaml     # YAML configuration
    python main.py --use-ilqr               # Enable iLQR
    python main.py --gain-scheduler 3d      # 3D trilinear gain scheduling
    python main.py --adaptive-q             # Inertia-scaled Q matrix
    python main.py --help
"""

import argparse
import sys

from parameters.config import SystemConfig
from pipeline.runner import run


def _build_parser():
    p = argparse.ArgumentParser(
        description="Triple Inverted Pendulum LQR Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # System parameters
    p.add_argument("--mc", type=float, default=2.4, help="Cart mass [kg]")
    p.add_argument("--m1", type=float, default=1.323, help="Link 1 mass [kg]")
    p.add_argument("--m2", type=float, default=1.389, help="Link 2 mass [kg]")
    p.add_argument("--m3", type=float, default=0.8655, help="Link 3 mass [kg]")
    p.add_argument("--L1", type=float, default=0.402, help="Link 1 length [m]")
    p.add_argument("--L2", type=float, default=0.332, help="Link 2 length [m]")
    p.add_argument("--L3", type=float, default=0.720, help="Link 3 length [m]")
    p.add_argument("--g", type=float, default=9.81, help="Gravity [m/s^2]")

    # Simulation parameters
    p.add_argument("--t-end", type=float, default=15.0, help="Duration [s]")
    p.add_argument("--dt", type=float, default=0.001, help="Time step [s]")
    p.add_argument("--impulse", type=float, default=5.0, help="Initial impulse [N*s]")
    p.add_argument("--dist-amplitude", type=float, default=15.0, help="Disturbance RMS [N]")
    p.add_argument("--dist-bandwidth", type=float, default=3.0, help="Disturbance bandwidth [Hz]")
    p.add_argument("--u-max", type=float, default=200.0, help="Actuator saturation [N]")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Control features
    p.add_argument("--use-ilqr", action="store_true", help="Enable iLQR trajectory optimization")
    p.add_argument("--ilqr-horizon", type=int, default=500, help="iLQR planning horizon")
    p.add_argument("--ilqr-iterations", type=int, default=10, help="iLQR iterations")
    p.add_argument("--gain-scheduler", choices=["1d", "3d"], default="1d",
                   help="Gain scheduler: 1d (cubic Hermite on theta1) or 3d (trilinear on theta1,2,3)")
    p.add_argument("--adaptive-q", action="store_true",
                   help="Use inertia-scaled Q matrix (Bryson's rule) instead of fixed default")

    # Config file
    p.add_argument("--config", type=str, default=None, help="YAML config file path")

    # Output
    p.add_argument("--no-display", action="store_true", help="Skip matplotlib display")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="Log level")

    return p


def _load_yaml_config(path):
    """Load and validate configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml required for --config. Install: pip install pyyaml")
        sys.exit(1)

    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in {path}: {e}")
        sys.exit(1)

    if cfg is None:
        return {}

    valid_sections = {"system", "simulation", "features"}
    for key in cfg:
        if key not in valid_sections:
            print(f"WARNING: Unknown config section '{key}' (expected: {valid_sections})")

    sys_params = cfg.get("system", {})
    for key in ["mc", "m1", "m2", "m3", "L1", "L2", "L3", "g"]:
        if key in sys_params:
            val = sys_params[key]
            if not isinstance(val, (int, float)) or val <= 0:
                print(f"ERROR: system.{key} must be a positive number, got {val}")
                sys.exit(1)

    return cfg


def main(argv=None):
    """Entry point for CLI and programmatic use."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    import logging
    logging.getLogger("triple_pendulum").setLevel(getattr(logging, args.log_level))

    # Validate simulation parameters
    if args.dt <= 0:
        print("ERROR: --dt must be positive"); sys.exit(1)
    if args.t_end <= 0:
        print("ERROR: --t-end must be positive"); sys.exit(1)
    if args.u_max < 0:
        print("ERROR: --u-max must be non-negative"); sys.exit(1)
    if args.t_end < args.dt:
        print(f"WARNING: --t-end ({args.t_end}) < --dt ({args.dt}), simulation will have only 1 step")

    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        sys_params = yaml_cfg.get("system", {})
        sim_params = yaml_cfg.get("simulation", {})
        feat_params = yaml_cfg.get("features", {})

        for key in ["mc", "m1", "m2", "m3", "L1", "L2", "L3", "g"]:
            if key in sys_params and f"--{key}" not in sys.argv:
                setattr(args, key, sys_params[key])

        yaml_to_cli = {"t_end": "--t-end", "dt": "--dt", "impulse": "--impulse",
                       "dist_amplitude": "--dist-amplitude", "dist_bandwidth": "--dist-bandwidth",
                       "u_max": "--u-max", "seed": "--seed"}
        yaml_to_attr = {"t_end": "t_end", "dt": "dt", "impulse": "impulse",
                        "dist_amplitude": "dist_amplitude", "dist_bandwidth": "dist_bandwidth",
                        "u_max": "u_max", "seed": "seed"}
        for ykey, cli_flag in yaml_to_cli.items():
            if ykey in sim_params and cli_flag not in sys.argv:
                setattr(args, yaml_to_attr[ykey], sim_params[ykey])

        if feat_params.get("use_ilqr"):
            args.use_ilqr = True
        if "ilqr_horizon" in feat_params:
            args.ilqr_horizon = feat_params["ilqr_horizon"]
        if "ilqr_iterations" in feat_params:
            args.ilqr_iterations = feat_params["ilqr_iterations"]
        if "gain_scheduler" in feat_params and "--gain-scheduler" not in sys.argv:
            args.gain_scheduler = feat_params["gain_scheduler"]
        if feat_params.get("adaptive_q") and "--adaptive-q" not in sys.argv:
            args.adaptive_q = True

    cfg = SystemConfig(
        mc=args.mc, m1=args.m1, m2=args.m2, m3=args.m3,
        L1=args.L1, L2=args.L2, L3=args.L3, g=args.g,
    )

    run(cfg,
        t_end=args.t_end, dt=args.dt, impulse=args.impulse,
        dist_amplitude=args.dist_amplitude, dist_bandwidth=args.dist_bandwidth,
        seed=args.seed, u_max=args.u_max,
        use_ilqr=args.use_ilqr, ilqr_horizon=args.ilqr_horizon,
        ilqr_iterations=args.ilqr_iterations,
        gain_scheduler_type=args.gain_scheduler,
        adaptive_q=args.adaptive_q,
        no_display=args.no_display)


if __name__ == "__main__":
    main()
