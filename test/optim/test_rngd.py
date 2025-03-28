from typing import List
from pytest import mark
from rla_pinns.utils import run_verbose

from test.exp1_poisson5d import rngd

ARGS = [
    # train with SPRING and RNGD on different equations
    *[
        [
            "--optimizer=RNGD", # NOTE: this is a placeholder, the actual optimizer is set in the script
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--SPRING_decay_factor=0.9",
            "--SPRING_damping=1e-10",
            "--SPRING_lr=0.001",
            "--RNGD_approximation=exact",
            "--RNGD_momentum=0.9",
            "--RNGD_damping=1e-10",
            "--RNGD_lr=0.001",
            "--model=mlp-tanh-64",
            "--N_Omega=1000",
            "--N_dOmega=200",
            "--batch_frequency=10000",
            "--dim_Omega=2",
        ]
        for (equation, condition) in list(
            [
                ("poisson", "cos_sum"),
                ("heat", "cos_sum"),
                ("fokker-planck-isotropic", "gaussian"),
                ("log-fokker-planck-isotropic", "gaussian"),
            ]
        )
    ],
]


ARG_IDS = ["_".join(cmd) for cmd in ARGS]


@mark.parametrize("args", ARGS, ids=ARG_IDS)
def test_rngd(args: List[str]) -> None:
    """Test the training script (integration test)."""
    # Run the training script with the provided arguments
    run_verbose(["python", rngd.__file__] + args)
