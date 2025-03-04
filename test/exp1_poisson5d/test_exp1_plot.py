"""Run the plotting script from exp16."""

from rla_pinns.exp1_poisson5d import plot
from rla_pinns.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]
    run_verbose(cmd)
