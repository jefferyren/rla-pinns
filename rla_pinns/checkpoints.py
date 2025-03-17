from os import path
from glob import glob
from torch import load
from numpy import log10, sum
from torch.nn import Sequential
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
from scipy.sparse.linalg import eigsh
from rla_pinns.train import set_up_layers


def evaluate_checkpoint(checkpoint: str):
    """Evaluate a single checkpoint and return its eigenvalues."""
    checkpoint_name = path.splitext(path.basename(checkpoint))[0]
    print(f"Processing checkpoint {checkpoint_name}.")
    data = load(checkpoint)

    config = data["config"]
    equation = config["equation"]
    dim_Omega = config["dim_Omega"]
    architecture = config["model"]

    X_Omega = data["X_Omega"]
    y_Omega = data["y_Omega"]
    X_dOmega = data["X_dOmega"]
    y_dOmega = data["y_dOmega"]

    layers = set_up_layers(architecture, equation, dim_Omega)
    layers = [layer.to(X_Omega.device, X_Omega.dtype) for layer in layers]
    model = Sequential(*layers).to(X_Omega.device)
    model.load_state_dict(data["model"])

    JJT = ...

    eigenvalues, _ = eigsh(JJT, k=100)
    print("G Rank:", sum(eigenvalues > 1e-10), flush=True)
    return log10(eigenvalues)


def main():
    """Visualize eigenvalues for each checkpoint."""
    parser = ArgumentParser(description="Plot solutions of checkpoints.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="sweeps/checkpoints",
        help="Directory containing checkpoints that should be visualized.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="sweeps/plots",
        help="Directory where plots should be saved.",
    )
    parser.add_argument(
        "--equation",
        type=str,
        default="heat",
        help="Equation to solve.",
    )
    args = parser.parse_args()
    checkpoint_dir = path.abspath(args.checkpoint_dir)

    all_eigenvalues = []
    checkpoint_steps = []

    # Filter checkpoints based on equation
    for i, checkpoint in enumerate(sorted(glob(path.join(checkpoint_dir, "*.pt")))):
        checkpoint_name = path.splitext(path.basename(checkpoint))[0]
        first_word = checkpoint_name.split("_")[0]  # Extract the first word
        if first_word != args.equation:
            continue  # Skip if the first word doesn't match the equation

        eigenvalues = evaluate_checkpoint(checkpoint)
        all_eigenvalues.append(eigenvalues)
        checkpoint_steps.append(i + 1)

    if not all_eigenvalues:
        print(f"No checkpoints matched the equation '{args.equation}'. Exiting.")
        return

    # Plot all eigenvalues in a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, eigenvalues in enumerate(all_eigenvalues):
        ax.plot(
            range(1, len(eigenvalues) + 1),
            [checkpoint_steps[i]] * len(eigenvalues),
            eigenvalues,
            marker="o",
        )

    ax.set_xlabel("Eigenvalue Ranking")
    ax.set_ylabel("Epoch")
    ax.set_zlabel("Log10(Eigenvalue)")
    ax.set_title(f"Eigenvalues Over Checkpoints for {args.equation}")
    plt.grid()
    plt.savefig(f"{args.plot_dir}/eigenvalues_3d_{args.equation}.png")
    plt.show()


if __name__ == "__main__":
    main()
