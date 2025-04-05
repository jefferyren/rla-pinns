from torch import cuda, device, manual_seed, allclose, zeros
from torch.nn import Sequential
from rla_pinns.optim import set_up_optimizer
from rla_pinns.train import parse_general_args, create_data_loader, set_up_layers


def main():  # noqa: C901
    """Execute training with the specified command line arguments."""
    # NOTE Do not move this down as the parsers remove arguments from argv
    args = parse_general_args(verbose=True)
    dev, dt = device("cuda" if cuda.is_available() else "cpu"), args.dtype
    print(f"Running on device {str(dev)} in dtype {dt}.")

    losses_SPRING = run(args, dev, dt, "SPRING")
    losses_RNGD = run(args, dev, dt, "RNGD")

    assert allclose(
        losses_SPRING, losses_RNGD
    ), f"Losses do not match: {losses_SPRING} != {losses_RNGD}"


def run(args, dev, dt, optim, n=5):
    print(f"Running on device {str(dev)} in dtype {dt}.")

    # DATA LOADERS
    manual_seed(args.data_seed)
    equation, condition = args.equation, args.boundary_condition
    dim_Omega, N_Omega, N_dOmega = args.dim_Omega, args.N_Omega, args.N_dOmega

    # for satisfying the PDE on the domain
    interior_train_data_loader = iter(
        create_data_loader(
            args.batch_frequency,
            "interior",
            equation,
            condition,
            dim_Omega,
            N_Omega,
            dev,
            dt,
        )
    )
    # for satisfying boundary and (maybe) initial conditions
    condition_train_data_loader = iter(
        create_data_loader(
            args.batch_frequency,
            "condition",
            equation,
            condition,
            dim_Omega,
            N_dOmega,
            dev,
            dt,
        )
    )

    manual_seed(args.model_seed)
    # NEURAL NET
    layers = set_up_layers(args.model, equation, dim_Omega)
    layers = [layer.to(dev, dt) for layer in layers]
    model = Sequential(*layers).to(dev)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # SPRING OPTIMIZER
    optimizer, _ = set_up_optimizer(layers, optim, equation, verbose=True)

    # check that the equation was correctly passed to PDE-aware optimizers
    assert optimizer.equation == equation

    # TRAINING
    losses = zeros(n)
    for i in range(n):
        X_Omega, y_Omega = next(interior_train_data_loader)
        X_dOmega, y_dOmega = next(condition_train_data_loader)

        optimizer.zero_grad()
        loss_interior, loss_boundary = optimizer.step(
            X_Omega, y_Omega, X_dOmega, y_dOmega
        )
        losses[i] = loss_boundary.item() + loss_interior.item()

    return losses


if __name__ == "__main__":
    main()
