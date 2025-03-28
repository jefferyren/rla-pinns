from torch import cuda, device, manual_seed
from torch.nn import Sequential
from rla_pinns.optim import set_up_optimizer
from rla_pinns.parse_utils import check_all_args_parsed
from rla_pinns.train import parse_general_args, create_data_loader,set_up_layers

def main():  # noqa: C901
    """Execute training with the specified command line arguments."""
    # NOTE Do not move this down as the parsers remove arguments from argv
    args = parse_general_args(verbose=True)
    dev, dt = device("cuda" if cuda.is_available() else "cpu"), args.dtype
    print(f"Running on device {str(dev)} in dtype {dt}.")

    # DATA LOADERS
    manual_seed(args.data_seed)
    equation, condition = args.equation, args.boundary_condition
    dim_Omega, N_Omega, N_dOmega = args.dim_Omega, args.N_Omega, args.N_dOmega

    # for satisfying the PDE on the domain
    interior_train_data_loader1 = iter(
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
    condition_train_data_loader1 = iter(
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
    interior_train_data_loader2 = iter(
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
    condition_train_data_loader2 = iter(
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
    # NEURAL NET 1
    layers_SPRING = set_up_layers(args.model, equation, dim_Omega)
    layers_SPRING = [layer.to(dev, dt) for layer in layers_SPRING]
    model_SPRING = Sequential(*layers_SPRING).to(dev)
    print(f"Model: {model_SPRING}")
    print(f"Number of parameters: {sum(p.numel() for p in model_SPRING.parameters())}")

    # SPRING OPTIMIZER
    optimizer_SPRING, _ = set_up_optimizer(
        layers_SPRING, "SPRING", equation, verbose=True
    )


    # NEURAL NET 2
    layers_RNGD = set_up_layers(args.model, equation, dim_Omega)
    layers_RNGD = [layer.to(dev, dt) for layer in layers_RNGD]
    model_RNGD = Sequential(*layers_RNGD).to(dev)
    print(f"Model: {model_RNGD}")
    print(f"Number of parameters: {sum(p.numel() for p in model_RNGD.parameters())}")

    # RNGD OPTIMIZER
    optimizer_RNGD, _ = set_up_optimizer(
        layers_RNGD, "RNGD", equation, verbose=True
    )

    check_all_args_parsed()

    # check that the equation was correctly passed to PDE-aware optimizers
    assert optimizer_SPRING.equation == optimizer_RNGD.equation == equation

    # TRAINING
    for step in range(5):
        # load next batch of data
        X_Omega1, y_Omega1 = next(interior_train_data_loader1)
        X_dOmega1, y_dOmega1 = next(condition_train_data_loader1)

        optimizer_SPRING.zero_grad()
        loss_interior_SPRING, loss_boundary_SPRING = optimizer_SPRING.step(X_Omega1, y_Omega1, X_dOmega1, y_dOmega1)
        loss_boundary_SPRING, loss_interior_SPRING = loss_boundary_SPRING.item(), loss_interior_SPRING.item()
        loss_SPRING = loss_boundary_SPRING + loss_interior_SPRING

        X_Omega2, y_Omega2 = next(interior_train_data_loader2)
        X_dOmega2, y_dOmega2 = next(condition_train_data_loader2)

        optimizer_RNGD.zero_grad()
        loss_interior_RNGD, loss_boundary_RNGD = optimizer_RNGD.step(X_Omega2, y_Omega2, X_dOmega2, y_dOmega2)
        loss_boundary_RNGD, loss_interior_RNGD = loss_boundary_RNGD.item(), loss_interior_RNGD.item()
        loss_RNGD = loss_boundary_RNGD + loss_interior_RNGD

        assert abs(loss_RNGD - loss_SPRING) < 1e-5,  f"Loss: {loss_SPRING} vs {loss_RNGD}"

        print(f"Step [{step + 1}/50], Loss: {loss_boundary_SPRING + loss_interior_SPRING:.10f}.", flush=True)


if __name__ == "__main__":
    main()