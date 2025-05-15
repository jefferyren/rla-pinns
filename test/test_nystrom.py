#!/usr/bin/env python3
import torch
import time
import argparse
from rla_pinns.optim.rand_utils import nystrom_stable, nystrom_stable_fast


def test_nystrom_stable(input_tensor, l, mu):
    n = input_tensor.shape[0]
    dt, dev = input_tensor.dtype, input_tensor.device
    g = torch.ones(n, device=dev)
    U, Lambda = nystrom_stable(A=input_tensor.__matmul__, dim=n, sketch_size=l, dt=dt, dev=dev)
    UTg = U.T @ g
    lhs = (g - U @ UTg) / mu
    rhs = U @ (torch.diag(1 / (Lambda + mu)) @ UTg)

    return lhs + rhs


def apply_B(B, mu, g):
    """Apply the inverse of B to g."""
    BTB = B.T @ B
    idx = torch.arange(BTB.shape[0], device=B.device)
    BTB[idx, idx] = BTB.diag() + mu

    L = torch.linalg.cholesky(BTB)
    BTg = B.T @ g

    invBTg = torch.cholesky_solve(BTg.unsqueeze(-1), L).squeeze(-1)
    P_inv = B @ invBTg
    out = (g - P_inv) / mu
    return out


def test_nystrom_fast(input_tensor, l, mu):
    n = input_tensor.shape[0]
    dt, dev = input_tensor.dtype, input_tensor.device
    g = torch.ones(n, device=dev)
    B = nystrom_stable_fast(A=input_tensor.__matmul__, dim=n, sketch_size=l, dt=dt, dev=dev)
    out = apply_B(B, mu, g)
    return out


def benchmark(func, input_tensor, l, mu, iterations=10, device='cpu'):
    """
    Benchmark `func` on `input_tensor` for a given number of iterations.
    Includes a warm-up phase and synchronizes CUDA if needed.
    """
    # Warm-up runs
    for _ in range(10):
        _ = func(input_tensor, l, mu)
    # Ensure all queued CUDA ops complete before timing
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # Timed runs
    start_time = time.time()
    for _ in range(iterations):
        _ = func(input_tensor, l, mu)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    return avg_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark two PyTorch functions on CPU or GPU"
    )
    parser.add_argument(
        '--device', choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run benchmarks on (default: cuda if available)"
    )
    parser.add_argument(
        '--n', type=int, default=3500,
    )
    parser.add_argument(
        '--p', type=int, default=10000,
    )
    parser.add_argument(
        '--l', type=int, default=250,
    )
    parser.add_argument(
        '--mu', type=float, default=1e-7,
    )
    parser.add_argument(
        '--iterations', type=int, default=100,
        help="Number of iterations to run for each benchmark"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create random input tensor on the specified device
    input_tensor = torch.randn(args.n, args.p, device=device)
    input_tensor = input_tensor.T @ input_tensor # Make it symmetric

    # Benchmark function1
    time1 = benchmark(test_nystrom_stable, input_tensor, args.l, args.mu, iterations=args.iterations, device=args.device)
    print(f"Average execution time of function1 on {device}: {time1:.6f} seconds")

    # Benchmark function2
    time2 = benchmark(test_nystrom_fast, input_tensor, args.l, args.mu, iterations=args.iterations, device=args.device)
    print(f"Average execution time of function2 on {device}: {time2:.6f} seconds")

    # Compare and report
    if time1 < time2:
        print(f"GPU-efficient is faster than Stable by {time2 / time1:.2f}x seconds per iteration")
    else:
        print(f"Stable is faster than GPU-efficient by {time1 / time2:.2f}x seconds per iteration")

    


if __name__ == '__main__':
    main()
