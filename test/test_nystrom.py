#!/usr/bin/env python3
import torch
import time
import argparse
import os
from datetime import datetime
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
    BTB = B.T @ B
    idx = torch.arange(BTB.shape[0], device=B.device)
    BTB[idx, idx] = BTB.diag() + mu
    L = torch.linalg.cholesky(BTB)
    BTg = B.T @ g
    invBTg = torch.cholesky_solve(BTg.unsqueeze(-1), L).squeeze(-1)
    return (g - (B @ invBTg)) / mu


def test_nystrom_fast(input_tensor, l, mu):
    n = input_tensor.shape[0]
    dt, dev = input_tensor.dtype, input_tensor.device
    g = torch.ones(n, device=dev)
    B = nystrom_stable_fast(A=input_tensor.__matmul__, dim=n, sketch_size=l, dt=dt, dev=dev)
    return apply_B(B, mu, g)


def benchmark(func, input_tensor, l, mu, iterations=10, device='cpu'):
    is_cuda = device.startswith('cuda')

    # Warm-up
    for _ in range(10):
        _ = func(input_tensor, l, mu)
    if is_cuda:
        torch.cuda.synchronize()

    # Reset peak-memory stats on CUDA
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    start_time = time.time()
    for _ in range(iterations):
        _ = func(input_tensor, l, mu)
    if is_cuda:
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations

    # Measure peak memory
    peak_mem = None
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return avg_time, peak_mem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Nystrom implementations with time, memory, and metrics logging"
    )
    parser.add_argument(
        '--device', choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run benchmarks on (default: cuda if available)"
    )
    parser.add_argument('--n', type=int, default=3500, help="Matrix size (n x n)")
    parser.add_argument('--l', type=int, default=1750, help="Sketch size")
    parser.add_argument('--mu', type=float, default=1e-7, help="Regularization parameter")
    parser.add_argument('--iterations', type=int, default=100,
                        help="Number of iterations per benchmark")
    parser.add_argument('--stable', action='store_true', help="Run the Stable implementation")
    parser.add_argument('--fast', action='store_true', help="Run the Fast implementation")
    parser.add_argument('--output', type=str, default='metrics.txt',
                        help="Output file to append benchmark metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # If no flag is set, run both
    run_stable = args.stable or not (args.stable or args.fast)
    run_fast = args.fast or not (args.stable or args.fast)

    # Prepare output file header if needed
    if not os.path.exists(args.output):
        with open(args.output, 'w') as f:
            f.write("timestamp,device,n,l,mu,iterations,impl,avg_time_s,peak_mem_MiB\n")

    # Create symmetric input tensor
    input_tensor = torch.randn(args.n, args.n, device=device)
    input_tensor = input_tensor.T @ input_tensor

    # Helper to log results
    def log_metrics(impl_name, avg_time, peak_mem):
        ts = datetime.now().isoformat()
        with open(args.output, 'a') as f:
            f.write(
                f"{ts},{args.device},{args.n},{args.l},{args.mu},{args.iterations},"
                f"{impl_name},{avg_time:.6f},{peak_mem if peak_mem is not None else ''}\n"
            )

    # Benchmark Stable
    if run_stable:
        t_st, m_st = benchmark(test_nystrom_stable, input_tensor, args.l, args.mu,
                               iterations=args.iterations, device=args.device)
        print(f"[Stable] Avg time: {t_st:.6f} s", end='')
        if m_st is not None:
            print(f", Peak GPU mem: {m_st:.1f} MiB")
        else:
            print()
        log_metrics('stable', t_st, m_st)

    # Benchmark Fast
    if run_fast:
        t_f, m_f = benchmark(test_nystrom_fast, input_tensor, args.l, args.mu,
                             iterations=args.iterations, device=args.device)
        print(f"[Fast]   Avg time: {t_f:.6f} s", end='')
        if m_f is not None:
            print(f", Peak GPU mem: {m_f:.1f} MiB")
        else:
            print()
        log_metrics('fast', t_f, m_f)

    # Summary if both ran
    if run_stable and run_fast:
        faster = 'Fast' if t_f < t_st else 'Stable'
        speedup = (t_st / t_f) if t_f < t_st else (t_f / t_st)
        print(f"\nResult: {faster} faster by {speedup:.2f}× per iteration")


if __name__ == '__main__':
    main()

