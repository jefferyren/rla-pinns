
from torch import cuda, device, randn, manual_seed, eye
from torch.linalg import qr, cholesky, svd, eigh
from time import time

device = device("cuda" if cuda.is_available() else "cpu")

def time_matrix_function(fn, A, nreps):
    fn(A)

    cuda.synchronize() if device.type == "cuda" else None
    start_time = time()

    for _ in range(nreps - 1):
        fn(A)

    cuda.synchronize() if device.type == "cuda" else None
    end_time = time()

    print(f"{(end_time - start_time) / nreps:.2e}")


def svd_fn(A):
    return svd(A, full_matrices=False)


N = 4000
S = 1000
nreps = 50
manual_seed(42)

ANN = randn(N, N, device=device)
ANS = randn(N, S, device=device)
ASS = randn(S, S, device=device)
ASS = ASS @ ASS.T

ANN_chol = ANN @ ANN.T + eye(N, device=device)

print("Cholesky on ASS:")
time_matrix_function(cholesky, ASS, nreps)

print("Cholesky on ANN:")
time_matrix_function(cholesky, ANN_chol, nreps)

print("QR on ANS:")
time_matrix_function(qr, ANS, nreps)

print("SVD on ANS:")
time_matrix_function(svd_fn, ANS, nreps)

print("Eigh on ASS:")
time_matrix_function(eigh, ASS, nreps)

print("SVD on ASS:")
time_matrix_function(svd_fn, ASS, nreps)

