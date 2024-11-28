import numpy as np
from numba import cuda

@cuda.jit
def multiply_matrices(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def main():
    N = 32  # Tamaño de las matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    # Copiar matrices a la GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array(C.shape, dtype=np.float32)

    # Configuración de hilos y bloques
    threads_per_block = (16, 16)
    blocks_per_grid_x = (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (B.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Llamada al kernel
    multiply_matrices[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

    # Copiar resultados de vuelta al host
    C = d_C.copy_to_host()

    print(C)

if __name__ == "__main__":
    main()
