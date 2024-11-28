import numpy as np
from numba import cuda
from cmath import exp, pi

@cuda.jit
def fft_kernel(x, y, N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < N:
        s = 0 + 0j
        for n in range(N):
            angle = 2j * pi * n * idx / N
            s += x[n] * exp(-angle)
        y[idx] = s

def main():
    N = 1024  # Número de puntos en la FFT
    x = np.exp(2j * pi * np.arange(N) / N).astype(np.complex64)  # Señal de entrada (e^j2πk/N)
    y = np.zeros_like(x)  # Salida de la FFT

    # Copiar los datos a la memoria de la GPU
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)

    # Ajustar el número de hilos por bloque y bloques por malla
    threads_per_block = 256  # Número de hilos por bloque
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block  # Ajustar el número de bloques

    # Asegurarse de que se utiliza la GPU de manera más eficiente
    print(f"Ejecutando con {blocks_per_grid} bloques y {threads_per_block} hilos por bloque.")

    # Llamar al kernel
    fft_kernel[blocks_per_grid, threads_per_block](d_x, d_y, N)

    # Copiar el resultado de vuelta al host
    y = d_y.copy_to_host()

    print(y)

if __name__ == "__main__":
    main()
