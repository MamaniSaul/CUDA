import numpy as np
import pygame
from numba import cuda, float32
import random

# Kernel para actualizar la posición de las partículas
@cuda.jit
def update_particles(positions, velocities, dt):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        positions[idx, 0] += velocities[idx, 0] * dt
        positions[idx, 1] += velocities[idx, 1] * dt

def main():
    # Parámetros de configuración
    num_particles = 1000
    width, height = 800, 600
    dt = 1.0  # incremento temporal

    # Inicializar Pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # Inicializar datos de partículas
    positions = np.random.rand(num_particles, 2).astype(np.float32) * [width, height]
    velocities = (np.random.rand(num_particles, 2) - 0.5).astype(np.float32)

    # Copiar los datos de posiciones y velocidades a la GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)

    # Configurar la ejecución del kernel
    threads_per_block = 256
    blocks = (num_particles + threads_per_block - 1) // threads_per_block

    running = True
    while running:
        # Manejar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Actualizar físicas en la GPU
        update_particles[blocks, threads_per_block](d_positions, d_velocities, dt)

        # Copiar posiciones actualizadas al host
        positions = d_positions.copy_to_host()

        # Renderizar partículas
        screen.fill((0, 0, 0))
        for x, y in positions:
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 2)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()