import numpy as np
from agd.Metrics import Riemann

Nt, Nx, Ny = 16, 32, 32
metric_np = np.zeros((Nt, Nx, Ny, 3, 3), dtype=np.float32)

for t in range(Nt):
    for x in range(Nx):
        for y in range(Ny):
            metric_np[t, x, y] = np.eye(3, dtype=np.float32)

print("metric_np shape:", metric_np.shape, "dtype:", metric_np.dtype)
geom = Riemann(metric_np)  # Should succeed without shape assertion
