import os
import psutil
from concurrent.futures import ProcessPoolExecutor

def work(x):
    # Load worker's affinity
    p = psutil.Process(os.getpid())
    aff = p.cpu_affinity()

    # Small CPU load
    s = 0
    for i in range(1000_000_00):
        s += i * i

    return (x, aff, s)

if __name__ == "__main__":
    # Parent affinity
    parent = psutil.Process(os.getpid())
    parent.cpu_affinity([0, 1, 2, 3])
    print("Parent affinity:", parent.cpu_affinity())

    print("Launching workers…")
    with ProcessPoolExecutor(max_workers=18) as ex:
        results = list(ex.map(work, range(10)))

    print("\nWorker results:")
    for r in results:
        print(r)
