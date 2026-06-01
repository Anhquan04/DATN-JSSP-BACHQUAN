
import numpy as np
from typing import List, Tuple


# ── Benchmark Instances Chuẩn 

# Fisher-Thompson 6x6 — Optimal makespan = 55
FT06 = [
    [(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],
    [(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],
    [(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],
    [(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],
    [(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],
    [(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)],
]

# Fisher-Thompson 10x10 — Optimal makespan = 930
FT10 = [
    [(0,29),(1,78),(2,9),(3,36),(4,49),(5,11),(6,62),(7,56),(8,44),(9,21)],
    [(0,43),(2,90),(4,75),(9,11),(3,69),(1,28),(6,46),(5,46),(7,72),(8,30)],
    [(1,91),(0,85),(3,39),(2,74),(8,90),(5,10),(7,12),(6,89),(9,45),(4,33)],
    [(1,81),(2,95),(0,71),(4,99),(6,9),(8,52),(7,85),(3,98),(9,22),(5,43)],
    [(2,14),(0,6),(1,22),(5,61),(3,26),(4,69),(8,21),(7,49),(9,72),(6,53)],
    [(2,84),(1,2),(5,52),(3,95),(8,48),(9,72),(0,47),(6,65),(4,6),(7,25)],
    [(1,46),(0,37),(3,61),(2,13),(6,32),(5,32),(8,57),(9,60),(4,42),(7,70)],
    [(0,31),(1,86),(2,46),(5,74),(4,32),(6,88),(8,19),(9,48),(7,36),(3,79)],
    [(0,76),(1,69),(3,76),(5,51),(2,85),(9,11),(6,40),(7,89),(4,26),(8,74)],
    [(1,85),(0,13),(2,61),(6,7),(8,64),(9,76),(5,47),(3,52),(4,90),(7,45)],
]

# Instance 3x3 nhỏ để test nhanh
SIMPLE_3x3 = [
    [(0,3),(1,2),(2,2)],
    [(0,2),(2,1),(1,4)],
    [(1,4),(2,3)],
]

# Instance 4x4
DEMO_4x4 = [
    [(0,4),(1,2),(2,5),(3,1)],
    [(1,3),(0,3),(3,2),(2,4)],
    [(2,6),(1,1),(0,2),(3,5)],
    [(3,2),(2,3),(1,4),(0,6)],
]

# Instance 5x5
DEMO_5x5 = [
    [(0,7),(1,3),(2,5),(3,2),(4,6)],
    [(1,5),(0,4),(4,3),(3,7),(2,2)],
    [(2,4),(3,6),(1,2),(4,5),(0,3)],
    [(3,3),(4,4),(0,6),(2,4),(1,5)],
    [(4,6),(2,5),(3,3),(0,4),(1,7)],
]


def generate_random_instance(
    n_jobs: int,
    n_machines: int,
    min_time: int = 1,
    max_time: int = 10,
    seed: int = 42
) -> List[List[Tuple[int, int]]]:
    """Sinh instance JSSP ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    jobs_data = []
    for _ in range(n_jobs):
        machines = rng.permutation(n_machines).tolist()
        times    = rng.integers(min_time, max_time + 1, size=n_machines).tolist()
        jobs_data.append([(machines[k], times[k]) for k in range(n_machines)])
    return jobs_data


def get_instance(name: str) -> List[List[Tuple[int, int]]]:
    """
    Lấy instance theo tên.
    Tên hợp lệ: '3x3', '4x4', '5x5', 'ft06', 'ft10', 'random_NxM'
    """
    catalog = {
        "3x3"  : SIMPLE_3x3,
        "4x4"  : DEMO_4x4,
        "5x5"  : DEMO_5x5,
        "ft06" : FT06,
        "ft10" : FT10,
    }
    if name in catalog:
        return catalog[name]
    if name.startswith("random_"):
        n, m = map(int, name.split("_")[1].split("x"))
        return generate_random_instance(n, m)
    raise ValueError(f"Unknown instance '{name}'. Options: {list(catalog.keys())} or 'random_NxM'")


def instance_info(jobs_data: List[List[Tuple[int, int]]]) -> dict:
    """Trả về thông tin cơ bản của instance."""
    n_jobs     = len(jobs_data)
    n_machines = max(op[0] for job in jobs_data for op in job) + 1
    total_time = sum(t for job in jobs_data for _, t in job)
    lb         = max(sum(t for _, t in job) for job in jobs_data)
    return {
        "n_jobs"                : n_jobs,
        "n_machines"            : n_machines,
        "total_processing_time" : total_time,
        "critical_path_lb"      : lb,
    }


def export_for_game(jobs_data: List[List[Tuple[int, int]]]) -> dict:
    """Export instance sang format JSON cho game visualization."""
    info = instance_info(jobs_data)
    return {
        "n_jobs"    : info["n_jobs"],
        "n_machines": info["n_machines"],
        "jobs"      : [
            [{"machine": m, "duration": t} for m, t in job]
            for job in jobs_data
        ],
    }
