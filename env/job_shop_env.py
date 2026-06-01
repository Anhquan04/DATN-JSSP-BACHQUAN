
import numpy as np
import gymnasium as gym
from typing import Optional, List, Tuple, Dict

# Gymnasium optional — fallback nếu chưa cài
try:
    _BASE = gym.Env
    class _Discrete:
        def __init__(self, n): self.n = n
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.shape = shape; self.dtype = dtype
    _Discrete_cls = gym.spaces.Discrete
    _Box_cls = gym.spaces.Box
    _use_gym = True
except ImportError:
    class _BASE: pass
    class _Discrete_cls:
        def __init__(self, n): self.n = n
    class _Box_cls:
        def __init__(self, low, high, shape, dtype):
            self.shape = shape; self.dtype = dtype
    _use_gym = False


class JobShopEnv(_BASE):
    """
    Môi trường mô phỏng Job-Shop Scheduling Problem.

    jobs_data[i][j] = (machine_id, processing_time)
      → Operation j của Job i chạy trên máy machine_id, mất processing_time.

    MDP:
      State  : Vector encode trạng thái máy + tiến độ job (float32, normalized)
      Action : Chọn job_id để schedule operation tiếp theo của job đó
      Reward : Dense reward: phạt thời gian rỗi, thưởng hoàn thành
    """

    def __init__(self, jobs_data: List[List[Tuple[int, int]]]):
        if _use_gym:
            super().__init__()

        self.jobs_data   = jobs_data
        self.n_jobs      = len(jobs_data)
        self.n_machines  = max(op[0] for job in jobs_data for op in job) + 1
        self.n_ops_per_job = [len(job) for job in jobs_data]
        self.max_time    = sum(t for job in jobs_data for _, t in job)

        self.action_space      = _Discrete_cls(self.n_jobs)
        state_size             = self.n_jobs * 3 + self.n_machines
        self.observation_space = _Box_cls(0.0, 1.0, (state_size,), np.float32)

        self._init_state_vars()

    # ── Gymnasium API

    def reset(self, seed: Optional[int] = None, options=None):
        if _use_gym:
            super().reset(seed=seed)
        self._init_state_vars()
        return self._get_state(), {}

    def step(self, action: int):
        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            if not valid_actions:
                return self._get_state(), -1.0, True, False, {}
            action = valid_actions[0]

        job_id   = action
        op_index = self.job_op_index[job_id]
        machine_id, proc_time = self.jobs_data[job_id][op_index]

        start_time = max(self.machine_available_at[machine_id],
                         self.job_available_at[job_id])
        end_time   = start_time + proc_time
        idle       = start_time - self.machine_available_at[machine_id]

        self.machine_available_at[machine_id] = end_time
        self.job_available_at[job_id]         = end_time
        self.job_op_index[job_id]            += 1
        self.current_time = max(self.machine_available_at)

        self.schedule.append({
            "job": job_id, "machine": machine_id,
            "start": start_time, "end": end_time, "op_index": op_index
        })

        reward     = self._calculate_reward(job_id, idle)
        terminated = self._is_done()

        info = {
            "makespan"   : self.get_makespan()    if terminated else None,
            "idle_time"  : self.get_idle_time()   if terminated else None,
            "utilization": self.get_utilization() if terminated else None,
            "schedule"   : self.schedule          if terminated else None,
        }
        return self._get_state(), reward, terminated, False, info

    def render(self, mode="human"):
        print(f"\n{'─'*45}")
        print(f"  Time     : {self.current_time}")
        print(f"  Job ops  : {self.job_op_index}")
        print(f"  Machines : {self.machine_available_at}")

    # ── State & Reward 

    def _get_state(self) -> np.ndarray:
        """
        Encode state → float32 vector normalized [0,1].

        Per job  (3 feats): progress, time_remaining_norm, is_done
        Per machine (1 feat): available_time_norm
        """
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        idx = 0
        for job_id in range(self.n_jobs):
            total_ops = self.n_ops_per_job[job_id]
            done_ops  = self.job_op_index[job_id]
            remaining = sum(self.jobs_data[job_id][op][1]
                            for op in range(done_ops, total_ops))
            state[idx]     = done_ops / total_ops
            state[idx + 1] = remaining / (self.max_time + 1e-8)
            state[idx + 2] = 1.0 if done_ops >= total_ops else 0.0
            idx += 3
        for m in range(self.n_machines):
            state[idx] = self.machine_available_at[m] / (self.max_time + 1e-8)
            idx += 1
        return state

    def _calculate_reward(self, job_id: int, idle: float) -> float:
        """
        Dense reward:
          -1  mỗi bước (time penalty)
          -0.5 * idle (phạt máy rỗi)
          +10 khi job hoàn thành
          +50 * efficiency khi tất cả jobs xong
        """
        reward  = -1.0
        reward -= idle * 0.5
        if self.job_op_index[job_id] >= self.n_ops_per_job[job_id]:
            reward += 10.0
        if self._is_done():
            makespan    = max(self.machine_available_at)
            lower_bound = max(sum(t for _, t in job) for job in self.jobs_data)
            efficiency  = lower_bound / (makespan + 1e-8)
            reward += 50.0 * efficiency
        return reward

    # ── Helpers 

    def _init_state_vars(self):
        self.current_time         = 0
        self.job_op_index         = [0] * self.n_jobs
        self.machine_available_at = [0] * self.n_machines
        self.job_available_at     = [0] * self.n_jobs
        self.schedule             = []

    def _is_done(self) -> bool:
        return all(self.job_op_index[j] >= self.n_ops_per_job[j]
                   for j in range(self.n_jobs))

    def get_valid_actions(self) -> List[int]:
        return [j for j in range(self.n_jobs)
                if self.job_op_index[j] < self.n_ops_per_job[j]]

    def get_makespan(self) -> float:
        return max(self.machine_available_at)

    def get_idle_time(self) -> float:
        makespan   = self.get_makespan()
        total_busy = sum(e["end"] - e["start"] for e in self.schedule)
        return makespan * self.n_machines - total_busy

    def get_utilization(self) -> float:
        makespan = self.get_makespan()
        if makespan == 0:
            return 0.0
        total_busy = sum(e["end"] - e["start"] for e in self.schedule)
        return total_busy / (makespan * self.n_machines)
