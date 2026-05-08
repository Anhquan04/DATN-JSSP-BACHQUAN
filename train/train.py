"""
main.py — Flask Web App: Job-Shop Scheduling Visualization
===========================================================
Entry point của toàn bộ project.

Cách chạy:
    python main.py

Mở trình duyệt: http://localhost:5000

Routes:
    GET  /                          → Trang game visualization
    GET  /api/instances             → Danh sách instances có sẵn
    GET  /api/schedule/<inst>/<algo>→ Lấy schedule data
    GET  /api/results               → Kết quả training tất cả instances
    POST /api/train                 → Chạy training mới (background)
    GET  /api/train/status          → Trạng thái training
    GET  /api/compare/<inst>        → So sánh tất cả algorithms trên 1 instance

Tác giả: Bạch Công Quân — ĐATN 2026
GVHD  : ThS. Tạ Chí Hiếu
"""

import os, sys, json, time, threading
import numpy as np
from flask import Flask, render_template, jsonify, request


# ── Fix: numpy int64/float64 không serialize được sang JSON ──────────────────
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder — tự động convert numpy types → Python native types."""
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.bool_):     return bool(obj)
        return super().default(obj)


def _to_python(obj):
    """Đệ quy convert toàn bộ dict/list chứa numpy types sang Python native."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(i) for i in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.bool_):    return bool(obj)
    return obj

# Thêm root vào sys.path để import các module trong project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.job_shop_env    import JobShopEnv
from agent.a2c_numpy     import A2CAgentNumpy
from baselines.dispatching_rules import run_dispatching_rule, evaluate_all_baselines
from data.instances      import get_instance, instance_info, export_for_game

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False
app.json_encoder = NumpyEncoder  # Dùng encoder tùy chỉnh

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Training status (dùng cho /api/train/status) ──────────────────────────────
_train_status = {
    "running" : False,
    "instance": None,
    "episode" : 0,
    "total"   : 0,
    "best_ms" : None,
    "log"     : [],
}


# ════════════════════════════════════════════════════════════════════════════
#  PAGES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Trang chính — Game visualization."""
    return render_template("index.html")


# ════════════════════════════════════════════════════════════════════════════
#  API — DATA
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/instances")
def api_instances():
    """Trả về danh sách instances và thông tin cơ bản."""
    catalog = {
        "3x3" : {"label": "Simple 3×3",       "n_jobs": 3, "n_machines": 3},
        "4x4" : {"label": "Demo 4×4",          "n_jobs": 4, "n_machines": 4},
        "5x5" : {"label": "Demo 5×5",          "n_jobs": 5, "n_machines": 5},
        "ft06": {"label": "FT06 Benchmark 6×6","n_jobs": 6, "n_machines": 6},
        "ft10": {"label": "FT10 Benchmark 10×10","n_jobs":10,"n_machines":10},
    }
    # Đánh dấu instance nào đã có model trained
    for name in catalog:
        model_path = os.path.join(RESULTS_DIR, f"model_{name}.npz")
        catalog[name]["trained"] = os.path.exists(model_path)
    return jsonify(catalog)


@app.route("/api/schedule/<instance_name>/<algo>")
def api_schedule(instance_name, algo):
    """
    Trả về schedule data cho 1 instance + algorithm.

    Returns JSON:
        {
          instance: {...},
          schedule: [...],
          makespan: int,
          idle_time: float,
          utilization: float,
          algo: str
        }
    """
    try:
        jobs_data = get_instance(instance_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    info = instance_info(jobs_data)

    if algo == "a2c":
        result = _run_a2c(jobs_data, instance_name)
    elif algo in ("fifo", "spt", "lpt", "edd", "random"):
        result = run_dispatching_rule(jobs_data, algo)
    else:
        return jsonify({"error": f"Unknown algorithm: {algo}"}), 400

    return jsonify(_to_python({
        "instance"   : export_for_game(jobs_data),
        "info"       : info,
        "algo"       : algo,
        "schedule"   : result["schedule"],
        "makespan"   : result["makespan"],
        "idle_time"  : result["idle_time"],
        "utilization": result["utilization"],
    }))


@app.route("/api/compare/<instance_name>")
def api_compare(instance_name):
    """
    So sánh tất cả algorithms trên 1 instance.

    Returns JSON:
        {
          instance: {...},
          results: {
            a2c:  {schedule, makespan, ...},
            fifo: {...}, spt: {...}, ...
          }
        }
    """
    try:
        jobs_data = get_instance(instance_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    results = {}
    for algo in ("fifo", "spt", "lpt", "edd"):
        r = run_dispatching_rule(jobs_data, algo)
        results[algo] = {
            "schedule"   : r["schedule"],
            "makespan"   : r["makespan"],
            "idle_time"  : r["idle_time"],
            "utilization": r["utilization"],
        }

    a2c_result = _run_a2c(jobs_data, instance_name)
    results["a2c"] = {
        "schedule"   : a2c_result["schedule"],
        "makespan"   : a2c_result["makespan"],
        "idle_time"  : a2c_result["idle_time"],
        "utilization": a2c_result["utilization"],
    }

    return jsonify(_to_python({
        "instance": export_for_game(jobs_data),
        "info"    : instance_info(jobs_data),
        "results" : results,
    }))


@app.route("/api/results")
def api_results():
    """Trả về kết quả training đã lưu (all_results.json)."""
    path = os.path.join(RESULTS_DIR, "all_results.json")
    if not os.path.exists(path):
        return jsonify({"error": "Chưa có kết quả training. Hãy chạy /api/train trước."}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ════════════════════════════════════════════════════════════════════════════
#  API — TRAINING
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Bắt đầu training A2C trên 1 instance (chạy background thread).

    Body JSON:
        { "instance": "ft06", "episodes": 2000 }
    """
    global _train_status

    if _train_status["running"]:
        return jsonify({"error": "Đang training. Chờ hoàn thành trước."}), 409

    body          = request.get_json(force=True, silent=True) or {}
    instance_name = body.get("instance", "ft06")
    n_episodes    = int(body.get("episodes", 1000))

    _train_status = {
        "running" : True,
        "instance": instance_name,
        "episode" : 0,
        "total"   : n_episodes,
        "best_ms" : None,
        "log"     : [f"🚀 Bắt đầu training {instance_name} ({n_episodes} eps)..."],
    }

    t = threading.Thread(
        target=_background_train,
        args=(instance_name, n_episodes),
        daemon=True,
    )
    t.start()

    return jsonify({"message": f"Training {instance_name} bắt đầu!", "episodes": n_episodes})


@app.route("/api/train/status")
def api_train_status():
    """Polling endpoint — trả về trạng thái training hiện tại."""
    return jsonify(_train_status)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _run_a2c(jobs_data, instance_name: str) -> dict:
    """
    Chạy A2C inference. Load model đã train nếu có,
    nếu không thì chạy random agent (fallback).
    """
    env        = JobShopEnv(jobs_data)
    model_path = os.path.join(RESULTS_DIR, f"model_{instance_name}.npz")

    if os.path.exists(model_path):
        agent = A2CAgentNumpy(
            state_dim =env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dim=128,
        )
        agent.load(model_path)

        # Chạy 20 lần, lấy kết quả tốt nhất
        best_ms, best_sched = float("inf"), None
        for _ in range(20):
            state, _ = env.reset()
            done = False
            while not done:
                valid = env.get_valid_actions()
                a, _, _ = agent.select_action(state, valid)
                state, _, done, _, _ = env.step(a)
            ms = env.get_makespan()
            if ms < best_ms:
                best_ms    = ms
                best_sched = env.schedule.copy()

        # Rebuild env để lấy metrics đúng
        env2 = JobShopEnv(jobs_data)
        env2.reset()
        env2.schedule             = best_sched
        env2.machine_available_at = [
            max((e["end"] for e in best_sched if e["machine"] == m), default=0)
            for m in range(env2.n_machines)
        ]

        return {
            "schedule"   : best_sched,
            "makespan"   : best_ms,
            "idle_time"  : env2.get_idle_time(),
            "utilization": env2.get_utilization(),
        }

    else:
        # Fallback: random agent
        state, _ = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            a     = np.random.choice(valid)
            state, _, done, _, _ = env.step(a)
        return {
            "schedule"   : env.schedule,
            "makespan"   : env.get_makespan(),
            "idle_time"  : env.get_idle_time(),
            "utilization": env.get_utilization(),
        }


def _background_train(instance_name: str, n_episodes: int):
    """
    Hàm training chạy trong background thread.
    Gọi train_numpy() — tự động export CSV sau khi xong.
    """
    global _train_status
    from collections import deque
    from train_numpy import train_numpy, save_training_log_csv, save_evaluation_csv, save_schedule_csv
    from baselines.dispatching_rules import run_dispatching_rule

    try:
        jobs_data = get_instance(instance_name)
        env       = JobShopEnv(jobs_data)
        np.random.seed(42)

        agent = A2CAgentNumpy(
            state_dim    = env.observation_space.shape[0],
            action_dim   = env.action_space.n,
            hidden_dim   = 128, lr_actor=3e-4, lr_critic=5e-4,
            gamma        = 0.99, entropy_coef=0.05,
        )

        best_ms       = float("inf")
        best_schedule = None
        rw            = deque(maxlen=100)
        log_every     = max(1, n_episodes // 10)
        training_rows = []

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            done = False; ep_r = 0
            while not done:
                valid      = env.get_valid_actions()
                a, lp, val = agent.select_action(state, valid)
                ns, r, done, _, _ = env.step(a)
                agent.store_transition(state, a, r, done, lp, val, valid)
                ep_r += r; state = ns

            losses = agent.update()
            ms     = env.get_makespan()
            rw.append(ep_r)

            is_best = ms < best_ms
            if is_best:
                best_ms       = ms
                best_schedule = env.schedule.copy()
                agent.save(os.path.join(RESULTS_DIR, f"model_{instance_name}"))

            training_rows.append({
                "episode"       : ep,
                "reward"        : round(ep_r, 4),
                "makespan"      : ms,
                "avg_reward_100": round(float(np.mean(rw)), 4),
                "critic_loss"   : round(losses.get("critic_loss", 0), 6),
                "entropy"       : round(losses.get("entropy", 0), 6),
                "best_makespan" : best_ms,
                "is_best"       : int(is_best),
            })

            _train_status["episode"] = ep
            _train_status["best_ms"] = best_ms

            if ep % log_every == 0 or ep == 1:
                msg = (f"Ep {ep}/{n_episodes} | AvgR={np.mean(rw):.1f} | "
                       f"MS={ms} | Best={best_ms}")
                _train_status["log"].append(msg)

        # ── Export CSV ─────────────────────────────────────────────────────
        _train_status["log"].append("💾 Đang xuất CSV...")

        # 1. Training log
        save_training_log_csv(
            training_rows,
            os.path.join(RESULTS_DIR, f"training_log_{instance_name}.csv")
        )

        # 2. Evaluation (A2C + baselines)
        eval_rows = []
        sched_map = {}

        for run in range(30):
            state, _ = env.reset(); done = False
            while not done:
                valid = env.get_valid_actions(); a, _, _ = agent.select_action(state, valid)
                state, _, done, _, _ = env.step(a)
            ms = env.get_makespan()
            eval_rows.append({"algorithm":"a2c","run":run+1,"makespan":ms,
                               "idle_time":round(env.get_idle_time(),2),
                               "utilization":round(env.get_utilization(),4)})

        sched_map["a2c"] = best_schedule

        for algo in ("fifo", "spt", "lpt", "edd"):
            r = run_dispatching_rule(jobs_data, algo)
            for run in range(30):
                eval_rows.append({"algorithm":algo,"run":run+1,"makespan":r["makespan"],
                                   "idle_time":round(r["idle_time"],2),
                                   "utilization":round(r["utilization"],4)})
            sched_map[algo] = r["schedule"]

        for run in range(30):
            r = run_dispatching_rule(jobs_data, "random", seed=run)
            eval_rows.append({"algorithm":"random","run":run+1,"makespan":r["makespan"],
                               "idle_time":round(r["idle_time"],2),
                               "utilization":round(r["utilization"],4)})

        save_evaluation_csv(
            eval_rows,
            os.path.join(RESULTS_DIR, f"evaluation_{instance_name}.csv")
        )

        # 3. Schedule CSV cho mỗi algo
        for algo, sched in sched_map.items():
            if sched:
                save_schedule_csv(
                    sched, algo,
                    os.path.join(RESULTS_DIR, f"schedule_{instance_name}_{algo}.csv")
                )

        _train_status["log"].append(f"✅ Hoàn thành! Best makespan = {best_ms}")
        _train_status["log"].append(f"📄 CSV đã lưu vào results/")

    except Exception as e:
        import traceback
        _train_status["log"].append(f"❌ Lỗi: {str(e)}")
        _train_status["log"].append(traceback.format_exc()[:300])

    finally:
        _train_status["running"] = False


# ════════════════════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🏭 Job-Shop Scheduling — Flask Web App")
    print("  ĐATN 2026 | Bạch Công Quân")
    print("═"*55)
    print("  ▶  Mở trình duyệt: http://localhost:5000")
    print("  ▶  Tắt server    : Ctrl + C")
    print("═"*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)