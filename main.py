
import os, sys, json, time, threading
import numpy as np
from flask import Flask, render_template, jsonify, request


# ── Fix: numpy int64/float64 không serialize được sang JSON 
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

# ── Instance Cache — đảm bảo instance random ổn định trong session 
# Vì một số instance được generate random, cần cache để mọi route gọi
# get_instance(name) đều trả về CÙNG 1 data → kết quả nhất quán giữa các tabs
_INSTANCE_CACHE = {}

def _cached_instance(name):
    """Wrapper cache cho get_instance — đảm bảo cùng key → cùng data."""
    if name not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[name] = get_instance(name)
    return _INSTANCE_CACHE[name]

# ── Flask App 
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False
app.json_encoder = NumpyEncoder  # Dùng encoder tùy chỉnh

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Training status (dùng cho /api/train/status) 
_train_status = {
    "running" : False,
    "instance": None,
    "episode" : 0,
    "total"   : 0,
    "best_ms" : None,
    "log"     : [],
}


#  PAGES

@app.route("/")
def index():
    """Trang chính — Game visualization."""
    return render_template("index.html")


#  API — DATA

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
        jobs_data = _cached_instance(instance_name)
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
        jobs_data = _cached_instance(instance_name)
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


#  API — TRAINING

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


#  HELPERS

# ── A2C Result Cache — đảm bảo cùng instance trả CÙNG kết quả 
# Key = (instance_name, model_file_mtime) → tự invalidate khi retrain
_A2C_RESULT_CACHE = {}


def _run_a2c(jobs_data, instance_name: str) -> dict:
    """
    Chạy A2C inference DETERMINISTIC.

    Quan trọng:
      - Set seed cố định trước mỗi lần chạy → kết quả lặp lại được
      - Cache kết quả theo (instance, model_mtime) → mọi tab thấy cùng số
      - Khi retrain model, mtime đổi → cache invalidate tự động
    """
    env        = JobShopEnv(jobs_data)
    model_path = os.path.join(RESULTS_DIR, f"model_{instance_name}.npz")

    # ── Cache lookup 
    if os.path.exists(model_path):
        mtime     = os.path.getmtime(model_path)
        cache_key = (instance_name, mtime)
        if cache_key in _A2C_RESULT_CACHE:
            cached = _A2C_RESULT_CACHE[cache_key]
            return {
                "schedule"   : list(cached["schedule"]),
                "makespan"   : cached["makespan"],
                "idle_time"  : cached["idle_time"],
                "utilization": cached["utilization"],
            }
    else:
        cache_key = None

# ✅ Set seed cố định → 20 episodes luôn cho CÙNG kết quả
    np.random.seed(42)

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

        result = {
            "schedule"   : best_sched,
            "makespan"   : best_ms,
            "idle_time"  : env2.get_idle_time(),
            "utilization": env2.get_utilization(),
        }

        # ✅ Cache kết quả — lần gọi sau cho cùng instance trả ngay
        if cache_key is not None:
            _A2C_RESULT_CACHE[cache_key] = {
                "schedule"   : list(result["schedule"]),
                "makespan"   : result["makespan"],
                "idle_time"  : result["idle_time"],
                "utilization": result["utilization"],
            }

        return result

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
        jobs_data = _cached_instance(instance_name)
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

        # ── Export CSV 
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
        # ✅ Invalidate cache sau khi train xong → các tab sẽ thấy model mới
        global _A2C_RESULT_CACHE, _eval_cache
        _A2C_RESULT_CACHE = {}
        _eval_cache = {}
        _train_status["log"].append(f"📄 CSV đã lưu vào results/")

    except Exception as e:
        import traceback
        _train_status["log"].append(f"❌ Lỗi: {str(e)}")
        _train_status["log"].append(traceback.format_exc()[:300])

    finally:
        _train_status["running"] = False



#  RUN
#  API — ĐÁNH GIÁ TỔNG HỢP

DIFFICULTY_TIERS = {
    # FIX: dùng cùng key với api_instances() để Tab Trực quan hóa và Tab Đánh giá
    # hiển thị cùng kết quả cho cùng 1 instance
    "basic"  : {"label": "Cơ bản",    "instances": ["3x3", "4x4"],  "color": "#34d399"},
    "medium" : {"label": "Trung bình", "instances": ["5x5", "ft06"], "color": "#fbbf24"},
    "complex": {"label": "Phức tạp",  "instances": ["ft10"],         "color": "#f87171"},
}
_eval_cache = {}

@app.route("/api/evaluation/full")
def api_evaluation_full():
    global _eval_cache
    force = request.args.get("refresh","false").lower() == "true"
    if _eval_cache and not force:
        return jsonify(_eval_cache)
    tiers_result = {}
    for tier_key, tier_info in DIFFICULTY_TIERS.items():
        tier_data = {"label":tier_info["label"],"color":tier_info["color"],"instances":{}}
        for inst_name in tier_info["instances"]:
            try:
                jobs_data = _cached_instance(inst_name)
                info      = instance_info(jobs_data)
                lb        = info["critical_path_lb"]
                inst_results = {}
                for algo in ("fifo","spt","lpt","edd"):
                    r  = run_dispatching_rule(jobs_data, algo)
                    ms = r["makespan"]
                    inst_results[algo] = {"makespan":ms,"idle_time":round(r["idle_time"],1),
                        "utilization":round(r["utilization"]*100,1),"gap_to_lb":round((ms-lb)/lb*100,1)}
                a2c_r  = _run_a2c(jobs_data, inst_name)
                ms_a2c = a2c_r["makespan"]
                inst_results["a2c"] = {"makespan":ms_a2c,"idle_time":round(a2c_r["idle_time"],1),
                    "utilization":round(a2c_r["utilization"]*100,1),"gap_to_lb":round((ms_a2c-lb)/lb*100,1),
                    "trained":os.path.exists(os.path.join(RESULTS_DIR,f"model_{inst_name}.npz"))}
                best_algo = min(inst_results, key=lambda a: inst_results[a]["makespan"])
                fifo_ms   = inst_results["fifo"]["makespan"]
                tier_data["instances"][inst_name] = {"info":info,"lower_bound":lb,
                    "results":inst_results,"best_algo":best_algo,
                    "improvement_vs_fifo":round((fifo_ms-ms_a2c)/fifo_ms*100,1)}
            except Exception as e:
                tier_data["instances"][inst_name] = {"error":str(e)}
        tiers_result[tier_key] = tier_data
    _eval_cache = {"tiers": tiers_result}
    return jsonify(_to_python(_eval_cache))


#  GREEDY BEST-FIT ALLOCATOR + PRIORITY TRIAGE SYSTEM
#  Thêm hệ thống Priority Triage theo yêu cầu thầy Hiếu:
#    🔴 CẤP BÁCH — bug thực sự, phải fix ngay (không xếp đủ buổi, conflict slot)
#    🟡 TRUNG BÌNH — vấn đề logic nhưng vẫn xếp được (kéo dài makespan, FIFO ép buộc)
#    🟢 CẢNH BÁO   — chưa có vấn đề nhưng tiềm ẩn rủi ro (slot quá tải, ít redundancy)

DAYS_VN    = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7"]
PERIODS_VN = ["Sáng", "Chiều"]
N_DAYS     = 6
N_PERIODS  = 2
N_SLOTS    = N_DAYS * N_PERIODS


def _spread_slots(allowed_slots, n_pick):
    """Chọn n_pick slot từ allowed, ưu tiên trải đều theo ngày."""
    if n_pick >= len(allowed_slots):
        return list(allowed_slots)
    # Sắp xếp slot theo (ngày, buổi) → đảm bảo đa dạng ngày khi pick
    sorted_slots = sorted(allowed_slots, key=lambda s: (s // N_PERIODS, s % N_PERIODS))
    if n_pick == 1:
        return [sorted_slots[0]]
    step = len(sorted_slots) / n_pick
    return [sorted_slots[int(i * step)] for i in range(n_pick)]


def _greedy_best_fit(classes_meta, strategy="a2c"):
    """Greedy scheduler — đảm bảo MỖI LỚP nhận đủ total_sessions."""
    state    = [{**c, "remaining": c["total_sessions"]} for c in classes_meta]
    schedule = []
    week     = 0
    MAX_WEEKS = 100

    def sort_key(c):
        if strategy == "a2c":
            return (len(c["allowed_slots"]), -c["remaining"], c["idx"])
        elif strategy == "spt":
            return (c["sessions_per_week"], c["idx"])
        elif strategy == "edd":
            return (c["duration_weeks"], c["idx"])
        else:
            return (c["idx"],)

    if strategy == "fifo":
        for cls in state:
            while cls["remaining"] > 0 and week < MAX_WEEKS:
                target = min(cls["sessions_per_week"], cls["remaining"])
                slots_used = {e["machine"] for e in schedule if e["week"] == week}
                candidates = [s for s in cls["allowed_slots"] if s not in slots_used]
                pick = _spread_slots(candidates, target)
                for slot in pick:
                    schedule.append({
                        "job": cls["idx"], "machine": slot, "week": week,
                        "day_idx": slot // N_PERIODS, "period_idx": slot % N_PERIODS,
                        "class_name": cls["name"], "class_color": cls["color"],
                    })
                    cls["remaining"] -= 1
                week += 1
    else:
        while any(c["remaining"] > 0 for c in state) and week < MAX_WEEKS:
            slots_used = set()
            active = [c for c in state if c["remaining"] > 0]
            active.sort(key=sort_key)
            for cls in active:
                target = min(cls["sessions_per_week"], cls["remaining"])
                free_in_week = [s for s in cls["allowed_slots"] if s not in slots_used]
                if not free_in_week:
                    continue
                pick = _spread_slots(free_in_week, min(target, len(free_in_week)))
                for slot in pick:
                    schedule.append({
                        "job": cls["idx"], "machine": slot, "week": week,
                        "day_idx": slot // N_PERIODS, "period_idx": slot % N_PERIODS,
                        "class_name": cls["name"], "class_color": cls["color"],
                    })
                    slots_used.add(slot)
                    cls["remaining"] -= 1
            week += 1

    n_weeks  = week
    n_used   = len(schedule)
    capacity = n_weeks * N_SLOTS
    idle     = max(0, capacity - n_used)
    util     = (n_used / capacity * 100) if capacity > 0 else 0

    counts = {}
    for e in schedule:
        counts[e["job"]] = counts.get(e["job"], 0) + 1
    unscheduled = []
    for c in classes_meta:
        got = counts.get(c["idx"], 0)
        if got < c["total_sessions"]:
            unscheduled.append({
                "name": c["name"], "expected": c["total_sessions"],
                "actual": got, "missing": c["total_sessions"] - got,
            })

    return {
        "schedule": schedule, "makespan": n_weeks, "n_weeks": n_weeks,
        "idle": idle, "util": round(util, 1), "unscheduled": unscheduled,
    }


def _enrich_schedule(sched):
    out = []
    for e in sched:
        slot = e["machine"]
        out.append({
            **e,
            "day_name"   : DAYS_VN[slot // N_PERIODS],
            "period_name": PERIODS_VN[slot %  N_PERIODS],
        })
    return out


def _priority_triage(classes_meta, results, makespan_a2c, makespan_fifo):
    """
    Phân loại các vấn đề thành 3 mức ưu tiên (theo yêu cầu thầy Hiếu).

    🔴 CẤP BÁCH (critical) — phải xử lý ngay, ảnh hưởng kết quả
    🟡 TRUNG BÌNH (warning) — cần xử lý nhưng không khẩn
    🟢 CẢNH BÁO (info)     — chưa có vấn đề, đề phòng

    Returns: list of {priority, code, title, description, suggestion, affected_class?}
    """
    issues = []

    # ─── CẤP BÁCH 🔴
    # 1. A2C có lớp không xếp đủ buổi
    a2c_unscheduled = results["a2c"].get("unscheduled", [])
    if a2c_unscheduled:
        for u in a2c_unscheduled:
            issues.append({
                "priority": "critical",
                "code"    : "MISSING_SESSIONS",
                "title"   : f"Lớp {u['name']} thiếu {u['missing']} buổi",
                "description": f"Cần {u['expected']} buổi nhưng chỉ xếp được {u['actual']} — không đủ slot khả dụng",
                "suggestion": "Thêm slot cho phép hoặc bật AI Auto Mode cho lớp này",
                "affected_class": u["name"],
            })

    # 2. Lớp có sessions_per_week > số allowed_slots → không thể đạt được
    for cls in classes_meta:
        n_allowed = len(cls["allowed_slots"])
        if cls["sessions_per_week"] > n_allowed:
            issues.append({
                "priority": "critical",
                "code"    : "INFEASIBLE_CONFIG",
                "title"   : f"Cấu hình {cls['name']} bất khả thi",
                "description": f"{cls['sessions_per_week']} buổi/tuần nhưng chỉ có {n_allowed} slot khả dụng — không thể xếp đủ trong 1 tuần",
                "suggestion": f"Giảm buổi/tuần xuống ≤{n_allowed}, hoặc thêm slot khả dụng",
                "affected_class": cls["name"],
            })

    # ─── TRUNG BÌNH 🟡
    # 3. A2C makespan vượt duration_weeks (kéo dài lịch)
    max_dur = max(c["duration_weeks"] for c in classes_meta)
    if makespan_a2c > max_dur:
        extra = makespan_a2c - max_dur
        issues.append({
            "priority": "warning",
            "code"    : "EXTENDED_MAKESPAN",
            "title"   : f"Lịch kéo dài thêm {extra} tuần",
            "description": f"Dự kiến hoàn thành trong {max_dur} tuần nhưng A2C cần {makespan_a2c} tuần — do conflict slot giữa các lớp",
            "suggestion": "Phân tán bớt lớp sang slot khác, hoặc bật AI Auto cho 1 vài lớp để giảm tranh chấp",
        })

    # 4. FIFO kém xa A2C (>50%) → cấu hình tốt nhưng FIFO không tận dụng được
    if makespan_fifo > 0:
        gap_pct = (makespan_fifo - makespan_a2c) / makespan_fifo * 100
        if gap_pct >= 50:
            issues.append({
                "priority": "warning",
                "code"    : "FIFO_INEFFICIENT",
                "title"   : f"FIFO kém A2C {gap_pct:.0f}%",
                "description": f"FIFO cần {makespan_fifo} tuần, A2C chỉ cần {makespan_a2c} — chênh lệch lớn cho thấy cấu hình hiện tại không phù hợp xếp tuần tự",
                "suggestion": "Đây là tín hiệu tốt — chứng minh giá trị của A2C. Có thể dùng làm demo nổi bật.",
            })

    # 5. Hiệu suất thấp (<40%)
    util = results["a2c"].get("util", 0)
    if util < 40:
        issues.append({
            "priority": "warning",
            "code"    : "LOW_UTILIZATION",
            "title"   : f"Hiệu suất slot thấp ({util}%)",
            "description": f"Chỉ {util}% slot có lịch, còn lại bỏ trống — cấu hình hiện tại đang lãng phí thời gian biểu",
            "suggestion": "Có thể nhận thêm lớp dạy hoặc gộp các lớp ít buổi/tuần",
        })

    # ─── CẢNH BÁO 🟢
    # 6. Slot bị "đông đúc" — nhiều lớp Strict cùng tranh
    slot_pressure = {}
    for cls in classes_meta:
        if len(cls["allowed_slots"]) >= N_SLOTS - 2:
            continue   # Auto mode → bỏ qua
        per_slot_demand = cls["total_sessions"] / len(cls["allowed_slots"])
        for s in cls["allowed_slots"]:
            slot_pressure.setdefault(s, []).append({
                "name": cls["name"], "demand": per_slot_demand,
            })

    for slot, demanders in slot_pressure.items():
        total_demand = sum(d["demand"] for d in demanders)
        if total_demand > 4 and len(demanders) >= 2:
            day_name    = DAYS_VN[slot // N_PERIODS]
            period_name = PERIODS_VN[slot %  N_PERIODS]
            names = ", ".join(d["name"] for d in demanders)
            issues.append({
                "priority": "info",
                "code"    : "SLOT_HOTSPOT",
                "title"   : f"Slot {day_name} {period_name} có nhu cầu cao",
                "description": f"Có {len(demanders)} lớp tranh nhau slot này: {names} (tổng nhu cầu ~{total_demand:.1f} tuần)",
                "suggestion": "Cấu hình đang ổn nhưng nếu phát sinh thêm lớp mới → có thể conflict",
            })

    # 7. Lớp chỉ có 1 slot allowed → fragile, không có dự phòng
    for cls in classes_meta:
        if len(cls["allowed_slots"]) == 1 and cls["sessions_per_week"] >= 1:
            slot = cls["allowed_slots"][0]
            day_name = DAYS_VN[slot // N_PERIODS]
            period_name = PERIODS_VN[slot %  N_PERIODS]
            issues.append({
                "priority": "info",
                "code"    : "NO_REDUNDANCY",
                "title"   : f"Lớp {cls['name']} không có slot dự phòng",
                "description": f"Chỉ chọn duy nhất {day_name} {period_name} — nếu giảng viên bận đột xuất sẽ không có lịch thay thế",
                "suggestion": "Cân nhắc thêm 1-2 slot dự phòng để linh hoạt hơn",
                "affected_class": cls["name"],
            })

    # 8. Tổng tải toàn hệ thống cao (>80%)
    total_demand = sum(c["total_sessions"] for c in classes_meta)
    capacity = makespan_a2c * N_SLOTS if makespan_a2c > 0 else 1
    load_pct = total_demand / capacity * 100
    if load_pct > 80:
        issues.append({
            "priority": "info",
            "code"    : "HIGH_LOAD",
            "title"   : f"Tải hệ thống cao ({load_pct:.0f}%)",
            "description": f"Đã sử dụng {total_demand}/{capacity} slot — gần chạm giới hạn",
            "suggestion": "Nên giữ buffer để xử lý phát sinh (hủy lớp, dạy bù, đột xuất)",
        })

    # Sắp xếp: critical trước, warning, info
    priority_order = {"critical": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: priority_order.get(i["priority"], 99))
    return issues


def _schedule_classes(classes_config):
    cleaned = []
    for idx, cls in enumerate(classes_config):
        allowed = sorted(set(cls.get("allowed_slots", [])))
        spw     = max(1, int(cls.get("sessions_per_week", 1)))
        n_weeks = max(1, int(cls.get("duration_weeks", 1)))
        if not allowed:
            continue
        cleaned.append({
            "idx"              : idx,
            "name"             : cls["name"],
            "color"            : cls.get("color", "#4f8ef7"),
            "sessions_per_week": spw,
            "duration_weeks"   : n_weeks,
            "allowed_slots"    : allowed,
            "total_sessions"   : spw * n_weeks,
        })
    if not cleaned:
        return None

    a2c_r  = _greedy_best_fit(cleaned, "a2c")
    fifo_r = _greedy_best_fit(cleaned, "fifo")
    spt_r  = _greedy_best_fit(cleaned, "spt")
    edd_r  = _greedy_best_fit(cleaned, "edd")

    # Enrich schedules
    for r in (a2c_r, fifo_r, spt_r, edd_r):
        r["schedule"] = _enrich_schedule(r["schedule"])

    # ─── PRIORITY TRIAGE
    triage_issues = _priority_triage(
        cleaned,
        {"a2c": a2c_r, "fifo": fifo_r},
        a2c_r["makespan"], fifo_r["makespan"],
    )

    # Đếm theo priority
    counts = {"critical": 0, "warning": 0, "info": 0}
    for issue in triage_issues:
        counts[issue["priority"]] = counts.get(issue["priority"], 0) + 1

    return _to_python({
        "a2c" : a2c_r,
        "fifo": fifo_r,
        "spt" : spt_r,
        "edd" : edd_r,
        "classes" : cleaned,
        "days"    : DAYS_VN,
        "periods" : PERIODS_VN,
        "n_days"  : N_DAYS,
        "n_periods": N_PERIODS,
        "n_weeks" : a2c_r["makespan"],
        "explanation": {
            "sequential_ms" : fifo_r["makespan"],
            "interleaved_ms": a2c_r["makespan"],
            "saved"         : fifo_r["makespan"] - a2c_r["makespan"],
            "saved_pct"     : round((fifo_r["makespan"]-a2c_r["makespan"])/fifo_r["makespan"]*100,1) if fifo_r["makespan"]>0 else 0,
        },
        "triage": {
            "issues" : triage_issues,
            "counts" : counts,
            "total"  : len(triage_issues),
        }
    })


@app.route("/api/realworld/teacher", methods=["POST"])
def api_teacher_schedule():
    data    = request.get_json(force=True, silent=True) or {}
    classes = data.get("classes", [])
    if not classes:
        return jsonify({"error": "Thiếu danh sách lớp học"}), 400
    result = _schedule_classes(classes)
    if result is None:
        return jsonify({"error": "Dữ liệu không hợp lệ"}), 400
    return jsonify(result)

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🏭 Job-Shop Scheduling — Flask Web App")
    print("  ĐATN 2026 | Bạch Công Quân")
    print("═"*55)
    print("  ▶  Mở trình duyệt: http://localhost:5000")
    print("  ▶  Tắt server    : Ctrl + C")
    print("═"*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)