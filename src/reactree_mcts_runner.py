"""
reactree_mcts_runner.py — CLI entry point for ReactreeMCTS on ALFWorld.

Two modes (cfg.exp_type):
  evaluate    — run over a full dataset split, log per-task-type success rates
  collect_llm — smoke test on a single task

Usage:
  python src/reactree_mcts_runner.py --config-name=alfworld_reactree_mcts
  python src/reactree_mcts_runner.py --config-name=alfworld_reactree_mcts exp_type=collect_llm
"""

import datetime
import glob
import json
import logging
import os
import random
import sys
from collections import defaultdict
from os.path import join as pjoin

# Keep src/ on sys.path even after Hydra changes the working directory.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

logger = logging.getLogger(__name__)

try:
    import hydra
except ImportError:
    hydra = None


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(cfg) -> None:
    log_dir = getattr(cfg, "out_dir", None)
    if not log_dir or (isinstance(log_dir, str) and "${" in log_dir):
        log_dir = "output/reactree_mcts"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "reactree_mcts.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
        for h in root.handlers
    ):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    logger.info("Log file: %s", log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Task discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_tasks(data_root: str, split: str, portion: int = 100, seed: int = 1) -> list:
    """
    Find ALFWorld task directories for the given split.
    Returns list of {"task_file": path_to_dir, "task_type": str}.
    """
    pattern = pjoin(data_root, "**", split, "**", "initial_state.pddl")
    pddls = glob.glob(pattern, recursive=True)
    if not pddls:
        pattern = pjoin(data_root, "**", "initial_state.pddl")
        pddls = glob.glob(pattern, recursive=True)

    tasks = []
    for pddl in pddls:
        task_dir = os.path.dirname(pddl)
        if not os.path.exists(os.path.join(task_dir, "traj_data.json")):
            continue
        task_type = os.path.basename(os.path.dirname(task_dir))
        tasks.append({"task_file": task_dir, "task_type": task_type})

    if portion < 100:
        random.seed(seed)
        n = max(1, int(len(tasks) * portion / 100))
        tasks = random.sample(tasks, n)
        logger.info("Subset: %d/%d tasks (seed=%d)", n, len(pddls), seed)

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _main_impl(cfg) -> None:
    import time
    from tqdm import tqdm

    from reactree_mcts import AlfWorldLlmAgent, run_task
    from mcts_src.environment.alfworld_env import AlfWorldEnv

    _configure_logging(cfg)

    try:
        from omegaconf import OmegaConf
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    except Exception:
        pass

    mcts_cfg = getattr(cfg, "reactree_mcts", None)
    mcts_budget = int(getattr(mcts_cfg, "mcts_budget", 5))
    mcts_max_depth = int(getattr(mcts_cfg, "mcts_max_depth", 4))

    alfworld_cfg = getattr(cfg, "alfworld", None)
    alfworld_config_path = getattr(alfworld_cfg, "config_path", "config/base_config.yaml")

    data_root = getattr(alfworld_cfg, "data_root", "")
    if not data_root:
        try:
            from alfworld.info import ALFWORLD_DATA  # type: ignore[import-not-found]
            data_root = ALFWORLD_DATA
        except ImportError:
            raise RuntimeError(
                "alfworld not installed and alfworld.data_root not set in config. "
                "Set alfworld.data_root in your config or install alfworld."
            )

    exp_type = getattr(cfg, "exp_type", "collect_llm")

    llm_agent = AlfWorldLlmAgent(cfg)

    # ------------------------------------------------------------------
    # evaluate mode
    # ------------------------------------------------------------------
    if exp_type == "evaluate":
        eval_set = getattr(getattr(cfg, "dataset", object()), "eval_set", "valid_seen")
        eval_portion = int(getattr(alfworld_cfg, "eval_portion_in_percent", 100))
        seed = int(getattr(alfworld_cfg, "random_seed_for_eval_subset", 1))

        tasks = _find_tasks(data_root, eval_set, eval_portion, seed)
        if not tasks:
            raise RuntimeError(f"No tasks found under '{data_root}' for split '{eval_set}'")

        total = len(tasks)
        logger.info("Evaluating %d tasks from '%s'", total, eval_set)
        results = []
        type_tried: dict = defaultdict(int)
        type_ok: dict = defaultdict(int)
        start = time.time()

        def _print_summary(res, elapsed_sec):
            n_done = len(res)
            n_ok = sum(1 for r in res if r["success"])
            sr = n_ok / n_done * 100 if n_done else 0.0
            elapsed = str(datetime.timedelta(seconds=int(elapsed_sec)))
            sep = "+" + "-" * 8 + "+" + "-" * 30 + "+" + "-" * 8 + "+" + "-" * 8 + "+"
            logger.info(sep)
            logger.info(
                "| %-55s |",
                f"ReactreeMCTS [{eval_set}]  {n_done}/{total}  SR {sr:.1f}%  {elapsed}",
            )
            logger.info(sep)
            logger.info("| %-28s | %6s | %6s | %6s |", "Task Type", "Tried", "OK", "SR%")
            logger.info(sep)
            for tt in sorted(type_tried):
                tried = type_tried[tt]
                ok = type_ok[tt]
                logger.info(
                    "| %-28s | %6d | %6d | %6.1f |",
                    tt[:28], tried, ok, ok / tried * 100 if tried else 0,
                )
            logger.info(sep)
            logger.info("| %-28s | %6d | %6d | %6.1f |", "TOTAL", n_done, n_ok, sr)
            logger.info(sep)

        for task_d in tqdm(tasks, desc="ReactreeMCTS"):
            env = AlfWorldEnv(alfworld_config_path, task_file=task_d["task_file"])
            result = run_task(
                cfg, llm_agent, env, task_d["task_file"], mcts_budget, mcts_max_depth
            )
            logger.info(
                "task=%s | success=%s | steps=%d | task_type=%s",
                task_d["task_file"], result["success"], result["steps"], task_d["task_type"],
            )
            results.append({**result, "task_type": task_d["task_type"]})
            type_tried[task_d["task_type"]] += 1
            if result["success"]:
                type_ok[task_d["task_type"]] += 1
            _print_summary(results, time.time() - start)

        n_ok = sum(1 for r in results if r["success"])
        results_path = os.path.join(cfg.out_dir, f"reactree_mcts_results_{eval_set}.json")
        with open(results_path, "w") as fh:
            json.dump({
                "eval_set": eval_set,
                "n": len(results),
                "n_success": n_ok,
                "success_rate": n_ok / len(results) if results else 0.0,
                "mcts_budget": mcts_budget,
                "results": results,
            }, fh, indent=2)
        logger.info("Results saved to %s", results_path)

    # ------------------------------------------------------------------
    # collect_llm / smoke-test mode
    # ------------------------------------------------------------------
    else:
        tasks = _find_tasks(data_root, "train")
        if not tasks:
            raise RuntimeError(f"No tasks found under '{data_root}'")

        task_d = tasks[1]
        logger.info("Smoke test | task_file=%s", task_d["task_file"])
        env = AlfWorldEnv(alfworld_config_path, task_file=task_d["task_file"])
        result = run_task(cfg, llm_agent, env, task_d["task_file"], mcts_budget, mcts_max_depth)
        logger.info("Result: %s", result)


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

if hydra is not None:
    @hydra.main(version_base=None, config_path="../conf", config_name="alfworld_reactree_mcts")
    def main(cfg) -> None:
        _main_impl(cfg)
else:
    def main(cfg=None):
        logging.basicConfig(level=logging.INFO)
        logger.error("Hydra not installed. Run: pip install hydra-core")


if __name__ == "__main__":
    main()
