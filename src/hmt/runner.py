"""
HMT CLI entry point and evaluation loop.

Supports two modes (controlled by cfg.exp_type):
  evaluate     — run over a full dataset split and log per-task-type success rates
  collect_llm  — smoke test on a single task
"""

import datetime
import json
import logging
import os
import random
import sys
from typing import Any

from alfred.utils import dotdict, save_vis_log

from .integration import AlfredReactreeWithHMT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(cfg) -> str:
    """Set up root logging once and write HMT logs to a dedicated file."""
    log_dir = getattr(cfg, "out_dir", None)
    if not log_dir or (isinstance(log_dir, str) and "${" in log_dir):
        log_dir = getattr(getattr(cfg, "dataset", object()), "collect_dir", None)
    if not log_dir:
        log_dir = os.path.join("output", "hmt_collect")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "hmt.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler_exists = False
    stream_handler_exists = False
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == os.path.abspath(log_path)
        ):
            handler.setFormatter(formatter)
            file_handler_exists = True
        elif isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)
            stream_handler_exists = True

    if not file_handler_exists:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    if not stream_handler_exists:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)

    logger.info("Logging initialized. HMT log file: %s", log_path)
    return log_path


# ---------------------------------------------------------------------------
# Main implementation
# ---------------------------------------------------------------------------

def _main_impl(cfg) -> None:
    import pprint
    import time
    from collections import defaultdict

    from tqdm import tqdm
    from alfred.alfred_env import ThorConnector
    from alfred.alfred_llm_agent import AlfredLlmAgent
    from alfred.data.preprocess import Dataset

    _configure_logging(cfg)

    try:
        from omegaconf import OmegaConf  # type: ignore
        logger.info("Loaded config:\n%s", OmegaConf.to_yaml(cfg))
    except Exception:
        logger.info("Loaded config object: %s", cfg)

    splits_path = getattr(
        getattr(cfg, "alfred", object()), "splits", "alfred/data/splits/oct21.json"
    )
    args_dict: dict = {
        "data": "alfred/data/json_2.1.0",
        "pframe": 300,
        "fast_epoch": False,
        "use_templated_goals": False,
        "dout": "exp/model",
        "pp_folder": "pp",
        "reward_config": "alfred/models/config/rewards.json",
        "max_steps": 100,
    }

    if not os.path.exists(splits_path):
        logger.error("Splits file not found: %s", splits_path)
        return

    with open(splits_path, "r") as fh:
        splits = json.load(fh)

    pprint.pprint({k: len(v) for k, v in splits.items()})
    logger.info("Loaded dataset splits from %s", splits_path)

    exp_type = getattr(cfg, "exp_type", "collect_llm")

    # ------------------------------------------------------------------
    # EVALUATE mode
    # ------------------------------------------------------------------
    if exp_type == "evaluate":
        eval_set = getattr(getattr(cfg, "dataset", object()), "eval_set", "valid_seen")
        if eval_set not in splits:
            logger.error(
                "eval_set '%s' not found in splits. Available: %s", eval_set, list(splits.keys())
            )
            return

        files = list(splits[eval_set])

        eval_portion = int(
            getattr(getattr(cfg, "alfred", object()), "eval_portion_in_percent", 100)
        )
        if eval_portion < 100:
            seed = int(
                getattr(getattr(cfg, "alfred", object()), "random_seed_for_eval_subset", 1)
            )
            random.seed(seed)
            n_sample = max(1, int(len(files) * eval_portion / 100))
            files = random.sample(files, n_sample)
            logger.info(
                "Using %d/%d tasks (%d%% subset, seed=%d)",
                n_sample, len(splits[eval_set]), eval_portion, seed,
            )

        number_of_dirs = (
            len(list(os.listdir(args_dict["data"]))) if os.path.exists(args_dict["data"]) else 0
        )
        if number_of_dirs < 50:
            logger.info("Preprocessing dataset (one-time)...")
            dataset = Dataset(dotdict(args_dict), None)
            dataset.preprocess_splits(splits)

        env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)
        llm_agent = AlfredLlmAgent(cfg)
        planner = AlfredReactreeWithHMT(cfg, llm_agent, env, use_hmt=True)

        results: list = []
        start = time.time()
        save_path = cfg.out_dir
        total_tasks = len(files)
        type_tried: Any = defaultdict(int)
        type_ok: Any = defaultdict(int)

        def _print_summary(results_so_far, elapsed_sec):
            n_done = len(results_so_far)
            n_ok = sum(1 for r in results_so_far if r["success"])
            n_fail = n_done - n_ok
            sr = n_ok / n_done * 100 if n_done else 0.0
            elapsed = str(datetime.timedelta(seconds=int(elapsed_sec)))
            sep = "+" + "-" * 8 + "+" + "-" * 30 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+"
            hdr = "| {:^6} | {:^28} | {:^6} | {:^6} | {:^6} | {:^6} |".format(
                "Done", "Task Type", "Tried", "OK", "Fail", "SR%"
            )
            logger.info(sep)
            logger.info(
                "| {:^74} |".format(
                    f"HMT Progress  [{eval_set}]  {n_done}/{total_tasks}  |  SR {sr:.1f}%  |  {elapsed}"
                )
            )
            logger.info(sep)
            logger.info(hdr)
            logger.info(sep)
            for ttype in sorted(type_tried.keys()):
                tried = type_tried[ttype]
                ok = type_ok[ttype]
                fail = tried - ok
                t_sr = ok / tried * 100 if tried else 0.0
                logger.info(
                    "| {:^6} | {:<28} | {:^6} | {:^6} | {:^6} | {:^6.1f} |".format(
                        "", ttype[:28], tried, ok, fail, t_sr
                    )
                )
            logger.info(sep)
            logger.info(
                "| {:^6} | {:<28} | {:^6} | {:^6} | {:^6} | {:^6.1f} |".format(
                    n_done, "TOTAL", n_done, n_ok, n_fail, sr
                )
            )
            logger.info(sep)

        for task_d in tqdm(files):
            terminate_info = planner.run(task_d, args_dict, logger)
            result = {
                "task": task_d["task"],
                "repeat_idx": task_d["repeat_idx"],
                "success": terminate_info["success"],
                "nl_inst": terminate_info["nl_inst"],
            }

            task_name = task_d["task"]
            task_type = task_name.split("/")[0]
            trial_num = task_name.split("/")[1]
            repeat_idx = task_d["repeat_idx"]
            vis_prefix = "success" if result["success"] else "failure"
            vis_log_name = f"{vis_prefix}_{task_type}_{trial_num}_ann_{repeat_idx}"
            save_vis_log(cfg, env.vis_log, vis_log_name, terminate_info["nl_inst"])
            results.append(result)

            type_tried[task_type] += 1
            if result["success"]:
                type_ok[task_type] += 1
            _print_summary(results, time.time() - start)

        n = len(results)
        n_success = sum(1 for r in results if r["success"])
        logger.info("=== HMT EVALUATION COMPLETE ===")
        _print_summary(results, time.time() - start)
        logger.info("Elapsed: %s", str(datetime.timedelta(seconds=int(time.time() - start))))
        try:
            from omegaconf import OmegaConf  # type: ignore
            logger.info(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

        results_path = os.path.join(save_path, f"hmt_results_{eval_set}.json")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "eval_set": eval_set,
                    "n": n,
                    "n_success": n_success,
                    "success_rate": n_success / n if n else 0.0,
                    "results": results,
                },
                fh,
                indent=2,
            )
        logger.info("Results saved to %s", results_path)

    # ------------------------------------------------------------------
    # COLLECT_LLM mode (single-task smoke test)
    # ------------------------------------------------------------------
    else:
        train_key = getattr(cfg.dataset, "train_set", "train")
        tasks = splits.get(train_key, [])
        if not tasks:
            logger.error("No tasks found for split key: %s", train_key)
            return

        task_d = tasks[1]
        logger.info("Running one-task HMT test | exp_type=%s", exp_type)
        logger.info("Selected task: %s", task_d)

        env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)
        llm_agent = AlfredLlmAgent(cfg)
        planner = AlfredReactreeWithHMT(cfg, llm_agent, env, use_hmt=True)

        result = planner.collect_llm(task_d, args_dict)
        logger.info("HMT result: %s", result)



