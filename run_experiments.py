"""Plan or execute the CIFAR-10 Early-Bird and t_s experiment suite."""
import argparse
import json
from pathlib import Path
from cifar_experiment import run_full
from utils.config import load_config, require_config


def build_plan(config_path):
    cfg = load_config(config_path)
    require_config(cfg, ["project.seed", "experiment.epochs", "experiment.dataset",
        "experiment.model.factory", "data.loader_factory", "pruning.selected_ratio",
        "earlybird.epsilon", "earlybird.mask_window", "ts.relative_to_tm.before_offset",
        "ts.relative_to_tm.after_offset", "ts.fixed_epoch", "ts.information_modes"])
    tm = "detected from Early-Bird search run"
    return {"status": "ready_to_execute", "dataset": cfg["experiment"]["dataset"],
        "model": "CIFAR VGG16-BN", "epochs": cfg["experiment"]["epochs"],
        "seed": cfg["project"]["seed"], "ratio": cfg["pruning"]["selected_ratio"],
        "detection": {"epsilon": cfg["earlybird"]["epsilon"],
                      "mask_window": cfg["earlybird"]["mask_window"]},
        "ts": tm, "information_modes": cfg["ts"]["information_modes"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ts_sweep.yaml")
    parser.add_argument("--execute", action="store_true", help="run full training and t_s comparisons")
    parser.add_argument("--resume-baseline", help="resume comparisons from a completed baseline run folder")
    parser.add_argument("--output", default="experiment_plan.json")
    args = parser.parse_args()
    plan = build_plan(args.config)
    if not args.execute:
        Path(args.output).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return
    results = run_full(load_config(args.config), resume_baseline=args.resume_baseline)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
