from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from comm_ai.datasets.signals_dataset import SignalsDataset, generate_and_save
from comm_ai.experiments.evaluate import evaluate_decoders
from comm_ai.utils.io import load_yaml, save_yaml
from comm_ai.utils.plotting import save_metric_plot
from comm_ai.utils.seed import set_seed


def run(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    set_seed(cfg["experiment"]["seed"])

    run_name = cfg["experiment"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["paths"]["outputs_root"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "signals.npz"
    ds = generate_and_save(cfg, data_path)

    # reload to ensure reproducibility path works
    ds = SignalsDataset.load(data_path)
    df = evaluate_decoders(cfg, ds)
    df.to_csv(out_dir / "results.csv", index=False)

    save_metric_plot(df, "ber", out_dir / "ber_plot.png")
    save_metric_plot(df, "fer", out_dir / "fer_plot.png")
    save_metric_plot(df.rename(columns={"decode_time_s": "timing"}), "timing", out_dir / "timing_plot.png")

    save_yaml(cfg, out_dir / "config_used.yaml")
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Experiment summary\n\n")
        f.write(df.to_markdown(index=False))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    out = run(args.config)
    print(f"Saved run to: {out}")


if __name__ == "__main__":
    main()
