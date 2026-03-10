from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plot(df: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")
    plt.figure(figsize=(6, 4))
    for decoder, g in df.groupby("decoder"):
        plt.plot(g["snr_db"], g[metric], marker="o", label=decoder)
    plt.xlabel("SNR [dB]")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
