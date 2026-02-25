import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s[:120] if len(s) > 120 else s


def _read_csv_matrix(path: str) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.to_numpy(dtype=np.float64)


def _load_variable_names(data_csv_path: str, drop_cols: int) -> list[str]:
    df = pd.read_csv(data_csv_path, nrows=1)
    cols = list(df.columns[drop_cols:])
    return [str(c) for c in cols]


def _mean_std_over_time(weights: np.ndarray) -> float:
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D weights, got shape {weights.shape}")
    per_var_std = np.std(weights, axis=0, ddof=0)
    return float(np.mean(per_var_std))


def _global_importance(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(alpha), axis=0) + np.mean(np.abs(beta), axis=0)


def _plot_lines(
    out_path: str,
    title: str,
    y_label: str,
    base_series: np.ndarray,
    att_series: np.ndarray,
    base_label: str,
    att_label: str,
) -> None:
    t = np.arange(len(base_series))
    plt.figure(figsize=(8, 4))
    plt.plot(t, base_series, linewidth=2.0, label=base_label)
    plt.plot(t, att_series, linewidth=2.0, label=att_label)
    plt.xlabel("Time Step")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _grouped_boxplot(
    out_path: str,
    title: str,
    y_label: str,
    var_names: list[str],
    base_weights: np.ndarray,
    att_weights: np.ndarray,
    base_label: str,
    att_label: str,
) -> None:
    n_vars = len(var_names)
    fig_w = max(14, int(n_vars * 0.9))
    plt.figure(figsize=(fig_w, 6))

    base_color = "#1f77b4"
    att_color = "#ff7f0e"

    positions = []
    labels_pos = []
    box_data = []
    colors = []

    for i in range(n_vars):
        p1 = i * 3 + 1
        p2 = i * 3 + 2
        positions.extend([p1, p2])
        labels_pos.append((p1 + p2) / 2)
        box_data.extend([base_weights[:, i], att_weights[:, i]])
        colors.extend([base_color, att_color])

    bp = plt.boxplot(
        box_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    plt.xticks(labels_pos, var_names, rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(
        handles=[Patch(facecolor=base_color, edgecolor="black", alpha=0.55, label=base_label),
                 Patch(facecolor=att_color, edgecolor="black", alpha=0.55, label=att_label)],
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@dataclass(frozen=True)
class Paths:
    base_dir: str
    dataset_name: str
    exp_id: int
    data_csv_path: str
    results_root: str

    @property
    def att_dir(self) -> str:
        return os.path.join(self.results_root, self.dataset_name, f"exp_{self.exp_id}", "AttentionDeLELSTM")

    @property
    def base_model_dir(self) -> str:
        return os.path.join(self.results_root, self.dataset_name, f"exp_{self.exp_id}", "Delelstm")


def _load_attention_weights(p: Paths, split: str) -> tuple[np.ndarray, np.ndarray]:
    split = split.lower()
    if split not in {"test", "val"}:
        raise ValueError("split must be 'test' or 'val'")
    alpha_path = os.path.join(p.att_dir, f"{p.exp_id}_{split}alpha.csv")
    beta_path = os.path.join(p.att_dir, f"{p.exp_id}_{split}beta.csv")
    if not os.path.exists(alpha_path):
        raise FileNotFoundError(alpha_path)
    if not os.path.exists(beta_path):
        raise FileNotFoundError(beta_path)
    return _read_csv_matrix(alpha_path), _read_csv_matrix(beta_path)


def _load_delelstm_weights(p: Paths, split: str, n_vars: int) -> tuple[np.ndarray, np.ndarray]:
    split = split.lower()
    if split == "test":
        path = os.path.join(p.base_model_dir, f"{p.exp_id}_Explain_test_weight.csv")
    elif split == "val":
        path = os.path.join(p.base_model_dir, f"{p.exp_id}_valweight.csv")
    else:
        raise ValueError("split must be 'test' or 'val'")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    w = _read_csv_matrix(path)
    if w.shape[1] < n_vars * 2:
        raise ValueError(f"Expected at least {n_vars * 2} columns in {path}, got {w.shape[1]}")
    alpha = w[:, :n_vars]
    beta = w[:, n_vars:n_vars * 2]
    return alpha, beta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Electricity")
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--results_dir", type=str, default="results_comparison")
    parser.add_argument("--data_csv", type=str, default=os.path.join("DATA", "Electricity", "newX_train.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("stability_outputs", "Electricity"))
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, args.results_dir)
    data_csv_path = os.path.join(base_dir, args.data_csv)

    p = Paths(
        base_dir=base_dir,
        dataset_name=args.dataset,
        exp_id=args.exp_id,
        data_csv_path=data_csv_path,
        results_root=results_root,
    )

    var_names = _load_variable_names(p.data_csv_path, drop_cols=2)
    n_vars = len(var_names)

    alpha_att, beta_att = _load_attention_weights(p, args.split)
    alpha_base, beta_base = _load_delelstm_weights(p, args.split, n_vars=n_vars)

    t = min(alpha_att.shape[0], alpha_base.shape[0])
    alpha_att = alpha_att[:t, :n_vars]
    beta_att = beta_att[:t, :n_vars]
    alpha_base = alpha_base[:t, :n_vars]
    beta_base = beta_base[:t, :n_vars]

    alpha_std_base = _mean_std_over_time(alpha_base)
    beta_std_base = _mean_std_over_time(beta_base)
    alpha_std_att = _mean_std_over_time(alpha_att)
    beta_std_att = _mean_std_over_time(beta_att)

    df = pd.DataFrame(
        [
            {"model": "Delelstm", "alpha_mean_std": alpha_std_base, "beta_mean_std": beta_std_base},
            {"model": "AttentionDeLELSTM", "alpha_mean_std": alpha_std_att, "beta_mean_std": beta_std_att},
        ]
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    os.makedirs(args.out_dir, exist_ok=True)

    importance = _global_importance(alpha_att, beta_att)
    topk = 4 if n_vars >= 4 else n_vars
    top_idx = np.argsort(-importance)[:topk]
    top_vars = [var_names[i] for i in top_idx]
    print("Top variables by global importance:", ", ".join(top_vars))

    for i in top_idx:
        v = var_names[i]
        safe_v = _safe_name(v)
        _plot_lines(
            out_path=os.path.join(args.out_dir, f"{_safe_name(args.dataset)}_alpha_timeseries_{safe_v}.png"),
            title=f"{args.dataset} alpha over time: {v}",
            y_label="alpha",
            base_series=alpha_base[:, i],
            att_series=alpha_att[:, i],
            base_label="Delelstm",
            att_label="AttentionDeLELSTM",
        )
        _plot_lines(
            out_path=os.path.join(args.out_dir, f"{_safe_name(args.dataset)}_beta_timeseries_{safe_v}.png"),
            title=f"{args.dataset} beta over time: {v}",
            y_label="beta",
            base_series=beta_base[:, i],
            att_series=beta_att[:, i],
            base_label="Delelstm",
            att_label="AttentionDeLELSTM",
        )

    _grouped_boxplot(
        out_path=os.path.join(args.out_dir, f"{_safe_name(args.dataset)}_alpha_boxplot_all_variables.png"),
        title=f"{args.dataset} alpha distribution over time (all variables)",
        y_label="alpha",
        var_names=var_names,
        base_weights=alpha_base,
        att_weights=alpha_att,
        base_label="Delelstm",
        att_label="AttentionDeLELSTM",
    )
    _grouped_boxplot(
        out_path=os.path.join(args.out_dir, f"{_safe_name(args.dataset)}_beta_boxplot_all_variables.png"),
        title=f"{args.dataset} beta distribution over time (all variables)",
        y_label="beta",
        var_names=var_names,
        base_weights=beta_base,
        att_weights=beta_att,
        base_label="Delelstm",
        att_label="AttentionDeLELSTM",
    )


if __name__ == "__main__":
    main()

