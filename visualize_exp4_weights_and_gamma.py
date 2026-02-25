import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    data_subdir: str
    results_subdir: str
    feature_start_col: int


DATASETS = [
    DatasetSpec(
        key="electricity",
        display="Electricity",
        data_subdir=os.path.join("DATA", "Electricity"),
        results_subdir="Electricity",
        feature_start_col=2,
    ),
    DatasetSpec(
        key="exchange",
        display="Exchange",
        data_subdir=os.path.join("DATA", "Exchange"),
        results_subdir="Exchange",
        feature_start_col=1,
    ),
    DatasetSpec(
        key="pm25",
        display="PM2.5",
        data_subdir=os.path.join("DATA", "PM2.5"),
        results_subdir="newrevision_pmc",
        feature_start_col=6,
    ),
]


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_feature_names(base_dir: str, ds: DatasetSpec) -> list[str]:
    path = os.path.join(base_dir, ds.data_subdir, "newX_train.csv")
    df = pd.read_csv(path, nrows=1)
    cols = list(df.columns)
    feat = cols[ds.feature_start_col:]
    return [str(c) for c in feat]


def _result_dir(save_root: str, ds: DatasetSpec, exp_id: int, model: str) -> str:
    return os.path.join(save_root, ds.results_subdir, f"exp_{exp_id}", model)


def _read_attention_alpha_beta(save_root: str, ds: DatasetSpec, exp_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _result_dir(save_root, ds, exp_id, "AttentionDeLELSTM")
    alpha_path = os.path.join(d, f"{exp_id}_testalpha.csv")
    beta_path = os.path.join(d, f"{exp_id}_testbeta.csv")
    alpha = pd.read_csv(alpha_path, index_col=0)
    beta = pd.read_csv(beta_path, index_col=0)
    return alpha, beta


def _read_delelstm_alpha_beta(save_root: str, ds: DatasetSpec, exp_id: int, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _result_dir(save_root, ds, exp_id, "Delelstm")
    w_path = os.path.join(d, f"{exp_id}_Explain_test_weight.csv")
    w = pd.read_csv(w_path, index_col=0)
    n_cols = int(w.shape[1])
    if n_cols % 2 != 0:
        raise ValueError(f"Unexpected DeLELSTM weight shape: {w.shape} at {w_path}")
    half = n_cols // 2
    names = feature_names[:half]
    alpha = w.iloc[:, :half].copy()
    beta = w.iloc[:, half:half * 2].copy()
    alpha.columns = names
    beta.columns = names
    return alpha, beta


def _align_attention_cols(alpha: pd.DataFrame, beta: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(alpha.shape[1])
    if n > len(feature_names):
        alpha = alpha.iloc[:, : len(feature_names)].copy()
        beta = beta.iloc[:, : len(feature_names)].copy()
        n = int(alpha.shape[1])
    names = feature_names[:n]
    alpha = alpha.copy()
    beta = beta.copy()
    alpha.columns = names
    beta.columns = names
    return alpha, beta


def _std_over_time(df: pd.DataFrame) -> np.ndarray:
    x = df.to_numpy(dtype=float)
    return np.std(x, axis=0, ddof=0)


def _plot_volatility_boxplots(
    ds_display: str,
    exp_id: int,
    out_dir: str,
    dele_alpha: pd.DataFrame,
    dele_beta: pd.DataFrame,
    att_alpha: pd.DataFrame,
    att_beta: pd.DataFrame,
) -> str:
    a1 = _std_over_time(dele_alpha)
    a2 = _std_over_time(att_alpha)
    b1 = _std_over_time(dele_beta)
    b2 = _std_over_time(att_beta)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([a1, a2], tick_labels=["DeLELSTM", "AttentionDeLELSTM"], showfliers=False)
    axes[0].set_title(f"{ds_display} Alpha volatility (exp_{exp_id})")
    axes[0].set_ylabel("Std over time")

    axes[1].boxplot([b1, b2], tick_labels=["DeLELSTM", "AttentionDeLELSTM"], showfliers=False)
    axes[1].set_title(f"{ds_display} Beta volatility (exp_{exp_id})")
    axes[1].set_ylabel("Std over time")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"volatility_boxplot_{ds_display}_exp_{exp_id}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _read_gamma(save_root: str, ds: DatasetSpec, exp_id: int) -> np.ndarray:
    d = _result_dir(save_root, ds, exp_id, "AttentionDeLELSTM")
    g_path = os.path.join(d, f"{exp_id}_testgamma.csv")
    g = pd.read_csv(g_path, index_col=0).to_numpy(dtype=float)
    n = int(g.shape[1])
    dim = int(round(n ** 0.5))
    if dim * dim != n:
        raise ValueError(f"Unexpected gamma columns: {n} at {g_path}")
    return g.reshape(g.shape[0], dim, dim)


def _topk_by_interaction_strength(mat: np.ndarray, k: int) -> np.ndarray:
    m = mat.copy()
    np.fill_diagonal(m, 0.0)
    strength = m.sum(axis=0) + m.sum(axis=1)
    k = int(min(k, len(strength)))
    idx = np.argsort(-strength)[:k]
    return np.sort(idx)


def _plot_gamma_heatmap(
    ds_display: str,
    exp_id: int,
    out_dir: str,
    gamma_ts: np.ndarray,
    feature_names: list[str],
    top_k: int,
) -> str:
    strength = np.mean(np.abs(gamma_ts), axis=0)
    mean_gamma = np.mean(gamma_ts, axis=0)
    k = int(min(top_k, strength.shape[0]))
    idx = _topk_by_interaction_strength(strength, k)
    sub = mean_gamma[np.ix_(idx, idx)]
    names = [feature_names[i] for i in idx.tolist()]

    vmax = float(np.percentile(np.abs(sub), 99)) if sub.size else None

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sub, cmap="RdBu_r", interpolation="nearest", vmin=None if vmax is None else -vmax, vmax=vmax)
    ax.set_title(f"{ds_display} Gamma interaction (mean gamma, exp_{exp_id}, top {len(names)})")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean gamma (signed)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"gamma_heatmap_{ds_display}_exp_{exp_id}_top_{len(names)}.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="results_benchmark_200e5")
    parser.add_argument("--exp_id", type=int, default=4)
    parser.add_argument("--top_k_gamma", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = _repo_root()
    save_root = os.path.join(base_dir, args.save_root)
    out_dir = args.out_dir if args.out_dir else os.path.join(save_root, "viz_exp4")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for ds in DATASETS:
        feature_names = _load_feature_names(base_dir, ds)

        att_alpha, att_beta = _read_attention_alpha_beta(save_root, ds, args.exp_id)
        att_alpha, att_beta = _align_attention_cols(att_alpha, att_beta, feature_names)

        dele_alpha, dele_beta = _read_delelstm_alpha_beta(save_root, ds, args.exp_id, feature_names)

        _plot_volatility_boxplots(
            ds_display=ds.display,
            exp_id=args.exp_id,
            out_dir=out_dir,
            dele_alpha=dele_alpha,
            dele_beta=dele_beta,
            att_alpha=att_alpha,
            att_beta=att_beta,
        )

        summary_rows.append(
            {
                "dataset": ds.display,
                "model": "DeLELSTM",
                "alpha_mean_std_over_time": float(np.mean(_std_over_time(dele_alpha))),
                "beta_mean_std_over_time": float(np.mean(_std_over_time(dele_beta))),
            }
        )
        summary_rows.append(
            {
                "dataset": ds.display,
                "model": "AttentionDeLELSTM",
                "alpha_mean_std_over_time": float(np.mean(_std_over_time(att_alpha))),
                "beta_mean_std_over_time": float(np.mean(_std_over_time(att_beta))),
            }
        )

        gamma_ts = _read_gamma(save_root, ds, args.exp_id)
        _plot_gamma_heatmap(
            ds_display=ds.display,
            exp_id=args.exp_id,
            out_dir=out_dir,
            gamma_ts=gamma_ts,
            feature_names=feature_names,
            top_k=args.top_k_gamma,
        )

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, f"alpha_beta_volatility_summary_exp_{args.exp_id}.csv"), index=False)


if __name__ == "__main__":
    main()
