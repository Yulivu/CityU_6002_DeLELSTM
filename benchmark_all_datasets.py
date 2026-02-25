import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from model_training import PM_training, elec_training, exchange_training
from networks import (
    AttentionDeLELSTM,
    DeLELSTM_AttnNoDecomp,
    Delelstm,
    IMVFullLSTM_pertime,
    IMVTensorLSTM_pertime,
    Retain_pertime,
    normalLSTMpertime,
)


MODEL_ORDER = ["LSTM", "Retain", "IMV_full", "IMV_tensor", "Delelstm", "DeLELSTM_AttnNoDecomp", "AttentionDeLELSTM"]
MODEL_DISPLAY = {
    "LSTM": "LSTM",
    "Retain": "RETAIN",
    "IMV_full": "IMV-Full",
    "IMV_tensor": "IMV-Tensor",
    "Delelstm": "DeLELSTM",
    "DeLELSTM_AttnNoDecomp": "DeLELSTM+Attn(w/o Decomp)",
    "AttentionDeLELSTM": "AttentionDeLELSTM",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    depth: int
    input_dim: int
    output_dim: int
    N_units: int
    n_units: int
    batch_size: int
    data_subdir: str
    feature_start_col: int
    dataset_arg: str


DATASETS = [
    DatasetSpec(
        key="electricity",
        display="Electricity",
        depth=24,
        input_dim=16,
        output_dim=1,
        N_units=64,
        n_units=64,
        batch_size=64,
        data_subdir=os.path.join("DATA", "Electricity"),
        feature_start_col=2,
        dataset_arg="electricity",
    ),
    DatasetSpec(
        key="pm25",
        display="PM2.5",
        depth=24,
        input_dim=8,
        output_dim=1,
        N_units=32,
        n_units=32,
        batch_size=32,
        data_subdir=os.path.join("DATA", "PM2.5"),
        feature_start_col=6,
        dataset_arg="PM",
    ),
    DatasetSpec(
        key="exchange",
        display="Exchange",
        depth=30,
        input_dim=8,
        output_dim=1,
        N_units=32,
        n_units=32,
        batch_size=128,
        data_subdir=os.path.join("DATA", "Exchange"),
        feature_start_col=1,
        dataset_arg="exchange",
    ),
]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _make_model(model_name: str, cfg: dict, short: int) -> torch.nn.Module:
    if model_name == "Delelstm":
        return Delelstm(cfg, short)
    if model_name == "DeLELSTM_AttnNoDecomp":
        return DeLELSTM_AttnNoDecomp(cfg, short)
    if model_name == "AttentionDeLELSTM":
        return AttentionDeLELSTM(cfg, short)
    if model_name == "IMV_full":
        return IMVFullLSTM_pertime(cfg, short)
    if model_name == "IMV_tensor":
        return IMVTensorLSTM_pertime(cfg, short)
    if model_name == "Retain":
        return Retain_pertime(cfg, short)
    if model_name == "LSTM":
        return normalLSTMpertime(cfg, short)
    raise ModuleNotFoundError(f"Unknown model_name: {model_name}")


def _load_split(
    data_path: str,
    feature_start_col: int,
    depth: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data = pd.read_csv(os.path.join(data_path, "newX_train.csv"))
    y = pd.read_csv(os.path.join(data_path, "second_y.csv"))

    data_idx = np.array(list(set(data["idx"])))
    _set_seed(seed)

    N = len(set(data["idx"]))
    cols = list(data.columns[feature_start_col:])
    for col in cols:
        data[col] = (data[col] - np.min(data[col])) / (np.max(data[col]) - np.min(data[col]))

    train_idx = np.random.choice(data_idx, int(0.75 * N), replace=False)
    remain = data_idx[~np.isin(data_idx, train_idx)]
    val_idx = np.random.choice(remain, int(0.15 * N), replace=False)
    test_idx = remain[~np.isin(remain, val_idx)]

    train_X = data.loc[data["idx"].isin(train_idx), :]
    val_X = data.loc[data["idx"].isin(val_idx), :]
    test_X = data.loc[data["idx"].isin(test_idx), :]

    X_train = torch.tensor(np.array(train_X.iloc[:, feature_start_col:]), dtype=torch.float32)
    X_val = torch.tensor(np.array(val_X.iloc[:, feature_start_col:]), dtype=torch.float32)
    X_test = torch.tensor(np.array(test_X.iloc[:, feature_start_col:]), dtype=torch.float32)

    train_Y = y.loc[y["idx"].isin(train_idx), :]
    val_Y = y.loc[y["idx"].isin(val_idx), :]
    test_Y = y.loc[y["idx"].isin(test_idx), :]

    X_train_t = X_train.reshape(len(train_idx), depth, len(cols))
    X_val_t = X_val.reshape(len(val_idx), depth, len(cols))
    X_test_t = X_test.reshape(len(test_idx), depth, len(cols))

    y_train_t = torch.tensor(train_Y["target"].to_numpy(), dtype=torch.float32).reshape(len(train_idx), depth - 2)
    y_val_t = torch.tensor(val_Y["target"].to_numpy(), dtype=torch.float32).reshape(len(val_idx), depth - 2)
    y_test_t = torch.tensor(test_Y["target"].to_numpy(), dtype=torch.float32).reshape(len(test_idx), depth - 2)

    return (
        DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1, shuffle=False),
        DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=1, shuffle=False),
        DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=1, shuffle=False),
        len(cols),
    )


def _rebuild_loaders(
    base_loader: DataLoader,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    ds = base_loader.dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _result_dir(save_root: str, ds: DatasetSpec, exp_id: int, model_name: str) -> str:
    if ds.key == "pm25":
        return os.path.join(save_root, "newrevision_pmc", f"exp_{exp_id}", model_name)
    if ds.key == "electricity":
        return os.path.join(save_root, "Electricity", f"exp_{exp_id}", model_name)
    if ds.key == "exchange":
        return os.path.join(save_root, "Exchange", f"exp_{exp_id}", model_name)
    raise ValueError(ds.key)


def _result_file(save_root: str, ds: DatasetSpec, exp_id: int, model_name: str) -> str:
    return os.path.join(_result_dir(save_root, ds, exp_id, model_name), f"{exp_id}_Explain_test_results.csv")


def _read_metrics(path: str) -> dict:
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {"rmse": float(row["rmse_test"]), "mae": float(row["mae_test"]), "mape": float(row["mape_test"])}


def _format_value(metric: str, mean: float, std: float) -> str:
    if metric == "mape":
        mean = mean * 100.0
        std = std * 100.0
        if abs(std) < 1e-4:
            return f"{mean:.2f}%±{std:.1e}%"
        return f"{mean:.2f}%±{std:.2f}%"
    if abs(mean) < 0.01:
        m = f"{mean:.4f}"
        s = f"{std:.1e}" if abs(std) < 1e-4 else f"{std:.4f}"
        return f"{m}±{s}"
    return f"{mean:.4f}±{std:.4f}"


def _render_table(summary: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (dataset, metric), row in summary.iterrows():
        r = {"Dataset": dataset, "Metric": metric.upper()}
        for m in MODEL_ORDER:
            mu = row[(m, "mean")]
            sd = row[(m, "std")]
            r[MODEL_DISPLAY[m]] = _format_value(metric, mu, sd)
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def _to_latex_table(pretty: pd.DataFrame) -> str:
    cols = list(pretty.columns)
    model_cols = cols[2:]
    header = "Dataset & Metric & " + " & ".join(model_cols) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "ll" + "c" * len(model_cols) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    current_dataset = None
    for _, row in pretty.iterrows():
        ds = row["Dataset"]
        metric = row["Metric"]
        ds_cell = ds if ds != current_dataset else ""
        current_dataset = ds
        vals = [str(row[c]) for c in model_cols]
        line = ds_cell + " & " + metric + " & " + " & ".join(vals) + " \\\\"
        lines.append(line)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Performance (mean $\\pm$ std) over 5 runs, 200 epochs}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _to_markdown_table(pretty: pd.DataFrame) -> str:
    cols = list(pretty.columns)
    rows = pretty.astype(str).values.tolist()
    widths = [len(str(c)) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))

    def fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    header = fmt_row(cols)
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join(fmt_row([str(v) for v in r]) for r in rows)
    return header + "\n" + sep + "\n" + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_exp", type=int, default=5)
    parser.add_argument("--save_root", type=str, default="results_benchmark")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=555)
    parser.add_argument("--models", type=str, default=",".join(MODEL_ORDER))
    parser.add_argument("--datasets", type=str, default="electricity,pm25,exchange")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ridge_lambda", type=float, default=0.01)
    parser.add_argument("--attention_threshold", type=float, default=0.05)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_root = os.path.join(base_dir, args.save_root)
    os.makedirs(save_root, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chosen_models = [s.strip() for s in args.models.split(",") if s.strip()]
    chosen_datasets = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}
    specs = [d for d in DATASETS if d.key in chosen_datasets]
    dataset_order = [d.display for d in DATASETS if d.key in chosen_datasets]

    runs_path = os.path.join(save_root, "benchmark_runs.csv")
    if not os.path.exists(runs_path):
        pd.DataFrame(columns=["dataset", "exp_id", "model", "rmse", "mae", "mape"]).to_csv(runs_path, index=False)

    run_records = []
    for ds in specs:
        data_path = os.path.join(base_dir, ds.data_subdir)
        for exp_id in range(args.num_exp):
            seed = int(args.base_seed) + int(exp_id)
            base_train, base_val, base_test, n_features = _load_split(
                data_path=data_path,
                feature_start_col=ds.feature_start_col,
                depth=ds.depth,
                seed=seed,
            )

            train_loader = _rebuild_loaders(base_train, ds.batch_size, shuffle=True, drop_last=True)
            val_loader = _rebuild_loaders(base_val, ds.batch_size, shuffle=False, drop_last=False)
            test_loader = _rebuild_loaders(base_test, ds.batch_size, shuffle=False, drop_last=False)

            short = int(ds.depth * 0.1)

            class A:
                pass

            a = A()
            a.seed = seed
            a.depth = ds.depth
            a.input_dim = ds.input_dim
            a.output_dim = ds.output_dim
            a.N_units = ds.N_units
            a.n_units = ds.n_units
            a.dataset = ds.dataset_arg
            a.save_dirs = save_root
            a.data_dir = None
            a.device = str(device)
            a.epochs = int(args.epochs)
            a.batch_size = ds.batch_size
            a.num_exp = args.num_exp
            a.log = True
            a.save_models = False
            a.attention_heads = None
            a.attention_threshold = float(args.attention_threshold)
            a.ridge_lambda = float(args.ridge_lambda)

            for model_name in chosen_models:
                if model_name not in MODEL_ORDER:
                    raise ValueError(f"Unknown model in --models: {model_name}")
                out_csv = _result_file(save_root, ds, exp_id, model_name)
                if args.resume and os.path.exists(out_csv):
                    metrics = _read_metrics(out_csv)
                    rec = {
                        "dataset": ds.display,
                        "exp_id": exp_id,
                        "model": MODEL_DISPLAY[model_name],
                        **metrics,
                    }
                    run_records.append(rec)
                    pd.DataFrame([rec]).to_csv(runs_path, mode="a", header=False, index=False)
                    continue

                _set_seed(seed)
                cfg = config.config(model_name, a)
                model = _make_model(model_name, cfg, short).to(device)

                if ds.key == "electricity":
                    elec_training(model, model_name, train_loader, val_loader, test_loader, a, device, exp_id)
                elif ds.key == "exchange":
                    exchange_training(model, model_name, train_loader, val_loader, test_loader, a, device, exp_id)
                elif ds.key == "pm25":
                    PM_training(model, model_name, train_loader, val_loader, test_loader, a, device, exp_id)
                else:
                    raise ValueError(ds.key)

                if not os.path.exists(out_csv):
                    raise FileNotFoundError(out_csv)
                metrics = _read_metrics(out_csv)
                rec = {
                    "dataset": ds.display,
                    "exp_id": exp_id,
                    "model": MODEL_DISPLAY[model_name],
                    **metrics,
                }
                run_records.append(rec)
                pd.DataFrame([rec]).to_csv(runs_path, mode="a", header=False, index=False)

    df_runs = pd.DataFrame(run_records)
    df_runs.to_csv(os.path.join(save_root, "benchmark_runs_last_session.csv"), index=False)

    df = pd.read_csv(runs_path)
    df["model"] = df["model"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df = df.drop_duplicates(subset=["dataset", "exp_id", "model"], keep="last")

    model_rev = {v: k for k, v in MODEL_DISPLAY.items()}
    df["model_key"] = df["model"].map(model_rev)

    summary = (
        df.melt(id_vars=["dataset", "model_key"], value_vars=["rmse", "mae", "mape"], var_name="metric", value_name="value")
        .groupby(["dataset", "metric", "model_key"])["value"]
        .agg(["mean", "std"])
    )

    cols = pd.MultiIndex.from_product([MODEL_ORDER, ["mean", "std"]])
    wide = pd.DataFrame(index=pd.MultiIndex.from_product([dataset_order, ["rmse", "mae", "mape"]], names=["dataset", "metric"]), columns=cols, dtype=float)
    for (dataset, metric, model_key), stats in summary.iterrows():
        wide.loc[(dataset, metric), (model_key, "mean")] = stats["mean"]
        wide.loc[(dataset, metric), (model_key, "std")] = stats["std"]

    wide.to_csv(os.path.join(save_root, "benchmark_summary_stats.csv"))

    pretty = _render_table(wide)
    pretty_path = os.path.join(save_root, "benchmark_summary_table.csv")
    pretty.to_csv(pretty_path, index=False)

    md_path = os.path.join(save_root, "benchmark_summary_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(pretty))

    tex_path = os.path.join(save_root, "benchmark_summary_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_to_latex_table(pretty))

    baseline_key = "DeLELSTM_AttnNoDecomp"
    if baseline_key in wide.columns.get_level_values(0):
        rows = []
        for ds_name in dataset_order:
            row = {"dataset": ds_name}
            for metric in ["rmse", "mae", "mape"]:
                row[f"{metric}_mean"] = wide.loc[(ds_name, metric), (baseline_key, "mean")]
                row[f"{metric}_std"] = wide.loc[(ds_name, metric), (baseline_key, "std")]
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(save_root, "baseline3_DeLELSTM_AttnNoDecomp_mean_std.csv"), index=False)


if __name__ == "__main__":
    main()
