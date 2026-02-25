import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from networks import AttentionDeLELSTM, Delelstm


@dataclass(frozen=True)
class SyntheticSpec:
    n_samples: int = 50
    seq_len: int = 300
    noise_std: float = 0.1
    train_ratio: float = 0.8
    input_dim: int = 2
    n_units: int = 32
    N_units: int = 32
    epochs: int = 100
    lr: float = 0.01
    batch_size: int = 8
    grad_clip: float = 5.0
    attention_heads: int = 4
    attention_threshold: float = 0.05
    ridge_lambda: float = 1e-3


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _random_walk(n_samples: int, seq_len: int, scale: float = 1.0) -> np.ndarray:
    return np.cumsum(np.random.normal(0.0, scale, size=(n_samples, seq_len)), axis=1)


def generate_groups(spec: SyntheticSpec, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _set_seed(seed)
    x1 = _random_walk(spec.n_samples, spec.seq_len, scale=1.0).astype(np.float32)
    x2 = _random_walk(spec.n_samples, spec.seq_len, scale=1.0).astype(np.float32)

    X = np.stack([x1, x2], axis=-1).astype(np.float32)

    noise = np.random.normal(0.0, spec.noise_std, size=(spec.n_samples, spec.seq_len)).astype(np.float32)

    y_A = 0.8 * x1 + 0.6 * x2 + 0.5 * (x1 * x2) + noise
    y_B = 0.8 * x1 + 0.6 * x2 + noise

    y_A = y_A.astype(np.float32)
    y_B = y_B.astype(np.float32)

    y_A_target = y_A[:, 1:-1]
    y_B_target = y_B[:, 1:-1]

    return (
        torch.from_numpy(X),
        torch.from_numpy(y_A_target),
        torch.from_numpy(X.copy()),
        torch.from_numpy(y_B_target),
    )


def _split_train_test(X: torch.Tensor, y: torch.Tensor, train_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _set_seed(seed)
    n = int(X.shape[0])
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(round(n * train_ratio))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _make_loaders(X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int) -> DataLoader:
    ds = TensorDataset(X_train, y_train)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def _train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float,
    log_every: int,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if hasattr(model, "short") and model.short > 0:
                batch_y = batch_y[:, model.short:]

            opt.zero_grad()
            out = model(batch_x.float(), device)
            y_pred = out[0].squeeze(-1)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()
            total += float(loss.detach().cpu().item())
            n_batches += 1

        if log_every and (epoch % int(log_every) == 0 or epoch == int(epochs) - 1):
            mse = total / max(n_batches, 1)
            rmse = mse**0.5
            print(f"Epoch {epoch:03d}/{int(epochs)-1}: mse={mse:.6f}, rmse={rmse:.6f}")


def _evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        if hasattr(model, "short") and model.short > 0:
            y = y[:, model.short:]
        out = model(X.float(), device)
        pred = out[0].squeeze(-1)
        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
    rmse = float(mean_squared_error(y_np, pred_np) ** 0.5)
    mae = float(mean_absolute_error(y_np, pred_np))
    return rmse, mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="synthetic_interaction_results.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    spec = SyntheticSpec()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, args.out_csv)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short = int(spec.seq_len * 0.1)

    X_A, y_A, X_B, y_B = generate_groups(spec, seed=int(args.seed))

    X_A_train, y_A_train, X_A_test, y_A_test = _split_train_test(X_A, y_A, train_ratio=spec.train_ratio, seed=int(args.seed))
    _, _, X_B_test, y_B_test = _split_train_test(X_B, y_B, train_ratio=spec.train_ratio, seed=int(args.seed))

    train_loader = _make_loaders(X_A_train, y_A_train, batch_size=spec.batch_size)

    models = []
    models.append(
        (
            "DeLELSTM",
            Delelstm(
                {
                    "input_dim": spec.input_dim,
                    "n_units": spec.n_units,
                    "time_depth": spec.seq_len,
                    "output_dim": 1,
                    "N_units": spec.N_units,
                },
                short,
            ).to(device),
        )
    )
    models.append(
        (
            "AttentionDeLELSTM",
            AttentionDeLELSTM(
                {
                    "input_dim": spec.input_dim,
                    "n_units": spec.n_units,
                    "time_depth": spec.seq_len,
                    "output_dim": 1,
                    "N_units": spec.N_units,
                    "attention_heads": spec.attention_heads,
                    "attention_threshold": spec.attention_threshold,
                    "ridge_lambda": spec.ridge_lambda,
                },
                short,
            ).to(device),
        )
    )

    rows = []
    for model_name, model in models:
        print(f"Training {model_name} on Group A (with interaction)...")
        _train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=spec.epochs,
            lr=spec.lr,
            grad_clip=spec.grad_clip,
            log_every=int(args.log_every),
        )
        print(f"Evaluating {model_name} on Group A / Group B...")

        rmse_A, mae_A = _evaluate(model, X_A_test, y_A_test, device=device)
        rmse_B, mae_B = _evaluate(model, X_B_test, y_B_test, device=device)

        rows.append({"model": model_name, "group": "A_with_interaction", "rmse": rmse_A, "mae": mae_A})
        rows.append({"model": model_name, "group": "B_no_interaction", "rmse": rmse_B, "mae": mae_B})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
