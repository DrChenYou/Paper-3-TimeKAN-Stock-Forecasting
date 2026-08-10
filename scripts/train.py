#!/usr/bin/env python3
"""Train and evaluate TimeKAN on one chronological stock CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from timekan import TimeKAN, prepare_datasets, read_market_csv, regression_metrics


def evaluate(model, loader, device):
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for inputs, expected in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
            targets.append(expected.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", default="configs/timekan.yaml")
    parser.add_argument("--output-dir", default="runs/timekan")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data_cfg, model_cfg, training_cfg = config["data"], config["model"], config["training"]
    frame = read_market_csv(args.csv, data_cfg["features"])
    datasets = prepare_datasets(
        frame,
        features=data_cfg["features"],
        target=data_cfg["target"],
        input_length=data_cfg["input_length"],
        forecast_length=data_cfg["forecast_length"],
        train_fraction=data_cfg["train_fraction"],
        validation_fraction=data_cfg["validation_fraction"],
    )
    train_loader = DataLoader(datasets.train, batch_size=training_cfg["batch_size"], shuffle=True)
    validation_loader = DataLoader(datasets.validation, batch_size=training_cfg["batch_size"])
    test_loader = DataLoader(datasets.test, batch_size=training_cfg["batch_size"])
    model = TimeKAN(
        input_dim=len(data_cfg["features"]),
        forecast_length=data_cfg["forecast_length"],
        d_model=model_cfg["d_model"],
        decomposition_layers=model_cfg["decomposition_layers"],
        chebyshev_order=model_cfg["chebyshev_order"],
        attention_heads=model_cfg["attention_heads"],
        minimum_window=model_cfg["minimum_window"],
        maximum_window=model_cfg["maximum_window"],
        dropout=model_cfg["dropout"],
        head_dropout=model_cfg["head_dropout"],
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg["maximum_epochs"],
        eta_min=training_cfg["minimum_learning_rate"],
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_loss, stale_epochs = float("inf"), 0
    for epoch in range(1, training_cfg["maximum_epochs"] + 1):
        model.train()
        for inputs, expected in train_loader:
            prediction = model(inputs.to(args.device))
            loss = torch.mean((prediction - expected.to(args.device)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation_targets, validation_predictions = evaluate(model, validation_loader, args.device)
        validation_loss = float(np.mean((validation_targets - validation_predictions) ** 2))
        print(f"epoch={epoch:03d} validation_mse={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            torch.save(model.state_dict(), output / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= training_cfg["early_stopping_patience"]:
                break
    model.load_state_dict(torch.load(output / "best_model.pt", map_location=args.device, weights_only=True))
    scaled_targets, scaled_predictions = evaluate(model, test_loader, args.device)
    target_scale = datasets.scaler.scale_[datasets.target_index]
    target_mean = datasets.scaler.mean_[datasets.target_index]
    targets = scaled_targets * target_scale + target_mean
    predictions = scaled_predictions * target_scale + target_mean
    metrics = regression_metrics(targets, predictions)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output / "test_forecasts.npz", targets=targets, predictions=predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
