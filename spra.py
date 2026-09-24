"""Training-free shape-phase retrieval with optional K-medoids compression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


SHAPE_WEIGHT = 2 / 3
PHASE_WEIGHT = 4 / 3


def device_from(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return device


def split_bounds(rows, train_ratio=.7, test_ratio=.2, *, train_end=None,
                 test_start=None, test_end=None):
    explicit = (train_end, test_start, test_end)
    if any(x is not None for x in explicit):
        if not all(x is not None for x in explicit):
            raise ValueError("provide all three explicit split boundaries")
    else:
        if not 0 < train_ratio < 1 or not 0 < test_ratio < 1:
            raise ValueError("split ratios must lie between zero and one")
        if train_ratio + test_ratio > 1:
            raise ValueError("train_ratio + test_ratio must not exceed one")
        train_end = int(rows * train_ratio)
        test_start = rows - int(rows * test_ratio)
        test_end = rows
    if not 0 < train_end <= test_start < test_end <= rows:
        raise ValueError("invalid chronological split")
    return train_end, test_start, test_end


def load_csv(path, train_ratio=.7, test_ratio=.2, **boundaries):
    frame = pd.read_csv(path)
    columns = {}
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().all():
            columns[str(name)] = values
    if not columns:
        raise ValueError("CSV has no finite numeric columns")
    numeric = pd.DataFrame(columns)
    raw = numeric.to_numpy(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("numeric columns must contain only finite values")
    bounds = split_bounds(len(raw), train_ratio, test_ratio, **boundaries)
    scaled = StandardScaler().fit(raw[:bounds[0]]).transform(raw).astype(np.float32)
    return scaled, bounds, list(numeric.columns)


def normalized_shape(history):
    mean = history.mean(1, keepdims=True)
    centered = history - mean
    scale = np.sqrt(np.mean(centered.astype(np.float64) ** 2, axis=1))
    flat = scale <= 1e-6 * np.maximum(1, np.abs(mean[:, 0]))
    return (centered / np.maximum(scale, 1e-12)[:, None]).astype(np.float32), flat


@torch.inference_mode()
def assign_medoids(normalized, flat, origins, medoids, period, device, batch_size):
    values = torch.as_tensor(np.ascontiguousarray(normalized), device=device)
    flats = torch.as_tensor(flat, device=device)
    times = torch.as_tensor(origins, device=device)
    ids = torch.as_tensor(medoids, device=device)
    centers, center_flat, center_time = values[ids], flats[ids], times[ids]
    labels, costs = [], []
    for start in range(0, len(values), batch_size):
        block = values[start:start + batch_size]
        shape = 1 - (block @ centers.T / values.shape[1]).clamp(-1, 1)
        block_flat = flats[start:start + len(block), None]
        shape = torch.where(block_flat | center_flat[None], .5, shape)
        shape = torch.where(block_flat & center_flat[None], 0., shape)
        delta = times[start:start + len(block), None] - center_time[None]
        phase = torch.remainder(delta, period)
        phase = torch.minimum(phase, period - phase) / (period / 2)
        cost, label = (SHAPE_WEIGHT * shape + PHASE_WEIGHT * phase).min(1)
        labels.append(label.cpu().numpy())
        costs.append(cost.cpu().numpy())
    return np.concatenate(labels), np.concatenate(costs)


def update_medoid(normalized, flat, origins, members, period, current, candidate_cap):
    candidates = members
    if candidate_cap and len(members) > candidate_cap:
        positions = np.linspace(0, len(members) - 1, candidate_cap).round().astype(int)
        candidates = members[positions]
        if current not in candidates:
            candidates = np.append(candidates, current)
    candidate_values = normalized[candidates].astype(np.float64)
    member_values = normalized[members].astype(np.float64)
    shape = 1 - np.clip(candidate_values @ member_values.T / normalized.shape[1], -1, 1)
    candidate_flat, member_flat = flat[candidates], flat[members]
    shape = np.where(candidate_flat[:, None] | member_flat[None], .5, shape)
    shape = np.where(candidate_flat[:, None] & member_flat[None], 0., shape)
    delta = np.remainder(origins[candidates, None] - origins[None, members], period)
    phase = np.minimum(delta, period - delta) / (period / 2)
    costs = (SHAPE_WEIGHT * shape + PHASE_WEIGHT * phase).sum(1)
    best = int(np.argmin(costs))
    current_position = np.flatnonzero(candidates == current)[0]
    tolerance = 1e-10 * max(1, abs(float(costs[current_position])))
    return int(current if costs[best] >= costs[current_position] - tolerance
               else candidates[best])


def kmedoids(history, origins, period, compression, device, max_iter,
             candidate_cap, batch_size):
    normalized, flat = normalized_shape(history)
    count = (len(history) + compression - 1) // compression
    medoids = np.linspace(0, len(history) - 1, count).round().astype(np.int64)
    for _ in range(max_iter):
        labels, costs = assign_medoids(
            normalized, flat, origins, medoids, period, device, batch_size)
        labels[medoids], costs[medoids] = np.arange(count), 0
        sizes = np.bincount(labels, minlength=count)
        order = np.argsort(labels, kind="stable")
        boundaries = np.concatenate(([0], np.cumsum(sizes)))
        updated = np.empty_like(medoids)
        for cluster in range(count):
            members = order[boundaries[cluster]:boundaries[cluster + 1]]
            updated[cluster] = update_medoid(
                normalized, flat, origins, members, period,
                int(medoids[cluster]), candidate_cap)
        if np.array_equal(updated, medoids):
            return updated
        medoids = updated
    return medoids


@torch.inference_mode()
def shape_distance(query, candidates):
    length = query.shape[-1]
    query_mean, candidate_mean = query.mean(-1), candidates.mean(-1)
    query_centered = query - query_mean[..., None]
    candidate_centered = candidates - candidate_mean[..., None]
    covariance = query_centered @ candidate_centered.T / length
    query_variance = query_centered.square().mean(-1)
    candidate_variance = candidate_centered.square().mean(-1)
    query_flat = query_variance <= (1e-6 * query_mean.abs().clamp(min=1)).square()
    candidate_flat = candidate_variance <= (
        1e-6 * candidate_mean.abs().clamp(min=1)).square()
    correlation = (covariance / (
        query_variance[:, None] * candidate_variance[None]
    ).sqrt().clamp(min=1e-12)).clamp(-1, 1)
    distance = torch.where(query_flat[:, None] | candidate_flat[None], .5,
                           1 - correlation)
    return torch.where(query_flat[:, None] & candidate_flat[None], 0., distance)


@torch.inference_mode()
def forecast_channel(series, query_origins, histories, futures, bank_origins,
                     period, top_k, temperature, device, batch_size):
    series = torch.as_tensor(np.ascontiguousarray(series), device=device)
    histories = torch.as_tensor(np.ascontiguousarray(histories), device=device)
    futures = torch.as_tensor(np.ascontiguousarray(futures), device=device)
    bank_origins = torch.as_tensor(np.ascontiguousarray(bank_origins), device=device)
    history_steps = torch.arange(histories.shape[1], device=device)
    output = np.empty((len(query_origins), futures.shape[1]), np.float32)
    for start in range(0, len(query_origins), batch_size):
        times = torch.as_tensor(query_origins[start:start + batch_size], device=device)
        query = series[times[:, None] - histories.shape[1] + history_steps]
        phase = torch.remainder(times[:, None] - bank_origins[None], period)
        phase = torch.minimum(phase, period - phase) / (period / 2)
        score = SHAPE_WEIGHT * shape_distance(query, histories) + PHASE_WEIGHT * phase
        best_score, best_id = torch.topk(
            score, min(top_k, len(histories)), largest=False, sorted=True)
        weights = torch.softmax(-(best_score - best_score[:, :1]) / temperature, -1)
        selected_history, selected_future = histories[best_id], futures[best_id]
        query_mean = query.mean(-1)
        history_mean = selected_history.mean(-1)
        query_centered = query - query_mean[:, None]
        history_centered = selected_history - history_mean[..., None]
        slope = ((query_centered[:, None] * history_centered).sum(-1)
                 / history_centered.square().sum(-1).clamp(min=1e-6))
        aligned = slope[..., None] * selected_future + (
            query_mean[:, None] - slope * history_mean)[..., None]
        prediction = (weights[..., None] * aligned).sum(-2)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("nonfinite prediction")
        output[start:start + len(times)] = prediction.cpu().numpy()
    return output


def top_k_by_horizon(values, horizons):
    if len(values) == 1:
        return values * len(horizons)
    if len(values) != len(horizons):
        raise ValueError("provide one Top-K or one per horizon")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", required=True)
    parser.add_argument("--compression", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--train-ratio", type=float, default=.7)
    parser.add_argument("--test-ratio", type=float, default=.2)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--test-start", type=int)
    parser.add_argument("--test-end", type=int)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--medoid-candidates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    top_ks = top_k_by_horizon(args.top_k, args.horizons)
    positive = [args.period, args.compression, args.seq_len, args.temperature,
                args.max_iter, args.batch_size, args.threads, *args.horizons, *top_ks]
    if min(positive) <= 0 or args.medoid_candidates < 0:
        parser.error("lengths, counts, period, and temperature must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty")

    torch.set_num_threads(args.threads)
    device = device_from(args.device)
    values, bounds, channels = load_csv(
        args.csv, args.train_ratio, args.test_ratio,
        train_end=args.train_end, test_start=args.test_start, test_end=args.test_end)
    train_end, test_start, test_end = bounds
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "compression": args.compression,
              "split": list(bounds), "channels": channels, "results": []}

    for horizon, top_k in zip(args.horizons, top_ks):
        if args.seq_len + horizon > train_end or horizon > test_end - test_start:
            raise ValueError(f"horizon {horizon} does not fit the selected split")
        query_origins = np.arange(test_start, test_end - horizon + 1)
        candidate_count = train_end - args.seq_len - horizon + 1
        bank_origins = np.arange(candidate_count) + args.seq_len
        squared = absolute = 0.0
        medoid_count = candidate_count
        for channel in range(values.shape[1]):
            windows = np.lib.stride_tricks.sliding_window_view(
                values[:train_end, channel], args.seq_len + horizon)
            if args.compression == 1:
                ids = np.arange(candidate_count)
            else:
                ids = kmedoids(
                    windows[:, :args.seq_len], bank_origins, args.period,
                    args.compression, device, args.max_iter,
                    args.medoid_candidates, args.batch_size)
            medoid_count = len(ids)
            prediction = forecast_channel(
                values[:, channel], query_origins,
                windows[ids, :args.seq_len].astype(np.float32),
                windows[ids, args.seq_len:].astype(np.float32),
                bank_origins[ids].astype(np.float32), args.period, top_k,
                args.temperature, device, args.batch_size)
            target = np.lib.stride_tricks.sliding_window_view(
                values[:, channel], horizon)[query_origins]
            error = prediction.astype(np.float64) - target.astype(np.float64)
            squared += float(np.square(error).sum())
            absolute += float(np.abs(error).sum())
            print(f"H={horizon} channel={channel + 1}/{values.shape[1]}", flush=True)
        count = values.shape[1] * len(query_origins) * horizon
        report["results"].append({
            "horizon": horizon, "top_k": top_k,
            "candidates": candidate_count, "medoids": medoid_count,
            "mse": squared / count, "mae": absolute / count,
        })

    report["status"] = "complete"
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
