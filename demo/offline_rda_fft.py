"""Offline packet-HDF5 to RDA FFT export."""

"""其功能是将采集后的数据完整的fft处理.尽量在后端执行"""
from pathlib import Path

import importlib
import json
import os

import numpy as np
import tyro
import yaml
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import xwr
from xwr.rsp import iq_from_iiqq, numpy as xwr_rsp

C0 = 299_792_458.0


def _to_hz_sample_rate(sample_rate: float) -> float:
    return sample_rate * 1e3 if sample_rate < 1e5 else sample_rate


def _to_hz_per_s_slope(freq_slope: float) -> float:
    return freq_slope * 1e12 if freq_slope < 1e6 else freq_slope


def _resolve_h5_path(h5_file: str | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    if h5_file is not None:
        p = Path(h5_file)
        if p.is_absolute():
            return p
        if p.parent == Path("."):
            return data_dir / p
        cwd_candidate = Path.cwd() / p
        return cwd_candidate if cwd_candidate.exists() else (root / p)
    files = sorted(data_dir.glob("*.h5"), key=lambda x: x.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")
    return files[-1]


def _import_h5py():
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'h5py'. Install project dependencies with uv sync."
        ) from e


def _import_jax():
    try:
        return importlib.import_module("jax")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'jax'. Install with `uv sync --extra jax`."
        ) from e


def _iter_frames_from_packets_h5(h5_path: Path, raw_shape: tuple[int, ...]):
    h5py = _import_h5py()
    frame_size = int(np.prod(raw_shape)) * np.dtype(np.uint16).itemsize
    with h5py.File(str(h5_path), "r") as f:
        packets = f["scan"]["packet"]
        offset = 0
        buf = bytearray()
        frame_t: float | None = None
        for rec in packets:
            byte_count = int(rec["byte_count"])
            payload = np.asarray(rec["packet_data"], dtype=np.uint16).tobytes()
            if frame_t is None:
                frame_t = float(rec["t"])

            if offset == 0:
                offset = byte_count - (byte_count % frame_size)

            missing = byte_count - offset
            if missing < 0:
                continue
            if missing > 0:
                buf.extend(b"\x00" * missing)
                offset = byte_count

            buf.extend(payload)
            offset += len(payload)

            while len(buf) >= frame_size:
                frame_bytes = bytes(buf[:frame_size])
                del buf[:frame_size]
                frame = np.frombuffer(frame_bytes, dtype=np.int16).reshape(raw_shape)
                yield frame, frame_t
                # Keep the same timestamp for additional full frames that come
                # from the same packet span; clear only when we need new data.
                if len(buf) < frame_size:
                    frame_t = None


def _count_frames_from_packets_h5(h5_path: Path, raw_shape: tuple[int, ...]) -> int:
    h5py = _import_h5py()
    frame_size = int(np.prod(raw_shape)) * np.dtype(np.uint16).itemsize
    with h5py.File(str(h5_path), "r") as f:
        packets = f["scan"]["packet"]
        offset = 0
        buf_len = 0
        total = 0
        for rec in packets:
            byte_count = int(rec["byte_count"])
            payload_len = int(np.asarray(rec["packet_data"], dtype=np.uint16).size) * 2

            if offset == 0:
                offset = byte_count - (byte_count % frame_size)

            missing = byte_count - offset
            if missing < 0:
                continue
            if missing > 0:
                buf_len += missing
                offset = byte_count

            buf_len += payload_len
            offset += payload_len

            complete = buf_len // frame_size
            if complete > 0:
                total += int(complete)
                buf_len = buf_len % frame_size
    return total


def _open_output(out_file: str | None, h5_path: Path) -> Path:
    if out_file is not None:
        p = Path(out_file)
        return p if p.is_absolute() else (Path.cwd() / p)
    return h5_path.with_name("radar.h5")


def _to_seconds_time(t: float) -> float:
    return t * 1e-6 if t > 1.0 else t


def _extract_range_bias_m(cfg_radar: dict) -> float:
    calibration = cfg_radar.get("calibration")
    if calibration is None:
        return 0.0
    if isinstance(calibration, str):
        parts = calibration.strip().split()
        if parts and parts[0] == "compRangeBiasAndRxChanPhase":
            parts = parts[1:]
        return float(parts[0]) if parts else 0.0
    if isinstance(calibration, dict):
        cmd = calibration.get("comp_range_bias_and_rx_chan_phase")
        if isinstance(cmd, str) and cmd.strip():
            parts = cmd.strip().split()
            if parts and parts[0] == "compRangeBiasAndRxChanPhase":
                parts = parts[1:]
            if parts:
                return float(parts[0])
        return float(calibration.get("range_bias", 0.0))
    return 0.0


def _infer_num_tx(rsp_inst, default: int = 2) -> int:
    for name in ("num_tx", "n_tx", "tx", "ntx"):
        if hasattr(rsp_inst, name):
            v = getattr(rsp_inst, name)
            if isinstance(v, (int, np.integer)) and int(v) > 0:
                return int(v)
    return default


def make_doppler_axis_mps(cfg_radar: dict, n_doppler: int, num_tx: int) -> np.ndarray:
    fc = float(cfg_radar["frequency"]) * 1e9
    lam = C0 / fc
    idle = _to_seconds_time(float(cfg_radar["idle_time"]))
    ramp = _to_seconds_time(float(cfg_radar["ramp_end_time"]))
    t_chirp = idle + ramp
    t_eff = t_chirp * max(1, int(num_tx))
    fd = np.fft.fftshift(np.fft.fftfreq(n_doppler, d=t_eff))
    return (lam / 2.0) * fd


def make_range_axis_m(cfg_radar: dict, n_range: int) -> np.ndarray:
    fs = _to_hz_sample_rate(float(cfg_radar["sample_rate"]))
    slope = _to_hz_per_s_slope(float(cfg_radar["freq_slope"]))
    k = np.arange(n_range, dtype=np.float64)
    return (C0 * fs / (2.0 * slope * n_range)) * k


def _estimate_speed_value(
    rda: np.ndarray,
    doppler_axis: np.ndarray,
    mode: str,
    percentile: float,
    min_count: int,
) -> float:
    speed_mode = mode.strip().lower()
    if speed_mode not in {"scalar", "signed"}:
        raise ValueError(f"Unsupported speed mode: {mode}. Choose one of: scalar, signed")

    rda_fp32 = rda.astype(np.float32, copy=False)
    threshold = np.percentile(rda_fp32, float(percentile), axis=(0, 1), keepdims=True)
    valid = np.sum(rda_fp32 > threshold, axis=(0, 2)) > int(max(0, min_count))
    n_doppler = int(valid.shape[0])
    nd_half = int(n_doppler // 2)
    if n_doppler == 0:
        return 0.0

    # Match rover-main logic: use nearest detected Doppler bin from both edges
    # and convert to a scalar speed (non-negative by default).
    left_valid = valid[:nd_half]
    right_valid = valid[nd_half:]
    edge_candidates: list[tuple[int, float]] = []
    if np.any(left_valid):
        left_idx = int(np.argmax(left_valid))
        edge_candidates.append((left_idx, float(abs(doppler_axis[left_idx]))))
    if np.any(right_valid):
        right_from_edge = int(np.argmax(right_valid[::-1]))
        right_idx = int(n_doppler - 1 - right_from_edge)
        edge_candidates.append((right_idx, float(abs(doppler_axis[right_idx]))))

    if edge_candidates:
        # Pick the farther speed edge; preserve sign if requested.
        speed_idx, speed_abs = max(edge_candidates, key=lambda x: x[1])
        if speed_mode == "signed":
            return float(doppler_axis[speed_idx])
        return speed_abs

    # Fallback for fully empty threshold mask.
    doppler_power = np.mean(rda_fp32, axis=(0, 2))
    speed_idx = int(np.argmax(doppler_power))
    speed_signed = float(doppler_axis[speed_idx])
    if speed_mode == "signed":
        return speed_signed
    return float(abs(speed_signed))


def _normalize_scale(scale: float) -> float:
    v = float(scale)
    return v if v > 0.0 else 1.0


def _remove_artifact_rover_style(
    images: np.ndarray,
    *,
    enabled: bool,
    percentile: float,
) -> np.ndarray:
    """Mimic rover remove_artifact: subtract percentile on zero-Doppler 3 bins."""
    if not enabled:
        return images.astype(np.float32, copy=False)

    images_fp32 = images.astype(np.float32, copy=True)
    if images_fp32.ndim != 4:
        return images_fp32

    n_doppler = int(images_fp32.shape[2])
    if n_doppler <= 0:
        return images_fp32

    center = n_doppler // 2
    lo = max(0, center - 1)
    hi = min(n_doppler, center + 2)
    if lo >= hi:
        return images_fp32

    # rover logic: percentile over frame axis, then subtract on zero-Doppler bins.
    artifact = np.percentile(images_fp32[:, :, lo:hi, :], float(percentile), axis=0)
    removed = np.maximum(images_fp32[:, :, lo:hi, :] - artifact[None, ...], 0.0)
    images_fp32[:, :, lo:hi, :] = removed
    return images_fp32


def _to_float_array_strict(values, *, dtype: np.dtype) -> np.ndarray:
    """Convert sequence-like values to a pure floating ndarray."""
    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "i", "u"}:
        return arr.astype(dtype, copy=False)

    cleaned: list[float] = []
    for v in arr.tolist():
        if v is None:
            cleaned.append(np.nan)
            continue
        if isinstance(v, str):
            vv = v.strip()
            if vv == "" or vv.lower() == "nan":
                cleaned.append(np.nan)
                continue
            cleaned.append(float(vv))
            continue
        cleaned.append(float(v))
    return np.asarray(cleaned, dtype=dtype)


def _make_sensor_intrinsics(
    cfg_radar: dict, rda_shape: tuple[int, int, int], num_tx: int
) -> dict[str, str | list[float | int]]:
    n_range, n_doppler, _ = rda_shape
    if n_range <= 0 or n_doppler <= 0:
        return {
            "gain": "awr1843boost_az8",
            "r": [0.0, 0.0, int(n_range)],
            "d": [0.0, 0.0, int(n_doppler)],
        }
    range_axis = make_range_axis_m(cfg_radar, n_range) - _extract_range_bias_m(cfg_radar)
    doppler_axis = make_doppler_axis_mps(cfg_radar, n_doppler, num_tx=num_tx)
    return {
        "gain": "awr1843boost_az8",
        "r": [float(range_axis[0]), float(range_axis[-1]), int(n_range)],
        "d": [float(doppler_axis[0]), float(doppler_axis[-1]), int(n_doppler)],
    }


def cli_main(
    h5_file: str | None = None,
    config: str | None = None,
    rsp: str = "AWR1642Boost",
    out_file: str | None = None,
    sample_every: int = 1,
    max_frames: int | None = None,
    azimuth_bins: int = 8,
    apply_window: bool = True,
    backend: str = "numpy",
    jax_device: str = "auto",
    jax_jit: bool = True,
    speed_mode: str = "scalar",
    speed_percentile: float = 99.75,
    speed_min_count: int = 10,
    rda_scale: float = 1e6,
    static_clutter_removal: bool = True,
    static_clutter_bins: int = 1,
    static_clutter_percentile: float = 50.0,
    static_clutter_batch_frames: int = 256,
) -> None:
    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config_awr1843l.yaml")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    radar_cfg = xwr.XWRConfig(**cfg["radar"])
    raw_shape = radar_cfg.raw_shape
    backend_name = backend.strip().lower()
    if backend_name not in {"numpy", "jax"}:
        raise ValueError(f"Unsupported backend: {backend}. Choose one of: numpy, jax")

    jax = None
    jnp = None
    jax_compute_device = None
    if backend_name == "jax":
        jax = _import_jax()
        jnp = importlib.import_module("jax.numpy")
        if jax_device != "auto":
            candidate_devices = jax.devices(jax_device)
            if not candidate_devices:
                raise RuntimeError(
                    f"No JAX device found for platform '{jax_device}'. "
                    f"Available: {[d.platform for d in jax.devices()]}"
                )
            jax_compute_device = candidate_devices[0]

    rsp_module = xwr_rsp if backend_name == "numpy" else importlib.import_module("xwr.rsp.jax")
    rsp_inst = getattr(rsp_module, rsp)(
        window=bool(apply_window), size={"azimuth": max(1, int(azimuth_bins))}
    )
    num_tx = _infer_num_tx(rsp_inst, default=radar_cfg.num_tx)
    rda_scale_value = _normalize_scale(rda_scale)
    clutter_percentile = float(static_clutter_percentile)
    clutter_batch_frames = max(1, int(static_clutter_batch_frames))
    if clutter_percentile < 0.0 or clutter_percentile > 100.0:
        raise ValueError("static_clutter_percentile must be in [0, 100].")
    if int(static_clutter_bins) != 1:
        print("Warning: rover-style artifact removal fixes zero_doppler_bins to 1 (3 bins total).")

    if backend_name == "numpy":
        def process_frame(frame: np.ndarray) -> np.ndarray:
            frame_batch: np.ndarray = frame[None, ...]
            if rsp_inst.SAMPLE_TYPE == "IQ":
                frame_for_rsp = iq_from_iiqq(frame_batch).astype(np.complex64, copy=False)
            else:
                frame_for_rsp = frame_batch.astype(np.float32, copy=False)

            rd = rsp_inst.doppler_range(frame_for_rsp)
            dear = np.abs(rsp_inst.elevation_azimuth(rd))[0]
            return np.transpose(np.mean(dear, axis=1), (2, 0, 1)).astype(np.float32, copy=False)
    else:
        assert jax is not None
        assert jnp is not None

        def _process_frame_jax(frame_batch):
            if rsp_inst.SAMPLE_TYPE == "IQ":
                frame_for_rsp = iq_from_iiqq(frame_batch)
            else:
                frame_for_rsp = frame_batch.astype(jnp.float32)

            rd = rsp_inst.doppler_range(frame_for_rsp)
            dear = jnp.abs(rsp_inst.elevation_azimuth(rd))[0]
            return jnp.transpose(jnp.mean(dear, axis=1), (2, 0, 1))

        if bool(jax_jit):
            _process_frame_jax = jax.jit(_process_frame_jax)

        def process_frame(frame: np.ndarray) -> np.ndarray:
            frame_batch = jnp.asarray(frame[None, ...])
            if jax_compute_device is not None:
                frame_batch = jax.device_put(frame_batch, jax_compute_device)
            return np.asarray(_process_frame_jax(frame_batch))

    h5_path = _resolve_h5_path(h5_file)
    out_path = _open_output(out_file, h5_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    h5py = _import_h5py()
    speeds: list[float] = []
    timestamps: list[float] = []

    with h5py.File(str(out_path), "w") as fout:
        rad_ds = None
        stored = 0
        processed = 0
        stride = max(1, int(sample_every))
        first_rda_shape: tuple[int, int, int] | None = None
        doppler_axis: np.ndarray | None = None
        target_frames = int(max_frames) if max_frames is not None else None
        pending_rda: list[np.ndarray] = []
        pending_t: list[float] = []

        def _flush_pending() -> None:
            nonlocal rad_ds, stored, first_rda_shape
            if not pending_rda:
                return

            batch_fp32 = np.stack(pending_rda, axis=0).astype(np.float32, copy=False)
            batch_fp32 = _remove_artifact_rover_style(
                batch_fp32,
                enabled=bool(static_clutter_removal),
                percentile=clutter_percentile,
            )
            batch_scaled = batch_fp32 / np.float32(rda_scale_value)
            batch_fp16 = np.minimum(batch_scaled, np.float32(65504.0)).astype(np.float16)

            if rad_ds is None:
                rad_ds = fout.create_dataset(
                    "rad",
                    shape=(0, *batch_fp16.shape[1:]),
                    maxshape=(None, *batch_fp16.shape[1:]),
                    chunks=(1, *batch_fp16.shape[1:]),
                    dtype=np.float16,
                )
                first_rda_shape = (
                    int(batch_fp16.shape[1]),
                    int(batch_fp16.shape[2]),
                    int(batch_fp16.shape[3]),
                )

            count = int(batch_fp16.shape[0])
            rad_ds.resize((stored + count, *batch_fp16.shape[1:]))
            rad_ds[stored:stored + count] = batch_fp16
            timestamps.extend(pending_t)
            stored += count
            pending_rda.clear()
            pending_t.clear()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Processing frames", total=target_frames)
            for i, (frame, t) in enumerate(_iter_frames_from_packets_h5(h5_path, raw_shape)):
                if i % stride != 0:
                    continue

                rda_raw = process_frame(frame)
                if doppler_axis is None or int(doppler_axis.shape[0]) != int(rda_raw.shape[1]):
                    doppler_axis = make_doppler_axis_mps(cfg["radar"], rda_raw.shape[1], num_tx)
                speeds.append(
                    _estimate_speed_value(
                        rda=rda_raw,
                        doppler_axis=doppler_axis,
                        mode=speed_mode,
                        percentile=float(speed_percentile),
                        min_count=int(speed_min_count),
                    )
                )
                pending_rda.append(rda_raw)
                pending_t.append(float(t) if t is not None else np.nan)
                processed += 1
                if len(pending_rda) >= clutter_batch_frames:
                    _flush_pending()
                progress.advance(task_id, 1)

                if target_frames is not None and processed >= target_frames:
                    break

            _flush_pending()

        speed_arr = _to_float_array_strict(speeds, dtype=np.float32)
        t_arr = _to_float_array_strict(timestamps, dtype=np.float64)
        fout.create_dataset("speed", data=speed_arr)
        fout.create_dataset("t", data=t_arr)

    sensor_path = out_path.with_name("sensor.json")
    if first_rda_shape is None:
        first_rda_shape = (0, 0, 0)
    sensor_data = _make_sensor_intrinsics(cfg["radar"], first_rda_shape, num_tx=num_tx)
    with open(sensor_path, "w") as f:
        json.dump(sensor_data, f, indent=4)

    radar_json_path = out_path.with_name("radar.json")
    if radar_json_path.exists():
        with open(radar_json_path) as f:
            radar_meta = json.load(f)
            if not isinstance(radar_meta, dict):
                radar_meta = {}
    else:
        radar_meta = {}
    radar_meta["radar"] = cfg["radar"]
    radar_meta["rsp"] = {
        "name": rsp,
        "window": bool(apply_window),
        "size": {"azimuth": int(max(1, int(azimuth_bins)))},
        "backend": backend_name,
    }
    radar_meta["speed"] = {
        "mode": speed_mode,
        "percentile": float(speed_percentile),
        "min_count": int(speed_min_count),
    }
    radar_meta["clutter"] = {
        "enabled": bool(static_clutter_removal),
        "zero_doppler_bins": 1,
        "percentile": float(clutter_percentile),
        "batch_frames": int(clutter_batch_frames),
        "mode": "rover_like_remove_artifact",
    }
    radar_meta["rda_scale"] = float(rda_scale_value)
    if backend_name == "jax":
        radar_meta["rsp"]["jax"] = {"device": jax_device, "jit": bool(jax_jit)}
    radar_meta["input_h5"] = str(h5_path)
    radar_meta["processed_frames"] = int(len(timestamps))
    radar_meta["rda_dimensions"] = [int(len(timestamps)), *[int(v) for v in first_rda_shape]]
    with open(radar_json_path, "w") as f:
        json.dump(radar_meta, f, indent=4)

    print(f"H5 input: {h5_path}")
    print(f"RSP backend: {backend_name}")
    print(f"RDA scale: {rda_scale_value}")
    print(f"Output file: {out_path}")
    print(f"Sensor file: {sensor_path}")
    print(f"Radar metadata file: {radar_json_path}")
    print(f"Saved frames: {len(timestamps)}")


if __name__ == "__main__":
    tyro.cli(cli_main)
