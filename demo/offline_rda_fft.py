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
from xwr.rsp import numpy as xwr_rsp

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
) -> None:
    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config_awr1843l.yaml")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    radar_cfg = xwr.XWRConfig(**cfg["radar"])
    raw_shape = radar_cfg.raw_shape
    rsp_inst = getattr(xwr_rsp, rsp)(
        window=bool(apply_window), size={"azimuth": max(1, int(azimuth_bins))}
    )
    num_tx = _infer_num_tx(rsp_inst, default=radar_cfg.num_tx)

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
        stride = max(1, int(sample_every))
        first_rda_shape: tuple[int, int, int] | None = None
        total_frames = _count_frames_from_packets_h5(h5_path, raw_shape)
        target_frames = (total_frames + stride - 1) // stride
        if max_frames is not None:
            target_frames = min(target_frames, int(max_frames))

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

                dear = np.abs(rsp_inst(frame[None, ...]))[0]
                rda = np.transpose(np.mean(dear, axis=1), (2, 0, 1)).astype(np.float16)
                if first_rda_shape is None:
                    first_rda_shape = tuple(int(v) for v in rda.shape)
                doppler_axis = make_doppler_axis_mps(cfg["radar"], rda.shape[1], num_tx)
                doppler_power = np.mean(rda.astype(np.float32), axis=(0, 2))
                speeds.append(float(doppler_axis[int(np.argmax(doppler_power))]))

                if rad_ds is None:
                    rad_ds = fout.create_dataset(
                        "rad",
                        shape=(0, *rda.shape),
                        maxshape=(None, *rda.shape),
                        chunks=(1, *rda.shape),
                        dtype=np.float16,
                    )

                rad_ds.resize((stored + 1, *rda.shape))
                rad_ds[stored] = rda
                timestamps.append(float(t) if t is not None else np.nan)
                stored += 1
                progress.advance(task_id, 1)

                if stored >= target_frames:
                    break

        fout.create_dataset("speed", data=np.asarray(speeds, dtype=np.float32))
        fout.create_dataset("t", data=np.asarray(timestamps, dtype=np.float64))

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
    }
    radar_meta["input_h5"] = str(h5_path)
    radar_meta["processed_frames"] = int(len(timestamps))
    radar_meta["rda_dimensions"] = [int(len(timestamps)), *[int(v) for v in first_rda_shape]]
    with open(radar_json_path, "w") as f:
        json.dump(radar_meta, f, indent=4)

    print(f"H5 input: {h5_path}")
    print(f"Output file: {out_path}")
    print(f"Sensor file: {sensor_path}")
    print(f"Radar metadata file: {radar_json_path}")
    print(f"Saved frames: {len(timestamps)}")


if __name__ == "__main__":
    tyro.cli(cli_main)
