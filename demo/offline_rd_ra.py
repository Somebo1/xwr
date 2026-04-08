"""Offline R-D and R-A plotting from packet HDF5."""

from pathlib import Path

import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml

import xwr
from xwr.rsp import numpy as xwr_rsp

C0 = 299_792_458.0


def _to_hz_sample_rate(sample_rate: float) -> float:
    return sample_rate * 1e3 if sample_rate < 1e5 else sample_rate


def _to_hz_per_s_slope(freq_slope: float) -> float:
    return freq_slope * 1e12 if freq_slope < 1e6 else freq_slope


def _to_seconds_time(t: float) -> float:
    return t * 1e-6 if t > 1.0 else t


def _infer_num_tx(rsp_inst, default: int = 2) -> int:
    for name in ("num_tx", "n_tx", "tx", "ntx"):
        if hasattr(rsp_inst, name):
            v = getattr(rsp_inst, name)
            if isinstance(v, (int, np.integer)) and int(v) > 0:
                return int(v)
    return default


def make_range_axis_m(cfg_radar: dict, n_range: int) -> np.ndarray:
    fs = _to_hz_sample_rate(float(cfg_radar["sample_rate"]))
    slope = _to_hz_per_s_slope(float(cfg_radar["freq_slope"]))
    k = np.arange(n_range, dtype=np.float64)
    return (C0 * fs / (2.0 * slope * n_range)) * k


def make_doppler_axis_mps(
    cfg_radar: dict, n_doppler: int, num_tx: int
) -> np.ndarray:
    fc = float(cfg_radar["frequency"]) * 1e9
    lam = C0 / fc
    idle = _to_seconds_time(float(cfg_radar["idle_time"]))
    ramp = _to_seconds_time(float(cfg_radar["ramp_end_time"]))
    t_chirp = idle + ramp
    t_eff = t_chirp * max(1, int(num_tx))
    fd = np.fft.fftshift(np.fft.fftfreq(n_doppler, d=t_eff))
    return (lam / 2.0) * fd


def make_azimuth_axis_deg(
    n_az: int, d_over_lambda: float = 0.5
) -> np.ndarray:
    fs = np.fft.fftshift(np.fft.fftfreq(n_az, d=1.0))
    sin_theta = np.clip(fs / float(d_over_lambda), -1.0, 1.0)
    return np.degrees(np.arcsin(sin_theta))


def _resolve_h5_path(h5_file: str | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    if h5_file is not None:
        p = Path(h5_file)
        return p if p.is_absolute() else data_dir / p
    files = sorted(data_dir.glob("*.h5"), key=lambda x: x.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")
    return files[-1]


def _iter_frames_from_packets_h5(h5_path: Path, raw_shape: tuple[int, ...]):
    h5py = importlib.import_module("h5py")
    frame_size = int(np.prod(raw_shape)) * np.dtype(np.uint16).itemsize
    with h5py.File(str(h5_path), "r") as f:
        packets = f["scan"]["packet"]
        offset = 0
        buf = bytearray()
        for rec in packets:
            byte_count = int(rec["byte_count"])
            payload = np.asarray(rec["packet_data"], dtype=np.uint16).tobytes()

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
                frame = np.frombuffer(frame_bytes, dtype=np.int16).reshape(
                    raw_shape
                )
                yield frame


def cli_main(
    h5_file: str | None = None,
    config: str | None = None,
    rsp: str = "AWR1642Boost",
    out_dir: str | None = None,
    sample_every: int = 20,
    max_outputs: int = 10,
) -> None:
    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    radar_cfg = xwr.XWRConfig(**cfg["radar"])
    raw_shape = radar_cfg.raw_shape
    rsp_inst = getattr(xwr_rsp, rsp)(window=False, size={"azimuth": 128})
    num_tx = _infer_num_tx(rsp_inst, default=radar_cfg.num_tx)

    h5_path = _resolve_h5_path(h5_file)
    if out_dir is None:
        out_root = Path(__file__).resolve().parents[1] / "data"
        out_dir_path = out_root / f"{h5_path.stem}_rd_ra"
    else:
        p = Path(out_dir)
        out_dir_path = p if p.is_absolute() else (Path.cwd() / p)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, frame in enumerate(_iter_frames_from_packets_h5(h5_path, raw_shape)):
        if i % max(1, sample_every) != 0:
            continue

        dear = np.abs(rsp_inst(frame[None, ...]))
        rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)
        ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)

        n_range, n_dop = rd.shape
        _, n_az = ra.shape
        range_axis = make_range_axis_m(cfg["radar"], n_range)
        doppler_axis = make_doppler_axis_mps(cfg["radar"], n_dop, num_tx)
        az_axis = make_azimuth_axis_deg(n_az, d_over_lambda=0.5)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].set_title("Range–Doppler")
        axs[1].set_title("Range–Azimuth")

        im1 = axs[0].imshow(
            rd,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[
                doppler_axis[0],
                doppler_axis[-1],
                range_axis[0],
                range_axis[-1],
            ],
        )
        axs[0].set_xlabel("Radial velocity (m/s)")
        axs[0].set_ylabel("Range (m)")
        fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

        im2 = axs[1].imshow(
            ra,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[az_axis[0], az_axis[-1], range_axis[0], range_axis[-1]],
        )
        axs[1].set_xlabel("Azimuth (deg)")
        axs[1].set_ylabel("Range (m)")
        fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        out_file = out_dir_path / f"frame_{i:06d}.png"
        fig.savefig(out_file, dpi=120)
        plt.close(fig)

        saved += 1
        if saved >= max(1, max_outputs):
            break

    print(f"H5 input: {h5_path}")
    print(f"Output dir: {out_dir_path}")
    print(f"Saved images: {saved}")


if __name__ == "__main__":
    tyro.cli(cli_main)
