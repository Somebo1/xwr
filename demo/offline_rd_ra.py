"""Offline R-D and R-A plotting from packet HDF5."""

"""其功能是快速检查采集的数据是否正常"""

from pathlib import Path

import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml

import xwr
from xwr.rsp import iq_from_iiqq
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


def _extract_rx_comp(cfg_radar: dict) -> list[tuple[float, float]] | None:
    calibration = cfg_radar.get("calibration")
    if calibration is None:
        return None
    if isinstance(calibration, str):
        parts = calibration.strip().split()
        if not parts:
            return None
        if parts[0] == "compRangeBiasAndRxChanPhase":
            parts = parts[1:]
        if len(parts) < 3:
            return None
        vals = [float(v) for v in parts[1:]]
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    if isinstance(calibration, dict):
        cmd = calibration.get("comp_range_bias_and_rx_chan_phase")
        if isinstance(cmd, str) and cmd.strip():
            parts = cmd.strip().split()
            if parts and parts[0] == "compRangeBiasAndRxChanPhase":
                parts = parts[1:]
            if len(parts) < 3:
                return None
            vals = [float(v) for v in parts[1:]]
            return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        pairs = (
            calibration.get("rx_comp")
            or calibration.get("rx_coeffs")
            or calibration.get("rx_phase")
        )
        if pairs is None:
            return None
        out: list[tuple[float, float]] = []
        for pair in pairs:
            out.append((float(pair[0]), float(pair[1])))
        return out
    return None


def _extract_range_bias_m(cfg_radar: dict) -> float:
    calibration = cfg_radar.get("calibration")
    if calibration is None:
        return 0.0
    if isinstance(calibration, str):
        parts = calibration.strip().split()
        if not parts:
            return 0.0
        if parts[0] == "compRangeBiasAndRxChanPhase":
            parts = parts[1:]
        if not parts:
            return 0.0
        return float(parts[0])
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


def _rx_comp_matrix(
    rx_comp: list[tuple[float, float]] | None,
    num_tx: int,
    num_rx: int,
    device: str | None,
) -> np.ndarray | None:
    if not rx_comp:
        return None
    needed = num_tx * num_rx
    pairs = rx_comp
    device_name = "" if device is None else str(device).upper()
    if len(pairs) == needed:
        pass
    elif device_name == "AWR1843L" and num_tx == 2 and num_rx == 4 and len(pairs) >= 12:
        pairs = pairs[0:4] + pairs[8:12]
    elif len(pairs) > needed:
        pairs = pairs[:needed]
    else:
        return None
    coeff = np.asarray([complex(re, im) for re, im in pairs], dtype=np.complex64)
    return coeff.reshape(num_tx, num_rx)


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
    try:
        h5py = importlib.import_module("h5py")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'h5py'. Install project dependencies with uv sync."
        ) from e
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
        config = os.path.join(os.path.dirname(__file__), "config_awr1843l.yaml")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    radar_cfg = xwr.XWRConfig(**cfg["radar"])
    raw_shape = radar_cfg.raw_shape
    rsp_inst = getattr(xwr_rsp, rsp)(window=False, size={"azimuth": 128})
    num_tx = _infer_num_tx(rsp_inst, default=radar_cfg.num_tx)
    rx_comp = _extract_rx_comp(cfg["radar"])
    range_bias_m = _extract_range_bias_m(cfg["radar"])
    rx_comp_mat = _rx_comp_matrix(
        rx_comp,
        num_tx=radar_cfg.num_tx,
        num_rx=radar_cfg.num_rx,
        device=cfg["radar"].get("device"),
    )

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

        frame_for_rsp = frame
        if rx_comp_mat is not None and rsp_inst.SAMPLE_TYPE == "IQ":
            frame_for_rsp = iq_from_iiqq(frame).astype(np.complex64, copy=False)
            frame_for_rsp = frame_for_rsp * rx_comp_mat[None, :, :, None]
        dear = np.abs(rsp_inst(frame_for_rsp[None, ...]))
        rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)
        ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)

        n_range, n_dop = rd.shape
        _, n_az = ra.shape
        range_axis = make_range_axis_m(cfg["radar"], n_range) - range_bias_m
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
