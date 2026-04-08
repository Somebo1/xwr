"""Simple range-doppler and range-azimuth visualization demo (with physical axes)."""

import logging
import os

import numpy as np
import tyro
import yaml
from matplotlib import pyplot as plt
from rich.logging import RichHandler

import xwr
from xwr.rsp import iq_from_iiqq
from xwr.rsp import numpy as xwr_rsp


C0 = 299_792_458.0  # speed of light (m/s)


def _to_hz_sample_rate(sample_rate: float) -> float:
    """TI config often uses ksps (e.g. 2500 -> 2.5e6)."""
    return sample_rate * 1e3 if sample_rate < 1e5 else sample_rate


def _to_hz_per_s_slope(freq_slope: float) -> float:
    """TI config often uses MHz/us (e.g. 67.012 -> 67.012e12 Hz/s)."""
    return freq_slope * 1e12 if freq_slope < 1e6 else freq_slope


def _to_seconds_time(t: float) -> float:
    """TI config often uses microseconds (e.g. 331 -> 331e-6 s)."""
    return t * 1e-6 if t > 1.0 else t


def _default_rsp_for_device(device: str | None) -> str:
    device_name = "" if device is None else str(device).upper()
    if device_name in {"AWR1843L", "AWR1642"}:
        return "AWR1642Boost"
    if device_name in {"AWR1843AOP", "AWR1843AOPEVM"}:
        return "AWR1843AOP"
    if device_name in {"AWR2944", "AWR2944EVM"}:
        return "AWR2944EVM"
    if device_name in {"AWRL6844", "AWRL6844EVM"}:
        return "AWRL6844EVM"
    return "AWR1843Boost"


def _resolve_rsp_name(rsp: str | None, device: str | None) -> str:
    fallback = _default_rsp_for_device(device)
    if rsp is None:
        return fallback
    device_name = "" if device is None else str(device).upper()
    if rsp == "AWR1843Boost" and device_name in {"AWR1843L", "AWR1642"}:
        return fallback
    return rsp


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


def _build_rsp_instance(rsp: str, device: str | None):
    ctor = getattr(xwr_rsp, rsp, None)
    if ctor is not None:
        return rsp, ctor(window=False, size={"azimuth": 128})

    fallback = _default_rsp_for_device(device)
    ctor = getattr(xwr_rsp, fallback, None)
    if ctor is None:
        raise ValueError(f"Unknown RSP class: {rsp}")
    return fallback, ctor(window=False, size={"azimuth": 128})


def make_range_axis_m(cfg_radar: dict, n_range: int) -> np.ndarray:
    fs = _to_hz_sample_rate(float(cfg_radar["sample_rate"]))
    slope = _to_hz_per_s_slope(float(cfg_radar["freq_slope"]))
    # r[k] = c * (k/N)*Fs / (2*slope)
    k = np.arange(n_range, dtype=np.float64)
    return (C0 * fs / (2.0 * slope * n_range)) * k


def make_doppler_axis_mps(cfg_radar: dict, n_doppler: int, num_tx: int) -> np.ndarray:
    fc = float(cfg_radar["frequency"]) * 1e9  # GHz -> Hz
    lam = C0 / fc

    idle = _to_seconds_time(float(cfg_radar["idle_time"]))
    ramp = _to_seconds_time(float(cfg_radar["ramp_end_time"]))
    t_chirp = idle + ramp
    t_eff = t_chirp * max(1, int(num_tx))  # TDM: same-TX repetition interval

    fd = np.fft.fftshift(np.fft.fftfreq(n_doppler, d=t_eff))  # Hz, monotonic after shift
    v = (lam / 2.0) * fd
    return v


def make_azimuth_axis_deg(n_az: int, d_over_lambda: float = 0.5) -> np.ndarray:
    # spatial freq in cycles/element, shifted to monotonic increasing
    fs = np.fft.fftshift(np.fft.fftfreq(n_az, d=1.0))
    sin_theta = fs / float(d_over_lambda)  # since (lambda/d) = 1/(d/lambda)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.degrees(np.arcsin(sin_theta))
    return theta


def cli_main(
    config: str | None = None,
    rsp: str | None = None,
    device: str | None = None,
    verbose: int = 20,
):
    logging.basicConfig(
        level=verbose,
        format="%(name)-12s  %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler()],
    )
    log = logging.getLogger("XWRDemo")

    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    if device is not None:
        cfg["radar"]["device"] = device
    rsp = _resolve_rsp_name(rsp, cfg["radar"].get("device"))

    plt.ion()
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title("Range–Doppler")
    axs[1].set_title("Range–Azimuth")

    fig.tight_layout()

    awr = xwr.XWRSystem(**cfg)
    rsp, rsp_inst = _build_rsp_instance(rsp, cfg["radar"].get("device"))
    num_tx = awr.config.num_tx
    fallback_rsp = _default_rsp_for_device(cfg["radar"].get("device"))
    range_bias_m = _extract_range_bias_m(cfg["radar"])
    rx_comp = _extract_rx_comp(cfg["radar"])
    rx_comp_mat = _rx_comp_matrix(
        rx_comp,
        num_tx=awr.config.num_tx,
        num_rx=awr.config.num_rx,
        device=cfg["radar"].get("device"),
    )
    if range_bias_m != 0.0:
        log.info(
            f"Applying range bias offset on displayed range axis: {range_bias_m:+.4f} m"
        )
    if rx_comp_mat is not None:
        log.info(
            f"Applying RX channel compensation in Python RSP for {rx_comp_mat.shape[0]}x{rx_comp_mat.shape[1]} channels."
        )

    im1 = None
    im2 = None
    range_axis = None
    doppler_axis = None
    az_axis = None

    try:
        for frame in awr.dstream(numpy=True):
            frame_for_rsp = frame
            if rx_comp_mat is not None and rsp_inst.SAMPLE_TYPE == "IQ":
                frame_for_rsp = iq_from_iiqq(frame).astype(np.complex64, copy=False)
                frame_for_rsp = frame_for_rsp * rx_comp_mat[None, :, :, None]
            try:
                dear = np.abs(rsp_inst(frame_for_rsp[None, ...]))
            except ValueError as e:
                if "Expected (tx, rx)=" not in str(e) or rsp == fallback_rsp:
                    raise
                log.warning(
                    f"RSP {rsp} does not match radar data shape, falling back to {fallback_rsp}."
                )
                rsp = fallback_rsp
                _, rsp_inst = _build_rsp_instance(rsp, cfg["radar"].get("device"))
                dear = np.abs(rsp_inst(frame_for_rsp[None, ...]))

            # RD: mean over (batch, elevation, azimuth) -> [doppler, range] then transpose -> [range, doppler]
            rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)

            # RA: mean over (batch, doppler, elevation) -> [azimuth, range] then transpose -> [range, azimuth]
            ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)

            # Shift to make x-axes monotonic: Doppler & Azimuth
            # xwr_rsp 很可能已经对 Doppler/Azimuth 做过 shift（0 在中间），这里不要再 shift
            rd_disp = rd
            ra_disp = ra


            if im1 is None:
                n_range, n_dop = rd_disp.shape
                _, n_az = ra_disp.shape

                range_axis = make_range_axis_m(cfg["radar"], n_range) - range_bias_m
                doppler_axis = make_doppler_axis_mps(cfg["radar"], n_dop, num_tx=num_tx)
                az_axis = make_azimuth_axis_deg(n_az, d_over_lambda=0.5)

                im1 = axs[0].imshow(
                    rd_disp,
                    cmap="viridis",
                    aspect="auto",
                    origin="lower",
                    extent=[doppler_axis[0], doppler_axis[-1], range_axis[0], range_axis[-1]],
                )
                axs[0].set_xlabel("Radial velocity (m/s)")
                axs[0].set_ylabel("Range (m)")
                fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

                im2 = axs[1].imshow(
                    ra_disp,
                    cmap="viridis",
                    aspect="auto",
                    origin="lower",
                    extent=[az_axis[0], az_axis[-1], range_axis[0], range_axis[-1]],
                )
                axs[1].set_xlabel("Azimuth (deg)")
                axs[1].set_ylabel("Range (m)")
                fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

                fig.tight_layout()
            else:
                im1.set_data(rd_disp)
                im1.set_clim(vmin=float(np.min(rd_disp)), vmax=float(np.max(rd_disp)))

                im2.set_data(ra_disp)
                im2.set_clim(vmin=float(np.min(ra_disp)), vmax=float(np.max(ra_disp)))

            plt.pause(0.001)

    except KeyboardInterrupt:
        log.warning("Demo interrupted by user.")
    finally:
        awr.stop()
        plt.ioff()


if __name__ == "__main__":
    tyro.cli(cli_main)
