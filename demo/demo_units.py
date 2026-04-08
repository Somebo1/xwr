"""Simple range-doppler and range-azimuth visualization demo (with physical axes)."""

import logging
import os

import numpy as np
import tyro
import yaml
from matplotlib import pyplot as plt
from rich.logging import RichHandler

import xwr
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


def _infer_num_tx(rsp_inst, default: int = 2) -> int:
    """Try to infer TDM TX count from RSP instance; fall back to default."""
    for name in ("num_tx", "n_tx", "tx", "ntx"):
        if hasattr(rsp_inst, name):
            v = getattr(rsp_inst, name)
            if isinstance(v, (int, np.integer)) and int(v) > 0:
                return int(v)
    return default


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
    rsp: str = "AWR1843Boost",
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

    plt.ion()
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title("Range–Doppler")
    axs[1].set_title("Range–Azimuth")

    fig.tight_layout()

    awr = xwr.XWRSystem(**cfg)
    rsp_inst = getattr(xwr_rsp, rsp)(window=False, size={"azimuth": 128})

    num_tx = _infer_num_tx(rsp_inst, default=2)  # AWR1843Boost 常见为 2Tx TDM（匹配你图里的 ~1.2 m/s）

    im1 = None
    im2 = None
    range_axis = None
    doppler_axis = None
    az_axis = None

    try:
        for frame in awr.dstream(numpy=True):
            # dear: [batch, doppler, elevation, azimuth, range]
            dear = np.abs(rsp_inst(frame[None, ...]))

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

                range_axis = make_range_axis_m(cfg["radar"], n_range)
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
        awr.stop()


if __name__ == "__main__":
    tyro.cli(cli_main)
