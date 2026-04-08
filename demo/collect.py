"""Raw radar packet capture to HDF5."""

from datetime import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np

import tyro
import yaml
from rich.logging import RichHandler

import xwr


def _default_output_path() -> str:
    """Return the default HDF5 output path under `<repo>/data/`.

    The filename uses local wall-clock time with second precision, e.g.
    `2026.04.08-14.32.10.h5`.
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + ".h5"
    return str(data_dir / name)


def _resolve_output_path(output: str | None) -> str:
    """Resolve and prepare the final output path for captured packets.

    Args:
        output: User-provided path. If `None`, a timestamped default path is
            generated under `<repo>/data/`.

    Returns:
        Absolute or relative-resolved path where the HDF5 file should be
        written. Parent directory is created if needed.
    """
    if output is None:
        return _default_output_path()
    path = Path(output)
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / path)


def _collect_packets_h5(
    system: xwr.XWRSystem,
    output_path: str,
    *,
    max_packets: int | None,
    flush_every: int,
    log: logging.Logger,
) -> tuple[str, int, float]:
    """Capture raw DCA packets and persist them into an HDF5 dataset.

    Packet stream is written to `/scan/packet` with a structured dtype:
    `(t, packet_num, byte_count, packet_data)`.

    Args:
        system: Initialized radar system object.
        output_path: Destination HDF5 file path.
        max_packets: Optional hard stop for total packet count.
        flush_every: Number of buffered packets before flushing to disk.
        log: Logger used for periodic progress reporting.

    Returns:
        Tuple of `(output_path, total_packets, duration_seconds)`.
    """
    import importlib

    h5py = importlib.import_module("h5py")
    dtype = np.dtype(
        [
            ("t", np.float64),
            ("packet_num", np.uint32),
            ("byte_count", np.uint64),
            ("packet_data", np.uint16, (728,)),
        ]
    )

    expected = ("t", "packet_num", "byte_count", "packet_data")
    actual = tuple(dtype.names or ())
    if actual != expected:
        raise ValueError(f"Unexpected packet fields: {actual} != {expected}")

    system.dca.stop()
    system.dca.reset_ar_device()
    system.dca.flush()
    system.dca.start()
    system.xwr.setup(**system.config.as_dict())
    system.xwr.start()

    start = time.time()
    f = h5py.File(output_path, "w")
    scan = f.create_group("scan")
    dset = scan.create_dataset(
        "packet", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
    )
    buf = np.empty((max(1, flush_every),), dtype=dtype)
    buf_i = 0
    total = 0
    last_log_count = 0
    last_log_t = start

    def flush() -> None:
        nonlocal buf_i
        if buf_i <= 0:
            return
        begin = dset.shape[0]
        dset.resize((begin + buf_i,))
        dset[begin : begin + buf_i] = buf[:buf_i]
        f.flush()
        buf_i = 0

    try:
        for packet in system.dca.packets():
            buf[buf_i]["t"] = time.time()
            buf[buf_i]["packet_num"] = np.uint32(packet.sequence_number)
            buf[buf_i]["byte_count"] = np.uint64(packet.byte_count)
            buf[buf_i]["packet_data"] = system.dca.payload_to_words(packet.data)
            buf_i += 1
            total += 1

            if buf_i >= buf.shape[0]:
                flush()
            if max_packets is not None and total >= max_packets:
                break

            now = time.time()
            if now - last_log_t >= 5.0:
                dt = now - last_log_t
                delta = total - last_log_count
                pps = delta / dt if dt > 0 else 0.0
                mbps = pps * 1456 * 8 / 1e6
                log.info(
                    f"Captured {total} packets | rate={pps:.1f} pkt/s "
                    f"({mbps:.2f} Mbps)"
                )
                last_log_t = now
                last_log_count = total
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt received, stopping capture.")
    finally:
        flush()
        f.close()
        system.stop()

    duration = max(0.0, time.time() - start)
    return output_path, total, duration


def cli_main(
    config: str | None = None,
    device: str | None = None,
    output: str | None = None,
    max_packets: int | None = None,
    flush_every: int = 1024,
    verbose: int = 20,
) -> None:
    """Raw packet capture demo that writes DCA stream to an HDF5 file.

    If `config` is not provided, defaults to `demo/config.yaml`. You can set
    `device` to override the radar device in configuration.

    Args:
        config: Path to radar configuration YAML.
        device: Optional radar device override (e.g. `AWR1843`).
        output: Output HDF5 path. If omitted, auto-saves to `<repo>/data/`.
        max_packets: Optional packet limit; `None` means capture until manual
            interruption.
        flush_every: Number of packets buffered before writing to disk.
        verbose: Logging verbosity (10-debug, 20-info, 30-warning, 40-error).
    """
    logging.basicConfig(
        level=verbose,
        format="%(name)-12s  %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler()],
    )
    log = logging.getLogger("XWRCollect")

    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config_awr1843l.yaml")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    if device is not None:
        cfg["radar"]["device"] = device

    out_path = _resolve_output_path(output)
    log.info(f"Output path: {out_path}")
    log.info(
        "H5 schema: /scan/packet fields "
        "[t(float64), packet_num(uint32), byte_count(uint64), packet_data(uint16[728])]"
    )
    log.info(
        f"Capture mode: max_packets={max_packets}, flush_every={flush_every}"
    )

    awr = xwr.XWRSystem(**cfg)
    path, total_packets, duration = _collect_packets_h5(
        awr,
        out_path,
        max_packets=max_packets,
        flush_every=flush_every,
        log=log,
    )
    avg_pps = total_packets / duration if duration > 0 else 0.0
    avg_mbps = avg_pps * 1456 * 8 / 1e6
    log.info(f"Total capture time: {duration:.3f}s")
    log.info(f"Total packets captured: {total_packets}")
    log.info(f"Average capture rate: {avg_pps:.1f} pkt/s ({avg_mbps:.2f} Mbps)")
    log.info(f"Wrote packet stream to: {path}")


if __name__ == "__main__":
    tyro.cli(cli_main)
