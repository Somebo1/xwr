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
from xwr.capture import types as capture_types

PACKET_WORDS = 728
PACKET_BYTES = PACKET_WORDS * 2


def _default_output_path() -> str:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + ".h5"
    return str(data_dir / name)


def _resolve_output_path(output: str | None) -> str:
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


def _import_h5py():
    import importlib

    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'h5py'. Install project dependencies with uv sync."
        ) from e


def _packet_dtype() -> np.dtype:
    dtype = np.dtype(
        [
            ("t", np.float64),
            ("packet_num", np.uint32),
            ("byte_count", np.uint64),
            ("packet_data", np.uint16, (PACKET_WORDS,)),
        ]
    )
    expected = ("t", "packet_num", "byte_count", "packet_data")
    actual = tuple(dtype.names or ())
    if actual != expected:
        raise ValueError(f"Unexpected packet fields: {actual} != {expected}")
    return dtype


def _payload_to_words(system: xwr.XWRSystem, payload: bytes | bytearray) -> np.ndarray:
    convert = getattr(system.dca, "payload_to_words", None)
    if callable(convert):
        return convert(payload)
    words = np.frombuffer(payload, dtype="<u2", count=len(payload) // 2)
    return words.astype(np.uint16, copy=False)


def _iter_packets(system: xwr.XWRSystem):
    recv = getattr(system.dca, "_recv", None)
    if callable(recv):
        while True:
            packet = recv()
            if packet is None:
                break
            yield packet
        return

    data_socket = getattr(system.dca, "data_socket", None)
    timeout = float(getattr(system.dca, "timeout", 1.0))
    max_packet = int(getattr(system.dca, "_MAX_PACKET_SIZE", 2048))
    if data_socket is None:
        raise AttributeError("DCA1000EVM does not expose a usable packet receive API.")

    deadline = time.perf_counter() + timeout
    while True:
        try:
            raw, _ = data_socket.recvfrom(max_packet)
            deadline = time.perf_counter() + timeout
            yield capture_types.DataPacket.from_bytes(raw)
        except BlockingIOError:
            if time.perf_counter() > deadline:
                break


def _prepare_capture(system: xwr.XWRSystem) -> None:
    system.dca.stop()
    system.dca.reset_ar_device()
    system.dca.flush()
    system.dca.start()
    system.xwr.setup(**system.config.as_dict())
    system.xwr.start()


def _collect_packets_h5(
    system: xwr.XWRSystem,
    output_path: str,
    *,
    max_packets: int | None,
    flush_every: int,
    log: logging.Logger,
) -> tuple[str, int, float]:
    h5py = _import_h5py()
    dtype = _packet_dtype()

    _prepare_capture(system)
    start = time.time()
    total = 0
    last_log_count = 0
    last_log_t = start

    with h5py.File(output_path, "w") as f:
        scan = f.create_group("scan")
        dset = scan.create_dataset(
            "packet", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
        )
        buf = np.empty((max(1, flush_every),), dtype=dtype)
        buf_i = 0

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
            for packet in _iter_packets(system):
                words = _payload_to_words(system, packet.data)
                if words.shape[0] != PACKET_WORDS:
                    fixed = np.zeros((PACKET_WORDS,), dtype=np.uint16)
                    n = min(PACKET_WORDS, int(words.shape[0]))
                    fixed[:n] = words[:n]
                    words = fixed

                buf[buf_i]["t"] = time.time()
                buf[buf_i]["packet_num"] = np.uint32(packet.sequence_number)
                buf[buf_i]["byte_count"] = np.uint64(packet.byte_count)
                buf[buf_i]["packet_data"] = words
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
                    mbps = pps * PACKET_BYTES * 8 / 1e6
                    log.info(
                        f"Captured {total} packets | rate={pps:.1f} pkt/s ({mbps:.2f} Mbps)"
                    )
                    last_log_t = now
                    last_log_count = total
        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt received, stopping capture.")
        finally:
            flush()
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
        f"[t(float64), packet_num(uint32), byte_count(uint64), packet_data(uint16[{PACKET_WORDS}])]"
    )
    log.info(f"Capture mode: max_packets={max_packets}, flush_every={flush_every}")

    awr = xwr.XWRSystem(**cfg)
    path, total_packets, duration = _collect_packets_h5(
        awr,
        out_path,
        max_packets=max_packets,
        flush_every=flush_every,
        log=log,
    )

    avg_pps = total_packets / duration if duration > 0 else 0.0
    avg_mbps = avg_pps * PACKET_BYTES * 8 / 1e6
    log.info(f"Total capture time: {duration:.3f}s")
    log.info(f"Total packets captured: {total_packets}")
    log.info(f"Average capture rate: {avg_pps:.1f} pkt/s ({avg_mbps:.2f} Mbps)")
    log.info(f"Wrote packet stream to: {path}")


if __name__ == "__main__":
    tyro.cli(cli_main)
