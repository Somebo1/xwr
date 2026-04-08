# `xwr`: Linux-Compatible Real-Time Raw Data Capture for TI mmWave Radars

[![pypi version](https://img.shields.io/pypi/v/xwr.svg)](https://pypi.org/project/xwr/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xwr)
![License - MIT](https://img.shields.io/badge/license-MIT-green)
![PyPI - Types](https://img.shields.io/pypi/types/xwr)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![CI](https://github.com/RadarML/xwr/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/xwr/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/xwr)

`xwr` is a pure-python, linux-based real time raw data capture system for TI mmWave radars, and includes four key components:

- [`xwr`](https://radarml.github.io/xwr/system/): a high-level data capture interface
- [`xwr.rsp`](https://radarml.github.io/xwr/rsp/rsp/): a radar signal processing library with Numpy, Pytorch, and Jax support
- [`xwr.radar`](https://radarml.github.io/xwr/radar/api/): a parameterized python interface for the default radar firmware
- [`xwr.capture`](https://radarml.github.io/xwr/dca/api/): a pure-python, real-time interface for the DCA1000EVM

See our [documentation site](https://radarml.github.io/xwr/) for more details, setup guides, the included demo, and more!

## Requirements

`xwr` assumes a linux-based system and radar hardware which consists of the DCA1000EVM and a supported TI mmWave Radar (XWR) development board.

> [!IMPORTANT] 
> Supported Devices:
>    - AWR1843 Family: AWR1843Boost, AWR1843AOPEVM
>    - AWR1642Boost
>    - AWR2944EVM
>    - AWRL6844EVM
>
> WIP:
>    - AWR2544LOPEVM

## Install

The `xwr` library can be installed from pypi or github:

```sh
pip install xwr
# or
pip install git+ssh://github.com/RadarML/xwr.git
```

> [!WARNING]
> `xwr` does not include a copy of `torch` or `jax` by default! You must specify your own dependency and/or use the `xwr[torch]` and `xwr[jax]` extras if you intend to use these backends for the radar signal processing (`xwr.rsp`) submodule.

## Setup Policy for This Fork

Please follow the official `xwr` documentation for hardware deployment and software configuration without deviation:

- [xwr documentation](https://radarml.github.io/xwr/)
- [Hardware and environment setup](https://radarml.github.io/xwr/setup/)

This fork is designed to stay compatible with the upstream setup workflow. The additional scripts below are intended to be used after completing the official installation and device bring-up steps.

## Additions in This Fork

On top of the original `xwr` project, this fork adds the following utilities:

1. `demo/collect.py`: raw packet capture to `radarpackets.h5` format for DART integration.
   - Default config is `demo/config_awr1843l.yaml`.
   - Radar profile is configured as **2 TX / 4 RX** to align with the DART pipeline.
2. `demo/offline_rd_ra.py`: offline data inspection after collection.
   - Input: captured raw packet HDF5 file.
   - Output: sampled-frame Range-Doppler (RD) and Range-Azimuth (RA) visualizations.
3. `demo/demo_units.py`: enhanced live demo with physical unit labels for faster operational checks and status interpretation.

## CLI Quick Start

```sh
uv run demo/collect.py
uv run demo/offline_rd_ra.py --h5-file radarpackets_name.h5 --rsp AWR1642Boost --sample-every 20 --max-outputs 10
```

## Calibration via Config (No API Hardcoding)

This fork supports passing radar calibration through the radar config file, instead of hardcoding calibration values in `xwr/radar/api.py`.

- Supported for `AWR1843`, `AWR1843L`, `AWR1642`, and `AWR2944`.
- The calibration entry is loaded from `radar.calibration` and applied to `compRangeBiasAndRxChanPhase` during `setup(...)`.
- You can provide either:
  - a full command string (`comp_range_bias_and_rx_chan_phase`), or
  - structured fields (`range_bias` + `rx_comp` pairs).

Example (`demo/config_awr1843l.yaml`):

```yaml
radar:
  device: AWR1843L
  calibration:
    comp_range_bias_and_rx_chan_phase: "compRangeBiasAndRxChanPhase -0.0071049 -0.34449 -0.74618 -0.23290 -0.97250 -0.23366 -0.87015 -0.33699 -0.78156 -0.22025 -0.77553 -0.18965 -0.85011 -0.17263 -0.82861 -0.16805 -0.79882 -0.17939 -0.63912 -0.03873 -0.77678 -0.14587 -0.70194 -0.19648 -0.63314"
```

Run with explicit config:

```sh
uv run demo/collect.py --config demo/config_awr1843l.yaml
uv run demo/demo_units.py --config demo/config_awr1843l.yaml
```
