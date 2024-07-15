# Benchmarking AI Performance

### Quick Start

Install on Linux (macOS isn't able to install the Python triton library):

```bash
git clone git@github.com:majestic-labs-AI/benchmarks-ai.git
cd benchmarks-ai
python -m venv venv
. venv/bin/activate
pip install matplotlib numpy pandas ffmpeg setuptools torch transformers triton
python bin/torch-answer.py
```

The last line of the output looks like this:

```json
{"device": "cpu", "tokens_per_second": 681}
```

Which means, "I used the device _cpu_, not _cuda_, (i.e. I ran on the Intel
CPU, not the NVIDIA GPU) to run the benchmark, and the resulting throughput
was 681 tokens/second."

### Troubleshooting

When seeing this error:

```
RuntimeError: Found no NVIDIA driver on your system.
```

```
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
lspci | grep -i nvidia
  03:00.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
ubuntu-drivers devices
  nvidia-driver-545
sudo ubuntu-drivers install
```

And when installing NVIDIA driver 545 fails with the following because
you're on Ubuntu Noble Numbat 24.04 and the kernel is too new and has
removed the variable that the NVIDIA driver expects:

```
dpkg: dependency problems prevent configuration of nvidia-driver-545:
 nvidia-driver-545 depends on nvidia-dkms-545 (<= 545.29.06-1); however:
  Package nvidia-dkms-545 is not configured yet.
 nvidia-driver-545 depends on nvidia-dkms-545 (>= 545.29.06); however:
  Package nvidia-dkms-545 is not configured yet.
```

```bash
sudo nvim /usr/src/linux-headers-6.8.0-36/include/drm/drm_ioctl.h
```

```C
#define DRM_UNLOCKED 0
```

```bash
sudo shutdown -r now
```

<!--
When I run my small `triton.py` code from the [tutorial](), I get the following error:

```
ModuleNotFoundError: No module named 'triton.language'; 'triton' is not a package
```

I think this was caused by `triton` version 3.0.0, but when I `pip install
triton` after I removed it I got version 2.3.1.

-->
