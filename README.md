# Benchmarking Large Language Model (LLM) Artificial Intelligence (AI) Performance

## Benchmarks Using Georgi Gerganov's [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)

| model                          |       size |     params | backend    | threads / graphic layers|          test |    tokens/second |  notes |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: | ------ |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg128 |     51.22 ± 0.25 | [Google A100](#google_a100) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg256 |     51.02 ± 0.31 | [Google A100](#google_a100) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg512 |     51.02 ± 0.02 | [Google A100](#google_a100) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg128 |     16.15 ± 0.01 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg256 |     16.10 ± 0.01 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg512 |     16.00 ± 0.01 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg128 |     16.41 ± 0.04 | [vSphere T4](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg256 |     16.18 ± 0.09 | [vSphere T4](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg512 |      9.01 ± 5.78 | [vSphere T4](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | Metal      |      99 |         tg128 |      5.26 ± 0.15 | [MacBook Air](#macbook_air) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | Metal      |      99 |         tg256 |      4.24 ± 0.11 | [MacBook Air](#macbook_air) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | Metal      |      99 |         tg512 |      4.16 ± 0.15 | [MacBook Air](#macbook_air) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         tg128 |      2.91 ± 0.01 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         tg128 |      2.15 ± 0.02 | [vSphere T4](#vsphere) |

- Headings (from the [documentation](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench))
  - **model**: LLM Model, such as Meta's [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
  - **size**: (¿RAM footprint?), e.g. 14.96 GiB
  - **params**: number of the model's parameters, e.g. "8.03 B" (8 billion)
  - **backend**:
    - **CPU**: you don't want this ;-)
    - **CUDA**: NVIDIA graphics card
    - **Metal** macOS GPU
  - **threads / graphics layers**: Number of threads (if CPU), number of graphics layers (if GPU). Specify eight threads by passing `-t 8`
  - **test**: [type of test](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench) performed:
    - **pp**: "Prompt processing: processing a prompt in batches". Higher is better. Disable by passing `-p 0`
    - **tg**: "Text generation: generating a sequence of tokens". Higher is better. To benchmark 128 (default), 256, and 512-token lengths, pass `-n 128,256,512`

### Disclaimers

- _llama-bench_ benchmarks only one engine (executor); it doesn't benchmark,
for example, [Triton](https://github.com/triton-lang/triton)
- Not all tokens are created equal. For example, a LLaMA 3 token is [worth 1.12 - 1.51](https://www.baseten.co/blog/comparing-tokens-per-second-across-llms/#3500573-making-clear-tps-comparisons-for-open-source-llms) LLaMA 2 tokens. One user suggested using "[number of characters per second](https://www.reddit.com/r/LocalLLaMA/comments/167cf4x/comment/jyp3h9s/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"

Where `~/workspace/Meta-Llama-3-8B/` was downloaded/cloned from
<https://huggingface.co/meta-llama/Meta-Llama-3-8B>

### Setting up a Google L4

```bash
gcloud auth login
gcloud config set project majestic-labs-ai
gcloud compute disks create benchmarks --size=300GB --type=pd-ssd --zone=northamerica-northeast2-a
gcloud compute instances create nvidia-l4 \
    --project=majestic-labs-ai \
    --zone=northamerica-northeast2-a \
    --machine-type=g2-standard-8 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-l4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=nvidia-l4,image=projects/ml-images/global/images/c1-deeplearning-tf-2-16-cu123-v20240708-debian-11-py310,mode=rw,size=200,type=projects/majestic-labs-ai/zones/northamerica-northeast2-c/diskTypes/pd-ssd \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
gcloud compute instances attach-disk nvidia-l4 --disk=benchmarks --zone=northamerica-northeast2-a
gcloud compute ssh nvidia-l4 -- -A
```

The VM will ask us, "Would you like to install the Nvidia driver?", to which there's only one answer, "y".

Remember to destroy it when done:

```bash
gcloud compute instances delete nvidia-l4
```

### Setting up a Google A100

We need to do this in my personal account because the Majestic account doesn't have quota. Also, we switched regions from Toronto to Idaho because Toronto didn't have any A100s (and Idaho didn't have any L4s)

```bash
gcloud compute instances create nvidia-a100 \
    --project=blabbertabber \
    --zone=us-central1-f \
    --machine-type=a2-highgpu-1g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    --create-disk=auto-delete=yes,boot=yes,device-name=nvidia-a100,image=projects/ml-images/global/images/c1-deeplearning-tf-2-16-cu123-v20240708-debian-11-py310,mode=rw,size=200,type=projects/blabbertabber/zones/us-central1-a/diskTypes/pd-ssd \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
```

```bash
gcloud compute instances delete nvidia-a100
```

### Setting Up the Persistent Disk

```bash
lsblk
sudo parted /dev/nvme0n2
  mklabel gpt
  mkpart primary ext4 0% 100%
  quit
sudo mkfs.ext4 /dev/nvme0n2p1
sudo mkdir /mnt/work
sudo mount /dev/nvme0n2p1 /mnt/work
sudo chown $USER /mnt/work
```

Let's download [Meta's LLMaMA 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and benchmark it:

```bash
 # We need Git LFS for some of the models, and I like fd-find, and clang is better than gcc
sudo apt-get install git-lfs fd-find clang cmake
cd /mnt/work
git clone git@github.com:ggerganov/llama.cpp
  # Install the necessary Python libraries
pip install torch torchvision torchaudio # --index-url https://download.pytorch.org/whl/cu121
pip install "huggingface_hub[cli]" matplotlib numpy pandas ffmpeg sentencepiece setuptools transformers triton
huggingface-cli login
  # Make sure to download LLaMA to the persistent disk, not the local SSD
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir Meta-Llama-3-8B
cd llama.cpp
  # We need to create the .gguf file for llama-bench
python convert_hf_to_gguf.py ../Meta-Llama-3-8B/
mv ../Meta-Llama-3-8B/ggml-model-f16.gguf models/Meta-Llama-3-8B.gguf
  # Compile llama-bench with CUDA
cmake -B build -DGGML_CUDA=ON # for macOS, "cmake -B build"
cmake --build build --config Release -j
  # Run the benchmark
build/bin/llama-bench -m models/Meta-Llama-3-8B.gguf -p 0 -n 128,256,512
```

### Setting up the NVIDIA T4 vSphere Noble Numbat AI workstation

Install on Linux (macOS isn't able to install the Python triton library):

```bash
git clone git@github.com:majestic-labs-AI/benchmarks-ai.git
cd benchmarks-ai
python -m venv venv
. venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ffmpeg matplotlib numpy pandas setuptools tabulate transformers triton
  # PyTorch benchmark!
cd ~/workspace
git clone https://github.com/pytorch/benchmark
cd benchmark
python3 install.py
```

Let's run some PyTorch benchmarks

```bash
pytest test_bench.py --ignore_machine_config
```

### Does `tmpfs` improve our benchmarks? Are our tests disk-bound?

Using `tmpfs` did not improve our benchmarks (our measurement with `tmpfs`
clocked in at 16.42 tokens-per-second, and the non-tmpfs runs were [16.40,
16.43, and 16.38])

Using `tmpfs` did not improve the time of the second-and-subsequent benchmarks.

`llama-bench` is disk bound on the first run, but it doesn't significantly
affect the measured tokens per second. The run time dropped 30% (62 seconds to
43 seconds) between the first and subsequent runs, but the tokens-per-second
generation only increased a negligible 0.1% (16.40 t/s to 16.43 t/s). I
speculate that the increase in run times was due to the 15 GiB model file being
cached in the kernel's page cache.

Running our tests after a fresh boot showed a 62-second run dropping
to 43 seconds on subsequent runs:

```bash
time build/bin/llama-bench -m models/Meta-Llama-3-8B.gguf -p 0 -v
  # 41.62s user 5.72s system 76% cpu 1:02.11 total
  # tokens per second was 16.40 ± 0.04
  # Let's try a second run:
time build/bin/llama-bench -m models/Meta-Llama-3-8B.gguf -p 0 -v
  # 41.88s user 1.33s system 100% cpu 43.197 total
  # tokens per second was 16.43 ± 0.04
  # Let's try a third run:
time build/bin/llama-bench -m models/Meta-Llama-3-8B.gguf -p 0 -v
  # 42.10s user 1.34s system 100% cpu 43.428 total
  # tokens per second was 16.38 ± 0.03
```

Let's try `tmpfs` (after bumping the RAM to 32 → 64 GiB to avoid inadvertently introducing another constraint)

```bash
sudo mkdir -p /mnt/tmpfs
sudo mount -t tmpfs -o size=24G tmpfs /mnt/tmpfs
sudo chmod 1777 /mnt/tmpfs
cp ~/workspace/llama.cpp/models/Meta-Llama-3-8B.gguf /mnt/tmpfs/ # 15G model
df -h /mnt/tmpfs
  # Filesystem      Size  Used Avail Use% Mounted on
  # tmpfs            24G   15G  9.1G  63% /mnt/tmpfs
time build/bin/llama-bench -m /mnt/tmpfs/Meta-Llama-3-8B.gguf -p 0 -v
  # 41.85s user 1.37s system 100% cpu 43.209 total
  # tokens per second was 16.42 ± 0.03
```

### Troubleshooting

Watch out! There might not be any _g2-standard-8_ available in zone _us-central-1a_

```text
A g2-standard-8 VM instance with 1 nvidia-l4-vws accelerator(s) is currently
unavailable in the us-central1-a zone. Alternatively, you can try your request
again with a different VM hardware configuration or at a later time. For more
information, see the troubleshooting documentation
```

```bash
gcloud compute machine-types list --zones us-central1-a --filter g2
```

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

### Footnotes

<a name="vsphere">vSphere T4</a>

- Linux distribution: Ubuntu Noble Numbat 24.04 (“cat /etc/lsb-release”)
- Linux kernel: 6.8.0-38-generic (“uname -a”)
- Hardware (virtual machine):
  - CPU: Intel(R) Xeon(R) D-1736NT CPU @ 2.70GHz (“cat /proc/cpuinfo”)
  - 4 cores
  - 32 GiB RAM (“htop”)
- NVIDIA Tesla T4 16 GiB TU104GL (“nvidia-smi”, “lspci | grep -i nvidia”)
- NVIDIA drivers 545.29.06 (“nvidia-smi”)
- CUDA Version: 12.3 (“nvidia-smi”)
- Python 3.12.3 (“python --version”)

<a name="google_l4">Google L4</a>

- Linux distribution: Debian GNU/Linux 11 (bullseye) (“cat /etc/os-release”)
- Linux kernel: 5.10.0-30-cloud-amd64 (“uname -a”)
- Hardware (virtual machine):
  - CPU: Intel(R) Xeon(R) CPU @ 2.20GHz (“cat /proc/cpuinfo”)
  - 8 cores
  - 32 GiB RAM (“htop”)
- NVIDIA Tesla L4 24 GiB (“nvidia-smi”, “lspci | grep -i nvidia”)
- NVIDIA drivers 550.90.07 (“nvidia-smi”)
- CUDA Version: 12.4 (“nvidia-smi”)
- Python 3.12.3 (“python --version”)

<a name="google_a100">Google A100</a>

- Linux distribution: Debian GNU/Linux 11 (bullseye) (“cat /etc/os-release”)
- Linux kernel: 5.10.0-30-cloud-amd64 (“uname -a”)
- Hardware (virtual machine):
  - CPU: Intel(R) Xeon(R) CPU @ 2.20GHz (“cat /proc/cpuinfo”)
  - 12 cores
  - 80 GiB RAM (“htop”)
- NVIDIA NVIDIA A100-SXM4-40GB (“nvidia-smi”, “lspci | grep -i nvidia”)
- NVIDIA drivers 550.90.07 (“nvidia-smi”)
- CUDA Version: 12.4 (“nvidia-smi”)
- Python 3.12.3 (“python --version”)

<a name="macbook_air">MacBook Air</a>

- macOS Sonoma 14.5
- Hardware:
  - CPU: Apple M2
  - 8 cores
  - 24 GiB
- GPU 8 cores

<!--
When I run my small `triton.py` code from the [tutorial](), I get the following error:

```
ModuleNotFoundError: No module named 'triton.language'; 'triton' is not a package
```

I think this was caused by `triton` version 3.0.0, but when I `pip install
triton` after I removed it I got version 2.3.1.

Here's an example, run from within the _llama.cpp_ repo:

```bash
. ~/workspace/benchmarks-ai/venv/bin/activate
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download deepset/roberta-base-squad2 --local-dir models/roberta-base-squad2
python convert_hf_to_gguf.py models/roberta-base-squad2
```

It appears that the conversion is [only for  LLaMA
models](https://github.com/ggerganov/llama.cpp/discussions/2948#discussioncomment-6925099),
which torpedoes my hope of using `llama-bench` as a golden standard.

-->
