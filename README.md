# Benchmarking Large Language Model (LLM) Artificial Intelligence (AI) Performance

## Benchmarks Using Georgi Gerganov's [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)

| model                          |       size |     params | backend    | threads |          test |    tokens/second |  notes |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: | ------ |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         pp512 |      8.95 ± 0.99 | [vSphere](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         tg128 |      2.15 ± 0.02 | [vSphere](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         pp512 |  1444.68 ± 60.51 | [vSphere](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg128 |     16.40 ± 0.04 | [vSphere](#vsphere) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         pp512 |     13.12 ± 0.10 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         tg128 |      2.91 ± 0.01 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         pp512 | 2270.05 ± 127.44 | [Google L4](#google_l4) |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CUDA       |      99 |         tg128 |     16.07 ± 0.03 | [Google L4](#google_l4) |

- _llama-bench_ benchmarks only one engine (executor); it doesn't benchmark,
for example, [Triton](https://github.com/triton-lang/triton)
- Headings
  - **model**: LLM Model, such as Meta's [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
  - **size**: (¿RAM footprint?), e.g. 14.96 GiB
  - **params**: number of the model's parameters, e.g. "8.03 B" (8 billion)
  - **backend**:
    - **CPU**: you don't want this ;-)
    - **GPU**: Graphic
  - **threads**: (?number of independent threads?) e.g. "4"
  - **test**: [type of test](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench) performed:
    - **pp**: "Prompt processing: processing a prompt in **batches**". Higher is better.
    - **tg**: "Text generation: generating a sequence of tokens". Higher is better.


Where `~/workspace/Meta-Llama-3-8B/` was downloaded/cloned from
<https://huggingface.co/meta-llama/Meta-Llama-3-8B>

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

### Setting up a Google L4

```bash
gcloud auth login
gcloud config set project blabbertabber
gcloud compute disks create benchmarks --size=300GB --type=pd-ssd --zone=northamerica-northeast2-a
gcloud compute instances create nvidia-l4 \
    --project=blabbertabber \
    --zone=northamerica-northeast2-a \
    --machine-type=g2-standard-8 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=787364477489-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-l4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=nvidia-l4,image=projects/ml-images/global/images/c1-deeplearning-tf-2-16-cu123-v20240708-debian-11-py310,mode=rw,size=200,type=projects/blabbertabber/zones/northamerica-northeast2-c/diskTypes/pd-ssd \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
gcloud compute instances attach-disk nvidia-l4 --disk=benchmarks --zone=northamerica-northeast2-a
gcloud compute ssh nvidia-l4 -- -A
```

The VM will ask us, "Would you like to install the Nvidia driver?", to which there's only one answer, "y".

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
 # We need Git LFS for some of the models, and I like fd-find
sudo apt-get install git-lfs fd-find
cd /mnt/work
git clone git@github.com:ggerganov/llama.cpp
  # Install the necessary Python libraries
pip install "huggingface_hub[cli]" matplotlib numpy pandas ffmpeg sentencepiece setuptools torch transformers triton
huggingface-cli login
  # Make sure to download LLaMA to the persistent disk, not the local SSD
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir Meta-Llama-3-8B
cd llama.cpp
  # We need to create the .gguf file for llama-bench
python convert_hf_to_gguf.py ../Meta-Llama-3-8B/
mv ../Meta-Llama-3-8B/ggml-model-f16.gguf models/Meta-Llama-3-8B.gguf
  # Compile llama-bench with CUDA
make GGML_CUDA=1 -j4
  # Run the benchmark
./llama-bench -m models/Meta-Llama-3-8B.gguf
```


## Setting up a Noble Numbat AI workstation

Install on Linux (macOS isn't able to install the Python triton library):

```bash
git clone git@github.com:majestic-labs-AI/benchmarks-ai.git
cd benchmarks-ai
python -m venv venv
. venv/bin/activate
pip install matplotlib numpy pandas ffmpeg setuptools torch transformers triton
```

### Troubleshooting

Watch out! There might not be any _g2-standard-8_ available in zone _us-central-1a_

```
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

<a name="vsphere">vSphere</a>

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


<!--
When I run my small `triton.py` code from the [tutorial](), I get the following error:

```
ModuleNotFoundError: No module named 'triton.language'; 'triton' is not a package
```

I think this was caused by `triton` version 3.0.0, but when I `pip install
triton` after I removed it I got version 2.3.1.

-->
