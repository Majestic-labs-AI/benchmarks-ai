# Benchmarking Large Language Model (LLM) Artificial Intelligence (AI) Performance

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         pp512 |      8.95 ± 0.99 |
| Meta llama 3 8B F16            |  14.96 GiB |     8.03 B | CPU        |       4 |         tg128 |      2.15 ± 0.02 |

- We're using Georgi Gerganov's
[llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)
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
    - **pp**: "Prompt processing: processing a prompt in batches". Higher is better.
    - **tg**: "Text generation: generating a sequence of tokens". Higher is better.

Typical invocation to run benchmark against a model:

```bash
./llama-bench -m models/ggml-model-f16.gguf
```

To convert a model from Hugging Face to llama.cpp's `.gguf` format:

```bash
python convert_hf_to_gguf.py ~/workspace/Meta-Llama-3-8B/
```

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
gcloud config set project nono
gcloud compute instances create nvidia-l4 \
    --project=blabbertabber \
    --zone=northamerica-northeast2-a \
    --machine-type=g2-standard-8  \
    --maintenance-policy=TERMINATE --restart-on-failure \
    --network-interface=nic-type=GVNIC \
    --accelerator=type=nvidia-l4-vws,count=1 \
    --image-family=c0-deeplearning-common-cpu-v20230925-debian-10 \
    --image-project=ml-images \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd
```

Watch out! There not be any _g2-standard-8_ available in zone _us-central-1a_

```
A g2-standard-8 VM instance with 1 nvidia-l4-vws accelerator(s) is currently
unavailable in the us-central1-a zone. Alternatively, you can try your request
again with a different VM hardware configuration or at a later time. For more
information, see the troubleshooting documentation
```

```bash
gcloud compute machine-types list --zones us-central1-a --filter g2
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
