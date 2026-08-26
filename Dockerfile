# Base image: Ubuntu 24.04 LTS. CUDA 12.6.3 is the oldest/most mature CUDA line NVIDIA
# ships a devel image for on 24.04 (no 11.8/12.1 image exists for this OS, unlike the
# 22.04 base this project originally targeted per reactree_README.md). "devel" (not
# "runtime") because some python packages (e.g. bitsandbytes) need build tools available
# at install time.
# NOTE: torch==2.3.1 (pinned below) only supports up to CUDA 12.1 — bump the torch/
# torchvision/torchaudio versions to a build compatible with cu126 before re-enabling
# the pip install below.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies:
#   - build-essential/libffi-dev: needed to build some python packages from source
#   - xserver-xorg/xserver-xorg-video-fbdev/xauth/mesa-utils/nvidia-xconfig: headless
#     display (X server) required by ai2thor/ALFRED and VirtualHome, which render
#     through Unity; nvidia-xconfig generates the X config (per alfred/README.md's
#     `nvidia-xconfig -a --use-display-device=None --virtual=1280x1024` step)
#   - pciutils: provides `lspci`, used by alfred/scripts/startx.py to find the GPU's
#     PCI bus ID when generating the X config
#   - x11-utils: provides `xdpyinfo`, which ai2thor's Controller.check_x_display()
#     uses to validate DISPLAY before launching Unity. Without it, that check silently
#     no-ops and a wrong/unset DISPLAY hangs Unity instead of raising a clear error.
#   - libsm6/libxext6/libxrender-dev: runtime libs Unity/OpenCV expect to find
#   - software-properties-common: needed to add the deadsnakes PPA below
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      curl wget git unzip ffmpeg p7zip-full sudo \
      build-essential libffi-dev cmake \
      libsm6 libxext6 libxrender-dev \
      xserver-xorg xserver-xorg-video-fbdev xauth mesa-utils nvidia-xconfig \
      pciutils x11-utils \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships Python 3.12 by default; reactree_README.md calls for 3.8. We install
# 3.11 via deadsnakes above and isolate it in its own virtualenv — confirmed compatible
# with the pinned torch==2.13.0 below.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# PyTorch is installed separately first (pinned to match the CUDA version above and to
# each other), per reactree_README.md, so that requirements.txt — which doesn't pin
# torch/torchvision/torchaudio at all — can't pull in an unpinned, mismatched pair as
# transitive deps (this happened once: torch 2.12.1 + torchvision 0.28.0, which don't
# match, raised "RuntimeError: operator torchvision::nms does not exist"). Versions below
# were resolved together in one pip pass against the cu126 index, confirmed compatible
# (import + torch.cuda.is_available() both succeed).
RUN pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu126

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
