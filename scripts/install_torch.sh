#!/bin/bash
source /root/autodl-tmp/wallx_ft/env/bin/activate
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export LC_ALL=C.UTF-8 LANG=C.UTF-8
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
