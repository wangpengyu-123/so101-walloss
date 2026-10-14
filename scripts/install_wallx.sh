#!/bin/bash
source /root/autodl-tmp/wallx_ft/env/bin/activate
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export PATH=$PATH:/root/miniconda3/bin:/root/miniconda3/nvvm/bin
export CUDA_HOME=/root/miniconda3/targets/x86_64-linux
cd /root/autodl-tmp/wallx_ft/wall-x
rm -rf build
echo "PATH=$PATH"
echo "CUDA_HOME=$CUDA_HOME"
pip install -e . --no-build-isolation
