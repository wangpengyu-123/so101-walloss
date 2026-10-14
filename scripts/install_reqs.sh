#!/bin/bash
source /root/autodl-tmp/wallx_ft/env/bin/activate
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export LC_ALL=C.UTF-8 LANG=C.UTF-8
cd /root/autodl-tmp/wallx_ft/wall-x
grep -vE "^(torch|torchvision|torchaudio)==" requirements.txt > /tmp/reqs_no_torch.txt
pip install -r /tmp/reqs_no_torch.txt --index-url https://pypi.org/simple
