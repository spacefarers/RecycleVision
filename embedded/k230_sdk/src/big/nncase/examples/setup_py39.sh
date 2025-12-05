#!/bin/bash

apt update
apt install -y python3.9 python3.9-venv
python3.9 -m venv ~/python39_venv
source ~/python39_venv/bin/activate
python3 -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install onnx==1.15.0 onnx-simplifier==0.4.36 onnxruntime==1.19.0 numpy==2.0.0 protobuf==4.25.8 Pillow==11.2.1
pip install ../pypi/nncase-2.10.0-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
pip install ../pypi/nncase_kpu-2.10.0-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
