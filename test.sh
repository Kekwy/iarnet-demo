#!/bin/bash
set -e  # 任何命令失败则立即退出

pip install -r requirements.txt
python "$1"  # 使用第一个参数作为 Python 文件名