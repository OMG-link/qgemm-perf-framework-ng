#!/usr/bin/env bash
# 独立构建 K3 / A100 微基准。这些程序不依赖 pfng 的 CMake 工程，
# 直接用 K3 分支的 clang++ 交叉编译成静态可执行文件。
set -euo pipefail

CXX=${CXX:-/home/linkai/projects/triton-riscv/llvm-project-spacemit-x60/builders/x86-k3/llvm/bin/clang++}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-$HERE/out}
mkdir -p "$OUT"

# -mcpu=spacemit-a100 打开 VLEN=1024 与 XSMTVDotII。
# main 线程启动时还在 X100 上（VLEN=256），因此所有用到 A100 专属
# 指令或 1024 位 VLEN 的代码都必须在绑定之后才执行。
FLAGS=(
  -O2 -static -std=c++20
  -mcpu=spacemit-a100
  -mllvm -riscv-v-vector-bits-min=1024
  -mllvm -riscv-v-vector-bits-max=1024
  -I"$HERE"
)

for src in "$@"; do
  name=$(basename "$src" .cpp)
  echo "[build] $name"
  "$CXX" "${FLAGS[@]}" -o "$OUT/$name" "$HERE/$name.cpp"
done
