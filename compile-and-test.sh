#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly TOOLCHAIN_DIR=${TOOLCHAIN_DIR:-/home/linkai/projects/triton-riscv/llvm-project-spacemit-x60/builders/x86-riscv/llvm/bin}
readonly CXX=${CXX:-${TOOLCHAIN_DIR}/clang++}
readonly DEPLOYMENT_SERVER=${DEPLOYMENT_SERVER:-spacemit-k1}
readonly BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/build}
readonly BIN=${BUILD_DIR}/ime-llama-bench

configure() {
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

build() {
    configure
    cmake --build "${BUILD_DIR}" --target ime-llama-bench ime-experiments
}

emit_assembly() {
    configure
    cmake --build "${BUILD_DIR}" --target assembly
}

run_remote() {
    local remote_dir
    remote_dir=$(ssh "${DEPLOYMENT_SERVER}" 'mktemp -d /tmp/ime-llama.XXXXXX')
    trap 'ssh "${DEPLOYMENT_SERVER}" "rm -rf -- ${remote_dir}" >/dev/null 2>&1 || true' RETURN
    scp -q "${BIN}" "${DEPLOYMENT_SERVER}:${remote_dir}/ime-llama-bench"
    ssh "${DEPLOYMENT_SERVER}" "taskset -c 0 '${remote_dir}/ime-llama-bench' $*"
}

usage() {
    cat <<EOF
Usage:
  $0 configure
  $0 build
  $0 run  <M> <N> <K>
  $0 test [M N K]
  $0 asm
  $0 clean

Environment overrides: TOOLCHAIN_DIR, CXX, DEPLOYMENT_SERVER, BUILD_DIR.
EOF
}

command=${1:-test}
case "${command}" in
    configure)
        configure
        ;;
    build)
        build
        ;;
    run)
        shift
        [[ $# -eq 3 ]] || { usage >&2; exit 2; }
        [[ -x "${BIN}" ]] || build
        run_remote "$@"
        ;;
    test)
        shift || true
        [[ $# -eq 0 || $# -eq 3 ]] || { usage >&2; exit 2; }
        build
        if [[ $# -eq 0 ]]; then
            run_remote 12 32 32
        else
            run_remote "$@"
        fi
        ;;
    asm)
        emit_assembly
        ;;
    clean)
        rm -rf -- "${BUILD_DIR}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
