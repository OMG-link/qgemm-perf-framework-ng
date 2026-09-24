#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<EOF
Usage:
  $0 configure --platform PLATFORM [--cxx FILE] [--build-dir DIR]
  $0 build     --platform PLATFORM [--cxx FILE] [--build-dir DIR]
  $0 asm       --platform PLATFORM [--cxx FILE] [--build-dir DIR]
  $0 run       --platform PLATFORM --remote HOST [--build-dir DIR] [benchmark arguments]
  $0 test      --platform PLATFORM --remote HOST [--cxx FILE] [--build-dir DIR] [benchmark arguments]
  $0 clean     --platform PLATFORM [--build-dir DIR]

The compiler comes from --cxx or .pfng-workspace.json; the repository contains
no compiler path. The platform and remote have no defaults.
Currently supported platforms: spacemit-k1-a60, spacemit-k1-x60, spacemit-k3-a100.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
command=$1
shift

platform=
cxx=
remote=
build_dir=
workspace_config=${SCRIPT_DIR}/.pfng-workspace.json
benchmark_arguments=()

while (($#)); do
    case "$1" in
        --platform|--cxx|--remote|--build-dir)
            (($# >= 2)) || die "$1 requires a value"
            case "$1" in
                --platform) platform=$2 ;;
                --cxx) cxx=$2 ;;
                --remote) remote=$2 ;;
                --build-dir) build_dir=$2 ;;
            esac
            shift 2
            ;;
        --)
            shift
            benchmark_arguments+=("$@")
            break
            ;;
        *)
            if [[ ${command} == run || ${command} == test ]]; then
                benchmark_arguments+=("$1")
                shift
            else
                die "unknown option for ${command}: $1"
            fi
            ;;
    esac
done

[[ -n ${platform} ]] || die "--platform is required"
[[ ${platform} == spacemit-k1-a60 || ${platform} == spacemit-k1-x60 ||
   ${platform} == spacemit-k3-a100 ]] ||
    die "unsupported platform: ${platform}"
build_dir=${build_dir:-${SCRIPT_DIR}/build/${platform}}
binary=${build_dir}/pfng-bench

resolve_cxx() {
    if [[ -z ${cxx} && -f ${workspace_config} ]]; then
        local resolved
        resolved=$(python3 - "${workspace_config}" "${platform}" <<'PY'
import json
import os
from pathlib import Path
import sys

config_path = Path(os.path.abspath(Path(sys.argv[1]).expanduser()))
platform = sys.argv[2]
try:
    entry = json.loads(config_path.read_text())["platforms"][platform]
    value = entry.get("cxx", "")
except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
    raise SystemExit(f"cannot read cxx for {platform!r}: {exc}")

if not isinstance(value, str):
    raise SystemExit(f"platform entry {platform!r}: cxx must be a string")
if value:
    # Absolute-ize lexically only: resolving symlinks would rewrite a compiler
    # driver name such as clang++ into its versioned target (clang-23) and
    # change how the compiler is invoked.
    print(os.path.abspath(Path(value).expanduser()))
PY
    ) || die "failed to read cxx from ${workspace_config}"
        cxx=${resolved}
    fi
    if [[ -z ${cxx} ]]; then
        die "no compiler configured for ${platform}: pass --cxx FILE or add platforms.${platform}.cxx to ${workspace_config}"
    fi
    [[ -x ${cxx} ]] || die "compiler is not executable: ${cxx}"
}

configure() {
    resolve_cxx
    local -a cmake_args=(
        -S "${SCRIPT_DIR}"
        -B "${build_dir}"
        -G Ninja
        -DPFNG_TARGET_PLATFORM="${platform}"
        -DCMAKE_CXX_COMPILER="${cxx}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
    cmake "${cmake_args[@]}"
}

build() {
    configure
    cmake --build "${build_dir}" --target pfng-bench pfng-kernels
}

check_manifest() {
    [[ -x ${binary} ]] || die "binary does not exist; build it first: ${binary}"
    local manifest=${build_dir}/build-manifest.json
    [[ -f ${manifest} ]] || die "build manifest does not exist: ${manifest}"
    grep -Fq "\"platform\": \"${platform}\"" "${manifest}" ||
        die "build manifest does not match --platform ${platform}"
}

run_remote() {
    [[ -n ${remote} ]] || die "--remote is required"
    check_manifest
    local remote_dir remote_command threads=1 cpus index first_cpu last_cpu max_threads
    local prologue
    read -r first_cpu last_cpu max_threads < <(python3 - "${build_dir}/build-manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
print(manifest["first_cpu"], manifest["last_cpu"], manifest["max_threads"])
PY
)
    prologue=$(python3 - "${build_dir}/build-manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1])).get("remote_prologue", ""))
PY
)
    for ((index = 0; index < ${#benchmark_arguments[@]}; ++index)); do
        if [[ ${benchmark_arguments[index]} == --threads ]]; then
            ((index + 1 < ${#benchmark_arguments[@]})) || die "--threads requires a value"
            threads=${benchmark_arguments[index + 1]}
        fi
    done
    [[ ${threads} =~ ^[1-9][0-9]*$ ]] || die "--threads requires a positive integer"
    ((threads <= max_threads)) ||
        die "--threads must be at most ${max_threads} for ${platform}"
    cpus="${first_cpu}-$((first_cpu + threads - 1))"
    ((first_cpu + threads - 1 <= last_cpu)) ||
        die "thread CPU range ${cpus} exceeds ${platform} CPUs ${first_cpu}-${last_cpu}"
    remote_dir=$(ssh -o BatchMode=yes "${remote}" 'mktemp -d /tmp/pfng-bench.XXXXXX')
    trap 'ssh -o BatchMode=yes "${remote}" "rm -rf -- ${remote_dir}" >/dev/null 2>&1 || true' RETURN
    scp -q "${binary}" "${remote}:${remote_dir}/pfng-bench"
    ssh -o BatchMode=yes "${remote}" "${remote_dir}/pfng-bench --check-platform"
    printf -v remote_command 'env OMP_NUM_THREADS=%q OMP_DYNAMIC=FALSE taskset -c %q %q' \
        "${threads}" "${cpus}" "${remote_dir}/pfng-bench"
    for argument in "${benchmark_arguments[@]}"; do
        printf -v remote_command '%s %q' "${remote_command}" "${argument}"
    done
    # Platforms whose CPUs sit outside the default affinity mask need a command
    # in the same shell before taskset can reach them.
    [[ -n ${prologue} ]] && remote_command="${prologue}; ${remote_command}"
    ssh -o BatchMode=yes "${remote}" "${remote_command}"
}

case "${command}" in
    configure) configure ;;
    build) build ;;
    asm)
        configure
        cmake --build "${build_dir}" --target assembly
        ;;
    run) run_remote ;;
    test)
        build
        if ((${#benchmark_arguments[@]} == 0)); then
            benchmark_arguments=(--kernel all --m 24 --n 32 --k 256 --samples 3 --iterations 10)
        fi
        run_remote
        ;;
    clean) rm -rf -- "${build_dir}" ;;
    *) usage >&2; exit 2 ;;
esac