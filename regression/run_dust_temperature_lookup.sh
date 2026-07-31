#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="${repo_dir}/src"
test_binary="$(mktemp "${TMPDIR:-/tmp}/radmc3d-dusttemp-test.XXXXXX")"
trap 'rm -f "${test_binary}"' EXIT

make -C "${src_dir}" radmc3d

objects=()
for object in "${src_dir}"/*.o; do
    if [[ "$(basename "${object}")" != "main.o" ]]; then
        objects+=("${object}")
    fi
done

gfortran -O2 -fopenmp -I"${src_dir}" \
    "${repo_dir}/regression/test_dust_temperature_lookup.f90" \
    "${objects[@]}" -o "${test_binary}"
"${test_binary}"
