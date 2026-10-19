#!/bin/bash

validate_positive_integer() {
  local value=${1:-}
  local label=${2:-value}
  if [[ ! ${value} =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "${label} must be a positive integer" >&2
    return 2
  fi
}

validate_refinement_source_loading() {
  local allocated_cpus=${1:-}
  local workers=${2:-}
  local prefetch_sources=${3:-}
  local prefetch_batches=${4:-}
  if [[ ! ${allocated_cpus} =~ ^[0-9]+$ || ! ${workers} =~ ^[0-9]+$ || ! ${prefetch_sources} =~ ^[0-9]+$ || ! ${prefetch_batches} =~ ^[0-9]+$ ]]; then
    echo "source-loading resource values must be non-negative integers" >&2
    return 2
  fi
  if (( allocated_cpus <= 0 || workers > allocated_cpus || prefetch_sources <= 0 || prefetch_batches <= 0 )); then
    echo "workers must fit allocated CPUs and prefetch counts must be positive" >&2
    return 2
  fi
}
