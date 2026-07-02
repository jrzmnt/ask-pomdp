#!/usr/bin/env bash
# Run all HigherLower ablations (threshold, MC samples, always-ask).
set -euo pipefail
cd "$(dirname "$0")/../.."

bash higher_lower/scripts/ablation_threshold.sh
bash higher_lower/scripts/ablation_mc_samples.sh
bash higher_lower/scripts/ablation_always_ask.sh

echo "========================================"
echo " All ablations done → higher_lower/results/"
echo "========================================"
