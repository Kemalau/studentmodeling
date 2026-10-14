#!/usr/bin/env bash
set -euo pipefail

export PROFILE_READOUT_TRAIN_ENTRYPOINT=scripts/train_state8_profile_ownership_v5.py
export PROFILE_READOUT_OWNERSHIP_NEGATIVE_COUNT=8
export PROFILE_READOUT_OWNERSHIP_SCOPE=seen
export STATE8_V5_PROFILE_GENERATION_WEIGHT="${STATE8_V5_PROFILE_GENERATION_WEIGHT:-0.1}"
export STATE8_PROFILE_OWNERSHIP_SCOPE=seen
export STATE8_PROFILE_LEGACY_CAUSAL_ENABLED=0

test "$(sha256sum scripts/train_state8_profile_ownership_v5.py | cut -d' ' -f1)" = 8cbf1deaeb18f794529a5efcab8918ab3dcd70190dfd5a5657902b38be299000
test "$(sha256sum scripts/train_state8_profile_ownership_repair.py | cut -d' ' -f1)" = 2ef6e546dc3666ef52cbb016ffddba6e6f079f11bbbebba127cd14d8c8e87f49
test "$(sha256sum scripts/run_state8_student_profile_readout.sh | cut -d' ' -f1)" = 08c0599c476e3993dc7c3a1c52eb0a845b2f15f489a40bcaf654f43a0672413c

exec bash scripts/run_state8_student_profile_readout.sh
