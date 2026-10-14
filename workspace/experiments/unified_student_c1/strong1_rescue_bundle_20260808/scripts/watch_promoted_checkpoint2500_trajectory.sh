#!/usr/bin/env bash
set -euo pipefail

export CHECKPOINT_ROUND=2500
export SOURCE_C4_SUBDIR=local_round2500_c4
export TRAJECTORY_SUBDIR=local_round2500_trajectory
exec bash /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1/scripts/watch_promoted_checkpoint_trajectory_from_c4.sh
