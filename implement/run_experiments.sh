#!/usr/bin/env bash
set -euo pipefail

# 리눅스(WSL)용 파이썬 선택
PYTHON_BIN="$(command -v python3 || command -v python)"
echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "========================================================"
echo " Ablation Study 실험 자동화 스크립트 시작"
echo "시작 시간: $(date)"
echo "========================================================"

# ----------------------------------------------------------------
# 1. Baseline (기준점)
# ----------------------------------------------------------------
echo ""
echo "▶ [1/6] Baseline 실험 진행 중..."

echo "  - Single Constraint..."
"${PYTHON_BIN}" evaluate_add_mpe.py --mode single_constraint --constraints "budget,cuisine,house_rule,people_number,room_type" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_baseline" --no-memory --no-priority --no-self-eval

echo "  - Second Constraints..."
"${PYTHON_BIN}" evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_baseline" --no-memory --no-priority --no-self-eval

echo "  - Preference..."
"${PYTHON_BIN}" evaluate_add_mpe.py --mode preference --budget_types "high,middle,small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_baseline" --no-memory --no-priority --no-self-eval

echo "========================================================"
echo " 모든 실험이 완료되었습니다!"
echo "종료 시간: $(date)"
echo "========================================================"
