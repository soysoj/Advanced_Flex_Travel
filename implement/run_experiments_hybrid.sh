echo "========================================================"
echo " Ablation Study 실험 자동화 스크립트 시작"
echo "시작 시간: $(date)"
echo "========================================================"

# ----------------------------------------------------------------
# 5. Priority: Hybrid Rank
# ----------------------------------------------------------------
echo ""
echo "▶ [5/6] Priority: Hybrid Rank 실험 진행 중..."

echo "  - Single Constraint..."
python.exe evaluate_add_mpe.py --mode single_constraint --constraints "budget,cuisine,house_rule,people_number,room_type" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_use_priority_hybrid_rank" --no-memory --use-priority --no-self-eval --priority-type "hybrid_rank"

echo "  - Second Constraints..."
python.exe evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_use_priority_hybrid_rank" --no-memory --use-priority --no-self-eval --priority-type "hybrid_rank"

echo "  - Preference..."
python.exe evaluate_add_mpe.py --mode preference --budget_types "high,middle,small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_use_priority_hybrid_rank" --no-memory --use-priority --no-self-eval --priority-type "hybrid_rank"


# ----------------------------------------------------------------
# 6. Priority: Hybrid Weight
# ----------------------------------------------------------------
echo ""
echo "▶ [6/6] Priority: Hybrid Weight 실험 진행 중..."

echo "  - Single Constraint..."
python.exe evaluate_add_mpe.py --mode single_constraint --constraints "budget,cuisine,house_rule,people_number,room_type" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_use_priority_hybrid_weight" --no-memory --use-priority --no-self-eval --priority-type "hybrid_weight"

echo "  - Second Constraints..."
python.exe evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_use_priority_hybrid_weight" --no-memory --use-priority --no-self-eval --priority-type "hybrid_weight"

echo "  - Preference..."
python.exe evaluate_add_mpe.py --mode preference --budget_types "high,middle,small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_use_priority_hybrid_weight" --no-memory --use-priority --no-self-eval --priority-type "hybrid_weight"


echo "========================================================"
echo " 모든 실험이 완료되었습니다!"
echo "종료 시간: $(date)"
echo "========================================================"