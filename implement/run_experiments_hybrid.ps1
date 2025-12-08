# run_experiments_hybrid.ps1

Write-Host "========================================================"
Write-Host " Ablation Study 실험 자동화 스크립트 시작 (Hybrid Models)"
Write-Host "시작 시간: $(Get-Date)"
Write-Host "========================================================"


# ----------------------------------------------------------------
# 5. Priority: Hybrid Rank
# ----------------------------------------------------------------
Write-Host ""
Write-Host "▶ [5/6] Priority: Hybrid Rank 실험 진행 중..."

Write-Host "  - Single Constraint..."
python.exe evaluate_add_mpe.py --mode single_constraint --constraints "budget,house_rule,people_number" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_use_priority_hybrid_rank" --no-memory --use-priority --no-self-eval --priority-type "hybrid_rank"

Write-Host "  - Second Constraints..."
python.exe evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_use_priority_hybrid_rank" --no-memory --use-priority --no-self-eval --priority-type "hybrid_rank"

Write-Host "  - Preference..."
python.exe evaluate_add_mpe.py --mode preference --budget_types "small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_use_priority_hybrid_rank" --no-memory --use-priority --no-self-eval --priority-type "hybrid_rank"


# ----------------------------------------------------------------
# 6. Priority: Hybrid Weight
# ----------------------------------------------------------------
Write-Host ""
Write-Host "▶ [6/6] Priority: Hybrid Weight 실험 진행 중..."

Write-Host "  - Single Constraint..."
python.exe evaluate_add_mpe.py --mode single_constraint --constraints "budget,house_rule,people_number" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_use_priority_hybrid_weight" --no-memory --use-priority --no-self-eval --priority-type "hybrid_weight"

Write-Host "  - Second Constraints..."
python.exe evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_use_priority_hybrid_weight" --no-memory --use-priority --no-self-eval --priority-type "hybrid_weight"

Write-Host "  - Preference..."
python.exe evaluate_add_mpe.py --mode preference --budget_types "small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_use_priority_hybrid_weight" --no-memory --use-priority --no-self-eval --priority-type "hybrid_weight"


Write-Host "========================================================"
Write-Host " 모든 실험이 완료되었습니다!"
Write-Host "종료 시간: $(Get-Date)"
Write-Host "========================================================"