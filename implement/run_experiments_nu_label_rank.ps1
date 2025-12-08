# run_experiments.ps1

Write-Host "========================================================"
Write-Host " Ablation Study 실험 자동화 스크립트 시작"
Write-Host "시작 시간: $(Get-Date)"
Write-Host "========================================================"


# ----------------------------------------------------------------
# 2. Priority: Numerical
# ----------------------------------------------------------------
Write-Host ""
Write-Host "▶ [2/6] Priority: Numerical 실험 진행 중..."

Write-Host "  - Single Constraint..."
python.exe evaluate_add_mpe.py --mode single_constraint --constraints "budget,house_rule,people_number" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_use_priority_numerical" --no-memory --use-priority --no-self-eval --priority-type "numerical"

Write-Host "  - Second Constraints..."
python.exe evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_use_priority_numerical" --no-memory --use-priority --no-self-eval --priority-type "numerical"

Write-Host "  - Preference..."
python.exe evaluate_add_mpe.py --mode preference --budget_types "small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_use_priority_numerical" --no-memory --use-priority --no-self-eval --priority-type "numerical"


# ----------------------------------------------------------------
# 3. Priority: Rank Only
# ----------------------------------------------------------------
Write-Host ""
Write-Host "▶ [3/6] Priority: Rank Only 실험 진행 중..."

Write-Host "  - Single Constraint..."
python.exe evaluate_add_mpe.py --mode single_constraint --constraints "budget,house_rule,people_number" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/single_constraint_use_priority_rank" --no-memory --use-priority --no-self-eval --priority-type "rank_only"

Write-Host "  - Second Constraints..."
python.exe evaluate_add_mpe.py --mode two_constraints --constraint_pairs "global_local,local_global" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/second_constraints_use_priority_rank" --no-memory --use-priority --no-self-eval --priority-type "rank_only"

Write-Host "  - Preference..."
python.exe evaluate_add_mpe.py --mode preference --budget_types "small" --preference_types "cuisine,rating" --dataset_dir "./agents/evaluation/database_with_ranks" --output_dir "results/preference_use_priority_rank" --no-memory --use-priority --no-self-eval --priority-type "rank_only"


Write-Host "========================================================"
Write-Host " 모든 실험이 완료되었습니다!"
Write-Host "종료 시간: $(Get-Date)"
Write-Host "========================================================"