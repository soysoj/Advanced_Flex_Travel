# run_experiments.ps1 (예시)
$ErrorActionPreference = "Stop"
$PY = "C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe"

Write-Host "Using Python: $PY"
& $PY --version

Write-Host "========================================================"
Write-Host " Ablation Study 실험 자동화 스크립트 시작"
Write-Host ("시작 시간: " + (Get-Date))
Write-Host "========================================================"

Write-Host ""
Write-Host "▶ [1/6] Baseline 실험 진행 중..."
Write-Host "  - Single Constraint..."

Write-Host "  - Preference..."
& $PY ".\evaluate_add_mpe.py" --mode preference --budget_types "small" --preference_types "cuisine,rating" --dataset_dir ".\agents\evaluation\database_with_ranks" --output_dir "results\preference_baseline" --no-memory --no-priority --no-self-eval

Write-Host "========================================================"
Write-Host " 모든 실험이 완료되었습니다!"
Write-Host ("종료 시간: " + (Get-Date))
Write-Host "========================================================"
