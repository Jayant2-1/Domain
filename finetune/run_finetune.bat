@echo off
REM ──────────────────────────────────────────────────────────────
REM run_finetune.bat — End-to-end LoRA fine-tuning pipeline (Windows)
REM
REM Steps:
REM   1. Export positive interactions from SQLite → JSONL
REM   2. Train LoRA adapter on the exported data
REM   3. (Optional) Merge adapter into base model
REM
REM Usage:
REM   finetune\run_finetune.bat
REM ──────────────────────────────────────────────────────────────

if "%MLML_DB_PATH%"=="" set MLML_DB_PATH=data\mlml.db
if "%MAX_STEPS%"=="" set MAX_STEPS=200
if "%MERGE%"=="" set MERGE=0

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set dt=%%I
set TIMESTAMP=%dt:~0,8%_%dt:~8,6%

set DATA_PATH=finetune\data\train_%TIMESTAMP%.jsonl
set ADAPTER_PATH=adapters\v_%TIMESTAMP%
set MERGED_PATH=models\merged_%TIMESTAMP%

echo === Step 1: Export positive interactions ===
python -m finetune.prepare_data --db-path "%MLML_DB_PATH%" --output "%DATA_PATH%" --min-feedback 1

if not exist "%DATA_PATH%" (
    echo No training data exported. Aborting.
    exit /b 0
)

echo.
echo === Step 2: LoRA fine-tuning ===
python -m finetune.train_lora --data "%DATA_PATH%" --output "%ADAPTER_PATH%" --max-steps %MAX_STEPS% --batch-size 1 --grad-accum 4

echo.
echo Adapter saved to: %ADAPTER_PATH%

if "%MERGE%"=="1" (
    echo.
    echo === Step 3: Merge adapter ===
    python -m finetune.merge_adapters --adapter "%ADAPTER_PATH%" --output "%MERGED_PATH%"
    echo Merged model saved to: %MERGED_PATH%
)

echo.
echo === Fine-tuning pipeline complete ===
echo To load the new adapter, set MLML_ADAPTER_DIR=%ADAPTER_PATH% and restart.
