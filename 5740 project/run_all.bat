@echo off
REM Complete pipeline script (Windows)

echo ========================================
echo AIMS5740 Project 1 - Complete Pipeline
echo ========================================

REM 激活虚拟环境（根据实际情况修改）
REM call venv\Scripts\activate

echo.
echo Step 1: Filter data...
echo ----------------------------------------
python scripts/01_filter_data.py

echo.
echo Step 2: SFT training...
echo ----------------------------------------
llamafactory-cli train configs/sft_config.yaml

echo.
echo Step 3: RL training...
echo ----------------------------------------
python scripts/03_rl_train.py

echo.
echo Step 4: Evaluation...
echo ----------------------------------------
python scripts/04_evaluate.py

echo.
echo ========================================
echo All done!
echo ========================================

pause
