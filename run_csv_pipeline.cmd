@echo off
chcp 65001 >nul
cd /d "%~dp0"

set BUSINESS=%USERPROFILE%\Downloads\businass_csv.csv
set CONSUMER=%USERPROFILE%\Downloads\cumsumer_csv.csv
set MERCHANT=%USERPROFILE%\Downloads\merchants_ref.csv

echo Business: %BUSINESS%
echo Consumer: %CONSUMER%
echo Merchant: %MERCHANT%

py -3 scripts\mastercard_hidden_entrepreneur_pipeline.py ^
  --business-path "%BUSINESS%" ^
  --consumer-path "%CONSUMER%" ^
  --merchant-path "%MERCHANT%" ^
  --output-dir mastercard_hidden_entrepreneur_artifacts

echo.
echo Готово:
echo   submission.csv
echo   pattern_consumers.csv  - паттерные consumer-карты
echo   consumer_scores_ranked.csv - все consumer со скором
pause
