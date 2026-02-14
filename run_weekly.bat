@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ============================================================
echo 週次運用自動化 (PDCA) 開始
echo   抽出 → マスタ更新 → 再学習 → 予測 → 予測ログ追記 → 精度検証
echo ============================================================
echo.

python run_weekly_automation.py
set EXIT_CODE=%ERRORLEVEL%

echo.
rem 最新のHTMLレポートをブラウザで開く
set "REPORT_DIR=%~dp0jv_data\reports"
if exist "%REPORT_DIR%" (
    for /f "delims=" %%f in ('dir /b /o-d "%REPORT_DIR%\report_*.html" 2^>nul') do (
        echo レポートを開いています: %%f
        start "" "%REPORT_DIR%\%%f"
        goto :opened
    )
)
echo レポートが見つかりませんでした。
:opened

rem 精度検証サマリがあれば案内
if exist "%~dp0jv_data\reports\evaluation_summary.csv" (
    echo.
    echo 精度検証: jv_data\reports\evaluation_summary.csv
    echo 予測ログ: jv_data\history\prediction_log.csv
)

echo.
echo ============================================================
echo GitHub へ自動デプロイ (docs 等をプッシュ)
echo ============================================================
git add .
git commit -m "Auto-update: Weekly Prediction %date%"
if %ERRORLEVEL% equ 0 (
    git push origin main
    if %ERRORLEVEL% equ 0 (
        echo プッシュ完了。GitHub Pages が更新されます。
    ) else (
        echo プッシュに失敗しました。リモート設定・認証を確認してください。
    )
) else (
    echo 変更がありませんでした。またはコミットに失敗しました。
)

echo.
echo ============================================================
echo 実行完了。任意のキーを押すと終了します。
echo ============================================================
pause
exit /b %EXIT_CODE%