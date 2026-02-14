@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  週末自動収支計算バッチ (run_weekend.bat)
REM
REM  使い方:
REM    ダブルクリック           → 予測JSONの日付で収支計算
REM    run_weekend.bat 20260214 → 指定日の収支計算
REM    run_weekend.bat auto     → pause なし (タスクスケジューラー用)
REM ============================================================

REM --- コードページを UTF-8 に ---
chcp 65001 > nul 2>&1

REM --- バッチ自身のディレクトリに移動 ---
cd /d "%~dp0"
if errorlevel 1 (
    echo [NG] ディレクトリ移動に失敗しました: %~dp0
    pause
    exit /b 1
)

REM --- calc_roi_local.py の存在確認 ---
if not exist "calc_roi_local.py" (
    echo [NG] calc_roi_local.py が見つかりません。
    echo   カレント: %cd%
    pause
    exit /b 1
)

REM --- 引数の処理 ---
set "ARG=%~1"
set "NO_PAUSE=0"
set "TARGET="

if /i "!ARG!"=="auto" (
    set "NO_PAUSE=1"
) else if not "!ARG!"=="" (
    set "TARGET=!ARG!"
)

echo ============================================================
echo  AI競馬予測 - 週末収支計算
echo  %date% %time%
echo ============================================================
echo.
echo  ディレクトリ: %cd%
echo.

REM --- Python 実行 ---
if "!TARGET!"=="" (
    python calc_roi_local.py
) else (
    python calc_roi_local.py !TARGET!
)

set "RC=!ERRORLEVEL!"

if !RC! NEQ 0 (
    echo.
    echo [NG] エラーが発生しました (exit code: !RC!)
)

echo.
echo ============================================================
echo  完了  %date% %time%
echo  CSV: jv_data\reports\roi_history.csv
echo  LOG: jv_data\reports\roi_history.log
echo ============================================================

if "!NO_PAUSE!"=="0" (
    echo.
    pause
)

endlocal
