@echo off
rem ragbench - abre o painel no navegador. Feche esta janela (ou Ctrl+C) para encerrar.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" -Painel %*
if errorlevel 1 pause
