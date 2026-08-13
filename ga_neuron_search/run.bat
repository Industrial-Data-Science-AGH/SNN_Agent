@echo off
REM Uruchamia dowolny skrypt GA w istniejacym .venv (torch 2.12).
REM Przyklady:
REM   run.bat test_genome.py
REM   run.bat run_search.py --neurons 4 6 8 10 --mode synth --out wyniki_synth
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" %*
