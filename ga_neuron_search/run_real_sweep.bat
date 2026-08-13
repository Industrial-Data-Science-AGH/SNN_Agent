@echo off
REM Realny sweep GA (proxy-trening z torchem) po liczbie neuronow.
REM Dane 7-kanalowe: spikes_manifest7 (train/val) — uzywa GOTOWEGO cache _cache_*.npz.
REM Bez --limit, bo inna wartosc zmusza przebudowe cache w katalogu danych (WinError 5).
cd /d "%~dp0"
set ARCH=..\architecture_14_neurons_patryk_09_07
"%~dp0..\.venv\Scripts\python.exe" run_search.py ^
  --neurons 4 6 8 10 ^
  --mode real ^
  --arch-dir %ARCH% ^
  --data %ARCH%\spikes_manifest7\train ^
  --val-data %ARCH%\spikes_manifest7\val ^
  --epochs 4 --pop 24 --gens 15 ^
  --out wyniki_real
