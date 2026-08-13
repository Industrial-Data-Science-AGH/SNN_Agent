# Uruchamia skrypt GA w istniejacym .venv. Przyklad:
#   .\run.ps1 test_genome.py
#   .\run.ps1 run_search.py --neurons 4 6 8 10 --mode synth
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $py @args
