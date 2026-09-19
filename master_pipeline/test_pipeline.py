import os
import json
import pytest
from unittest.mock import patch

from pipeline_config import PipelineConfig
from hardware import get_device
from tracker import RunTracker
import pipeline

## Test 1: Konfiguracja i poprawne przypisanie datasetów
def test_config_io_and_datasets(tmp_path):
    cfg = PipelineConfig()
    
    # Weryfikacja jawnych flag dla ról datasetów
    assert hasattr(cfg.data, "train")
    assert hasattr(cfg.data, "val")
    assert hasattr(cfg.data, "test")
    assert hasattr(cfg.data, "spikes_ext")
    
    # Symulacja zapisu i odczytu
    cfg.data.train = "custom_train_dataset"
    cfg_path = tmp_path / "config.json"
    cfg.to_json(str(cfg_path))
    
    assert cfg_path.exists()
    
    loaded_cfg = PipelineConfig.from_json(str(cfg_path))
    assert loaded_cfg.data.train == "custom_train_dataset"
    assert loaded_cfg.ga.pop_size == cfg.ga.pop_size

## Test 2: Wybór urządzenia (fallback i czytelne błędy niedostępności)
@patch("hardware.torch.cuda.is_available", return_value=False)
@patch("hardware.torch.backends.mps.is_available", return_value=False, create=True)
def test_device_selection(mock_mps, mock_cuda):
    # Tryb auto na maszynie bez akceleratorów wymusza CPU
    assert get_device("auto") == "cpu"
    
    # Jawne zażądanie niedostępnego cuda/mps musi rzucić wyjątek RuntimeError
    with pytest.raises(RuntimeError):
        get_device("cuda")
        
    with pytest.raises(RuntimeError):
        get_device("mps")

## Test 3: Generowanie raportu (Manifest)
def test_tracker_manifest_generation(tmp_path):
    cfg = PipelineConfig()
    tracker = RunTracker(config=cfg, device="cpu", workers=1, hw_benchmark=None)
    
    # Przekierowujemy wyjście trackera do wirtualnego folderu tmp_path z pytest
    tracker.run_dir = str(tmp_path)
    tracker.manifest_path = os.path.join(tracker.run_dir, "manifest.json")
    
    tracker.log_metrics("test_stage", {"clip_f1": 0.85})
    tracker.log_stage_time("test_stage", 10.5)
    tracker.update_manifest(status="COMPLETED")
    
    assert os.path.exists(tracker.manifest_path)
    
    with open(tracker.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["status"] == "COMPLETED"
    assert "test_stage" in manifest["metrics"]
    assert manifest["metrics"]["test_stage"]["clip_f1"] == 0.85
    
## Test 4: Wznawianie eksperymentu (Resume Logic)
@patch("pipeline.run_ga_stage")
def test_resume_logic_skips_ga(mock_ga_stage, tmp_path):
    # Przygotowanie sztucznego folderu po przerwanym runie (Etap 1 ukończony)
    fake_run_dir = tmp_path / "run_fake"
    fake_run_dir.mkdir()
    manifest_path = fake_run_dir / "manifest.json"
    
    fake_manifest = {
        "status": "IN_PROGRESS",
        "metrics": {
            "ga_stage": {
                "best_topology": {"layers": [[0, 1], [0]]}
            }
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(fake_manifest, f)
        
    test_args = ["pipeline.py", "--device", "cpu", "--resume", str(fake_run_dir), "run-all"]
    
    # Patchowanie argumentów systemowych oraz pozostałych etapów, aby nie liczyły się w tle
    with patch("sys.argv", test_args), \
         patch("pipeline.run_ext_evaluation_stage") as mock_ext, \
         patch("pipeline.run_final_evaluation_stage") as mock_eval, \
         patch("pipeline.run_hardware_export_stage") as mock_hw:
         
         pipeline.main()
         
         # Algorytm genetyczny został pominięty, bo topologia jest już w pliku
         mock_ga_stage.assert_not_called()
         
         # Skrypt przeszedł bezpośrednio do Etapu 2, 3 i 4
         mock_ext.assert_called_once()
         mock_eval.assert_called_once()
         mock_hw.assert_called_once()
