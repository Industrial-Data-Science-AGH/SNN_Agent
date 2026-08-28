"""Opcje pytesta dla testów zbioru."""


def pytest_addoption(parser):
    parser.addoption("--dataset-version", default="v2.0.0",
                     help="wersja zbioru sprawdzana przez testy w tym katalogu")
    parser.addoption("--spikes-dir",
                     default="architecture_14_neurons_patryk_09_07/spikes_v2",
                     help="katalog artefaktu spike'owego (train/val/test)")
