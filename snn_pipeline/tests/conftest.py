"""Opcje pytesta dla testów zbioru."""


def pytest_addoption(parser):
    parser.addoption("--dataset-version", default="v2.0.0",
                     help="wersja zbioru sprawdzana przez testy w tym katalogu")
