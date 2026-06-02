from pathlib import Path


def test_ci_workflow_exists():
    assert Path(".github/workflows/ci.yml").is_file()


def test_python_project_files_exist():
    assert Path("pyproject.toml").is_file()
    assert Path("requirements.txt").is_file()


def test_tests_directory_exists():
    assert Path("tests").is_dir()
