import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_main_package_can_be_imported():
    import smart_waste_multimodal

    assert smart_waste_multimodal.__file__ is not None


def test_experiment_manifest_template_is_valid_json():
    manifest_path = (
        PROJECT_ROOT / "experiments" / "templates" / "run_manifest.json"
    )

    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "experiment_id" in manifest
    assert "git_commit" in manifest
    assert "metrics" in manifest
    assert "status" in manifest


def test_required_project_directories_exist():
    required_directories = [
        "src/smart_waste_multimodal",
        "contracts",
        "configs",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs/adr",
        "experiments",
    ]

    for relative_path in required_directories:
        assert (PROJECT_ROOT / relative_path).is_dir(), relative_path
