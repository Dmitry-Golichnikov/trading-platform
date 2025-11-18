"""Tests for ExperimentManager."""

import tempfile
from pathlib import Path

import pytest

from src.orchestration import ExperimentManager


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def experiment_manager(temp_artifacts_dir):
    """Create experiment manager with temporary directory."""
    return ExperimentManager(
        artifacts_dir=temp_artifacts_dir,
        mlflow_tracking_uri=None,  # Disable MLflow for tests
    )


def test_create_experiment(experiment_manager):
    """Test experiment creation."""
    config = {
        "model": "lightgbm",
        "ticker": "SBER",
        "timeframe": "1h",
    }

    exp_id = experiment_manager.create_experiment(
        name="test_experiment",
        config=config,
        description="Test experiment",
        tags={"test": "true"},
    )

    assert exp_id is not None
    assert "test_experiment" in exp_id

    # Verify experiment was saved
    exp = experiment_manager.get_experiment(exp_id)
    assert exp["name"] == "test_experiment"
    assert exp["status"] == "created"
    assert exp["config"] == config
    assert exp["tags"]["test"] == "true"


def test_list_experiments(experiment_manager):
    """Test listing experiments."""
    # Create multiple experiments
    for i in range(3):
        experiment_manager.create_experiment(
            name=f"exp_{i}",
            config={"index": i},
        )

    experiments = experiment_manager.list_experiments()
    assert len(experiments) == 3


def test_list_experiments_with_filters(experiment_manager):
    """Test listing experiments with filters."""
    # Create experiments with different statuses
    exp_id_1 = experiment_manager.create_experiment(
        name="exp_1",
        config={},
    )

    exp_id_2 = experiment_manager.create_experiment(
        name="exp_2",
        config={},
    )

    # Update statuses
    experiment_manager.update_experiment(exp_id_1, status="completed")
    experiment_manager.update_experiment(exp_id_2, status="failed")

    # Filter by status
    completed = experiment_manager.list_experiments(status="completed")
    assert len(completed) == 1
    assert completed[0]["id"] == exp_id_1

    failed = experiment_manager.list_experiments(status="failed")
    assert len(failed) == 1
    assert failed[0]["id"] == exp_id_2


def test_list_experiments_with_tags(experiment_manager):
    """Test filtering experiments by tags."""
    experiment_manager.create_experiment(
        name="exp_1",
        config={},
        tags={"model": "lightgbm", "ticker": "SBER"},
    )

    experiment_manager.create_experiment(
        name="exp_2",
        config={},
        tags={"model": "xgboost", "ticker": "SBER"},
    )

    # Filter by tags
    experiments = experiment_manager.list_experiments(tags={"model": "lightgbm"})

    assert len(experiments) == 1
    assert experiments[0]["name"] == "exp_1"


def test_update_experiment(experiment_manager):
    """Test updating experiment."""
    exp_id = experiment_manager.create_experiment(
        name="test",
        config={},
    )

    # Update status
    experiment_manager.update_experiment(exp_id, status="running")

    exp = experiment_manager.get_experiment(exp_id)
    assert exp["status"] == "running"

    # Update tags
    experiment_manager.update_experiment(
        exp_id,
        tags={"new_tag": "value"},
    )

    exp = experiment_manager.get_experiment(exp_id)
    assert exp["tags"]["new_tag"] == "value"


def test_delete_experiment(experiment_manager):
    """Test deleting experiment."""
    exp_id = experiment_manager.create_experiment(
        name="test",
        config={},
    )

    experiment_manager.delete_experiment(exp_id)

    with pytest.raises(ValueError, match="Experiment not found"):
        experiment_manager.get_experiment(exp_id)


def test_export_experiments(experiment_manager, temp_artifacts_dir):
    """Test exporting experiments."""
    # Create experiments
    for i in range(3):
        experiment_manager.create_experiment(
            name=f"exp_{i}",
            config={"index": i},
        )

    # Export to CSV
    output_path = temp_artifacts_dir / "experiments.csv"
    experiment_manager.export_experiments(output_path, format="csv")

    assert output_path.exists()

    # Export to JSON
    output_path = temp_artifacts_dir / "experiments.json"
    experiment_manager.export_experiments(output_path, format="json")

    assert output_path.exists()


def test_generate_summary_report(experiment_manager):
    """Test generating summary report."""
    # Create experiments
    experiment_manager.create_experiment(
        name="exp_1",
        config={},
    )

    experiment_manager.create_experiment(
        name="exp_2",
        config={},
    )

    report = experiment_manager.generate_summary_report()

    assert "Total Experiments: 2" in report
    assert "exp_1" in report
    assert "exp_2" in report


def test_cleanup_failed_experiments(experiment_manager):
    """Test cleaning up failed experiments."""
    # Create experiments
    exp_id_1 = experiment_manager.create_experiment(name="exp_1", config={})
    exp_id_2 = experiment_manager.create_experiment(name="exp_2", config={})
    exp_id_3 = experiment_manager.create_experiment(name="exp_3", config={})

    # Mark some as failed
    experiment_manager.update_experiment(exp_id_1, status="failed")
    experiment_manager.update_experiment(exp_id_2, status="completed")
    experiment_manager.update_experiment(exp_id_3, status="failed")

    # Cleanup
    experiment_manager.cleanup_failed_experiments()

    # Check remaining experiments
    experiments = experiment_manager.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["id"] == exp_id_2
