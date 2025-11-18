"""Tests for experiment tracker."""

from src.orchestration.experiment_tracker import ExperimentTracker


def test_experiment_tracker_initialization(tmp_path):
    """Test experiment tracker initialization."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    assert tracker.experiment_name == "test_exp"
    assert tracker.local_db_path == db_path


def test_log_experiment(tmp_path):
    """Test logging an experiment."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiment
    run_id = tracker.log_experiment(
        run_name="test_run",
        config={"learning_rate": 0.1, "max_depth": 5},
        metrics={"accuracy": 0.85, "f1_score": 0.82},
        tags={"version": "v1"},
    )

    assert run_id is not None
    assert db_path.exists()


def test_get_experiment(tmp_path):
    """Test getting an experiment."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiment
    run_id = tracker.log_experiment(
        run_name="test_run",
        config={"learning_rate": 0.1},
        metrics={"accuracy": 0.85},
    )

    # Get experiment
    exp = tracker.get_experiment(run_id)

    assert exp["run_id"] == run_id
    assert exp["run_name"] == "test_run"
    assert exp["config"]["learning_rate"] == 0.1
    assert exp["metrics"]["accuracy"] == 0.85


def test_list_experiments(tmp_path):
    """Test listing experiments."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log multiple experiments
    for i in range(3):
        tracker.log_experiment(
            run_name=f"run_{i}",
            config={"lr": 0.1 + i * 0.05},
            metrics={"accuracy": 0.8 + i * 0.02},
        )

    # List all
    experiments = tracker.list_experiments()

    assert len(experiments) == 3


def test_list_experiments_with_filters(tmp_path):
    """Test listing experiments with filters."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiments with tags
    tracker.log_experiment(
        run_name="run_1",
        config={"lr": 0.1},
        metrics={"accuracy": 0.85},
        tags={"model": "lightgbm"},
    )

    tracker.log_experiment(
        run_name="run_2",
        config={"lr": 0.2},
        metrics={"accuracy": 0.87},
        tags={"model": "xgboost"},
    )

    # Filter by tag
    experiments = tracker.list_experiments(filters={"tags": {"model": "lightgbm"}})

    assert len(experiments) == 1
    assert experiments[0]["run_name"] == "run_1"


def test_compare_experiments(tmp_path):
    """Test comparing experiments."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiments
    run_ids = []
    for i in range(3):
        run_id = tracker.log_experiment(
            run_name=f"run_{i}",
            config={"lr": 0.1 + i * 0.05},
            metrics={"accuracy": 0.8 + i * 0.02, "f1": 0.75 + i * 0.03},
        )
        run_ids.append(run_id)

    # Compare
    comparison_df = tracker.compare_experiments(run_ids, metrics=["accuracy", "f1"])

    assert len(comparison_df) == 3
    assert "accuracy" in comparison_df.columns
    assert "f1" in comparison_df.columns


def test_get_best_experiment(tmp_path):
    """Test getting best experiment."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiments with different scores
    for i in range(3):
        tracker.log_experiment(
            run_name=f"run_{i}",
            config={"lr": 0.1 + i * 0.05},
            metrics={"accuracy": 0.8 + i * 0.02},
        )

    # Get best (maximize)
    best_exp = tracker.get_best_experiment(metric="accuracy", direction="maximize")

    assert best_exp["run_name"] == "run_2"  # Highest accuracy
    assert best_exp["metrics"]["accuracy"] == 0.84


def test_delete_experiment(tmp_path):
    """Test deleting an experiment."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiment
    run_id = tracker.log_experiment(
        run_name="test_run",
        config={"lr": 0.1},
        metrics={"accuracy": 0.85},
    )

    # Delete
    tracker.delete_experiment(run_id)

    # Check deleted
    experiments = tracker.list_experiments()
    assert len(experiments) == 0


def test_get_summary_statistics(tmp_path):
    """Test getting summary statistics."""
    db_path = tmp_path / "experiments.json"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiments
    for i in range(5):
        tracker.log_experiment(
            run_name=f"run_{i}",
            config={"lr": 0.1},
            metrics={"accuracy": 0.8 + i * 0.02, "f1": 0.75 + i * 0.01},
        )

    # Get statistics
    stats = tracker.get_summary_statistics()

    assert stats["n_experiments"] == 5
    assert "accuracy" in stats["metric_statistics"]
    assert "f1" in stats["metric_statistics"]

    accuracy_stats = stats["metric_statistics"]["accuracy"]
    assert accuracy_stats["count"] == 5
    assert 0.8 <= accuracy_stats["mean"] <= 0.9


def test_export_to_csv(tmp_path):
    """Test exporting experiments to CSV."""
    db_path = tmp_path / "experiments.json"
    csv_path = tmp_path / "experiments.csv"

    tracker = ExperimentTracker(
        experiment_name="test_exp",
        local_db_path=db_path,
        use_mlflow=False,
    )

    # Log experiments
    for i in range(3):
        tracker.log_experiment(
            run_name=f"run_{i}",
            config={"lr": 0.1 + i * 0.05},
            metrics={"accuracy": 0.8 + i * 0.02},
        )

    # Export
    tracker.export_to_csv(csv_path)

    assert csv_path.exists()

    # Read and verify
    import pandas as pd

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert "run_name" in df.columns
    assert "metric_accuracy" in df.columns
