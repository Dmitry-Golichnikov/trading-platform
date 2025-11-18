"""Tests for meta-learning."""

import numpy as np

from src.modeling.hyperopt.meta_learning import MetaLearning


def test_meta_learning_initialization(tmp_path):
    """Test meta-learning initialization."""
    db_path = tmp_path / "meta_learning.json"

    meta = MetaLearning(history_db_path=db_path)

    assert len(meta.history) == 0
    assert db_path.parent.exists()


def test_compute_dataset_meta_features():
    """Test computation of dataset meta-features."""
    db_path = "test_meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Create synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    meta_features = meta._compute_dataset_meta_features(X, y)

    # Check meta-features
    assert "n_samples" in meta_features
    assert "n_features" in meta_features
    assert "n_classes" in meta_features
    assert "class_balance" in meta_features

    assert meta_features["n_samples"] == 100
    assert meta_features["n_features"] == 10
    assert meta_features["n_classes"] == 2


def test_add_experiment(tmp_path):
    """Test adding experiment to history."""
    db_path = tmp_path / "meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Create synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    # Add experiment
    meta.add_experiment(
        model_type="lightgbm",
        best_params={"num_leaves": 31, "learning_rate": 0.1},
        best_score=0.85,
        X=X,
        y=y,
    )

    assert len(meta.history) == 1
    assert meta.history[0]["model_type"] == "lightgbm"
    assert meta.history[0]["best_score"] == 0.85

    # Check that file was saved
    assert db_path.exists()


def test_suggest_starting_point(tmp_path):
    """Test suggesting starting point."""
    db_path = tmp_path / "meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Add some experiments
    np.random.seed(42)

    for i in range(3):
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)

        meta.add_experiment(
            model_type="lightgbm",
            best_params={
                "num_leaves": 30 + i * 10,
                "learning_rate": 0.1 + i * 0.05,
            },
            best_score=0.8 + i * 0.02,
            X=X,
            y=y,
        )

    # Suggest for similar dataset
    X_new = np.random.randn(100, 10)
    y_new = (X_new[:, 0] > 0).astype(int)

    suggested = meta.suggest_starting_point(X=X_new, y=y_new, model_type="lightgbm")

    # Should suggest something
    assert isinstance(suggested, dict)

    if suggested:  # May be empty if no similar datasets
        assert "num_leaves" in suggested or "learning_rate" in suggested


def test_get_best_configs_for_model(tmp_path):
    """Test getting best configs for a model."""
    db_path = tmp_path / "meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Add experiments for different models
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    meta.add_experiment(
        model_type="lightgbm",
        best_params={"num_leaves": 31},
        best_score=0.85,
        X=X,
        y=y,
    )

    meta.add_experiment(
        model_type="lightgbm",
        best_params={"num_leaves": 40},
        best_score=0.87,
        X=X,
        y=y,
    )

    meta.add_experiment(
        model_type="xgboost",
        best_params={"max_depth": 5},
        best_score=0.82,
        X=X,
        y=y,
    )

    # Get best configs for lightgbm
    best_configs = meta.get_best_configs_for_model("lightgbm", top_k=2)

    assert len(best_configs) == 2
    # Should be sorted by score
    assert best_configs[0]["best_score"] >= best_configs[1]["best_score"]


def test_get_statistics(tmp_path):
    """Test getting statistics."""
    db_path = tmp_path / "meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Add experiments
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    for model_type in ["lightgbm", "xgboost", "lightgbm"]:
        meta.add_experiment(
            model_type=model_type,
            best_params={"param": 1},
            best_score=np.random.rand(),
            X=X,
            y=y,
        )

    stats = meta.get_statistics()

    assert stats["n_experiments"] == 3
    assert "model_counts" in stats
    assert stats["model_counts"]["lightgbm"] == 2
    assert stats["model_counts"]["xgboost"] == 1


def test_clear_history(tmp_path):
    """Test clearing history."""
    db_path = tmp_path / "meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Add experiment
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    meta.add_experiment(
        model_type="lightgbm",
        best_params={"num_leaves": 31},
        best_score=0.85,
        X=X,
        y=y,
    )

    assert len(meta.history) == 1

    # Clear
    meta.clear_history()

    assert len(meta.history) == 0


def test_similarity_computation(tmp_path):
    """Test similarity computation between datasets."""
    db_path = tmp_path / "meta_learning.json"
    meta = MetaLearning(history_db_path=db_path)

    # Create meta-features for two similar datasets
    meta1 = {
        "n_samples": 1000,
        "n_features": 50,
        "class_balance": 0.8,
    }

    meta2 = {
        "n_samples": 1100,
        "n_features": 55,
        "class_balance": 0.75,
    }

    similarity = meta._compute_similarity(meta1, meta2)

    assert 0.0 <= similarity <= 1.0

    # Very similar datasets should have high similarity
    meta3 = meta1.copy()
    similarity_identical = meta._compute_similarity(meta1, meta3)

    assert similarity_identical > 0.9
