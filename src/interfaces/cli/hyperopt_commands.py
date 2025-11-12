"""CLI commands for hyperparameter optimization."""

from pathlib import Path
from typing import Optional

import click
import pandas as pd
import yaml


@click.group(name="hyperopt")
def hyperopt_group():
    """Hyperparameter optimization commands."""
    pass


@hyperopt_group.command(name="run")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to hyperopt config YAML file",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data (Parquet)",
)
@click.option(
    "--target-col",
    type=str,
    default="target",
    help="Name of target column",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts/hyperopt",
    help="Output directory for results",
)
@click.option(
    "--experiment-name",
    type=str,
    default="hyperopt",
    help="MLflow experiment name",
)
def run_hyperopt(
    config: str,
    data_path: str,
    target_col: str,
    output_dir: str,
    experiment_name: str,
):
    """Run hyperparameter optimization."""
    from src.modeling.hyperopt.bayesian import BayesianOptimizer
    from src.modeling.hyperopt.grid_search import GridSearchOptimizer
    from src.modeling.hyperopt.optuna_backend import OptunaOptimizer
    from src.modeling.hyperopt.random_search import RandomSearchOptimizer
    from src.modeling.registry import ModelRegistry
    from src.orchestration.experiment_tracker import ExperimentTracker

    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data
    click.echo(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Train/val split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    click.echo(f"Training set: {len(X_train)} samples")
    click.echo(f"Validation set: {len(X_val)} samples")

    # Create objective function
    model_type = cfg.get("model_type", "lightgbm")
    metric_name = cfg.get("metric_name", "roc_auc")

    def objective(params):
        """Objective function to optimize."""
        from sklearn.metrics import roc_auc_score

        # Create model
        model_cls = ModelRegistry.get_model(model_type)
        model = model_cls(**params, random_state=42)

        # Train
        model.fit(X_train.values, y_train, eval_set=(X_val.values, y_val))

        # Evaluate
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X_val.values)[:, 1]
        else:
            y_pred = model.predict(X_val.values)

        score = roc_auc_score(y_val, y_pred)

        return score

    # Create optimizer
    search_space = cfg.get("search_space", {})
    optimizer_type = cfg.get("optimizer_type", "bayesian")
    optimizer_config = cfg.get("optimizer_config", {})

    click.echo(f"Optimizer: {optimizer_type}")
    click.echo(f"Model: {model_type}")

    from src.modeling.hyperopt.base import BaseOptimizer

    optimizer: BaseOptimizer
    if optimizer_type == "grid":
        optimizer = GridSearchOptimizer(
            objective_func=objective,
            search_space=search_space,
            metric_name=metric_name,
            **optimizer_config,
        )
    elif optimizer_type == "random":
        optimizer = RandomSearchOptimizer(
            objective_func=objective,
            search_space=search_space,
            metric_name=metric_name,
            **optimizer_config,
        )
    elif optimizer_type == "bayesian":
        optimizer = BayesianOptimizer(
            objective_func=objective,
            search_space=search_space,
            metric_name=metric_name,
            **optimizer_config,
        )
    elif optimizer_type == "optuna":
        optimizer = OptunaOptimizer(
            objective_func=objective,
            search_space=search_space,
            metric_name=metric_name,
            **optimizer_config,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Run optimization
    result = optimizer.optimize()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / "result.yaml"
    result.save(result_file)

    click.echo("\n✅ Optimization complete!")
    click.echo(f"Best {metric_name}: {result.best_value:.4f}")
    click.echo(f"Best params: {result.best_params}")
    click.echo(f"Results saved to: {result_file}")

    # Log to experiment tracker
    tracker = ExperimentTracker(experiment_name=experiment_name)
    tracker.log_experiment(
        run_name=f"{model_type}_{optimizer_type}",
        config=result.best_params,
        metrics={metric_name: result.best_value},
        artifacts={"result": result_file},
    )


@hyperopt_group.command(name="automl")
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to training data (Parquet)",
)
@click.option(
    "--target-col",
    type=str,
    default="target",
    help="Name of target column",
)
@click.option(
    "--time-budget",
    type=int,
    default=3600,
    help="Time budget in seconds",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts/automl",
    help="Output directory",
)
@click.option(
    "--experiment-name",
    type=str,
    default="automl",
    help="MLflow experiment name",
)
def run_automl(
    data_path: str,
    target_col: str,
    time_budget: int,
    output_dir: str,
    experiment_name: str,
):
    """Run AutoML pipeline."""
    from src.modeling.hyperopt.automl import AutoMLPipeline

    # Load data
    click.echo(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Train/val split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    click.echo(f"Training set: {len(X_train)} samples")
    click.echo(f"Validation set: {len(X_val)} samples")

    # Create AutoML pipeline
    automl = AutoMLPipeline(
        time_budget=time_budget,
        mlflow_tracking=True,
    )

    # Fit
    automl.fit(X_train, y_train, X_val, y_val)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save leaderboard
    leaderboard = automl.get_leaderboard()
    leaderboard_file = output_path / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_file, index=False)

    click.echo("\n✅ AutoML complete!")
    click.echo("\nLeaderboard:")
    click.echo(leaderboard.to_string())
    click.echo(f"\nResults saved to: {output_path}")


@hyperopt_group.command(name="compare")
@click.option(
    "--run-ids",
    type=str,
    required=True,
    help="Comma-separated list of run IDs to compare",
)
@click.option(
    "--metrics",
    type=str,
    help="Comma-separated list of metrics to compare (optional)",
)
@click.option(
    "--experiment-name",
    type=str,
    default="default",
    help="Experiment name",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output CSV file (optional)",
)
def compare_experiments(
    run_ids: str,
    metrics: Optional[str],
    experiment_name: str,
    output: Optional[str],
):
    """Compare multiple experiments."""
    from src.orchestration.experiment_tracker import ExperimentTracker

    # Parse run IDs
    run_id_list = [rid.strip() for rid in run_ids.split(",")]

    # Parse metrics
    metric_list = None
    if metrics:
        metric_list = [m.strip() for m in metrics.split(",")]

    # Create tracker
    tracker = ExperimentTracker(experiment_name=experiment_name)

    # Compare
    comparison_df = tracker.compare_experiments(run_id_list, metric_list)

    if comparison_df.empty:
        click.echo("No experiments found to compare")
        return

    # Display
    click.echo("\nExperiment Comparison:")
    click.echo("=" * 80)
    click.echo(comparison_df.to_string())

    # Save if requested
    if output:
        comparison_df.to_csv(output, index=False)
        click.echo(f"\nSaved comparison to: {output}")


@hyperopt_group.command(name="best")
@click.option(
    "--metric",
    type=str,
    required=True,
    help="Metric to optimize",
)
@click.option(
    "--direction",
    type=click.Choice(["maximize", "minimize"]),
    default="maximize",
    help="Optimization direction",
)
@click.option(
    "--experiment-name",
    type=str,
    default="default",
    help="Experiment name",
)
def get_best_experiment(metric: str, direction: str, experiment_name: str):
    """Find best experiment based on a metric."""
    from src.orchestration.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(experiment_name=experiment_name)

    try:
        best_exp = tracker.get_best_experiment(metric, direction)

        click.echo("\n🏆 Best Experiment:")
        click.echo("=" * 80)
        click.echo(f"Run ID: {best_exp['run_id']}")
        click.echo(f"Run Name: {best_exp['run_name']}")
        click.echo(f"Timestamp: {best_exp['timestamp']}")
        click.echo(f"\nBest {metric}: {best_exp['metrics'][metric]:.4f}")
        click.echo("\nConfiguration:")
        for key, value in best_exp.get("config", {}).items():
            click.echo(f"  {key}: {value}")

    except ValueError as e:
        click.echo(f"❌ Error: {e}")


@hyperopt_group.command(name="list")
@click.option(
    "--experiment-name",
    type=str,
    default="default",
    help="Experiment name",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of experiments to show",
)
def list_experiments(experiment_name: str, limit: Optional[int]):
    """List all experiments."""
    from src.orchestration.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(experiment_name=experiment_name)
    experiments = tracker.list_experiments(limit=limit)

    if not experiments:
        click.echo("No experiments found")
        return

    click.echo(f"\n📊 Experiments ({len(experiments)} total):")
    click.echo("=" * 80)

    for i, exp in enumerate(experiments, 1):
        click.echo(f"\n{i}. {exp['run_name']}")
        click.echo(f"   Run ID: {exp['run_id']}")
        click.echo(f"   Timestamp: {exp['timestamp']}")

        metrics = exp.get("metrics", {})
        if metrics:
            click.echo("   Metrics:")
            for key, value in metrics.items():
                click.echo(f"     {key}: {value:.4f}")


@hyperopt_group.command(name="export")
@click.option(
    "--experiment-name",
    type=str,
    default="default",
    help="Experiment name",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output CSV file",
)
def export_experiments(experiment_name: str, output: str):
    """Export experiments to CSV."""
    from src.orchestration.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(experiment_name=experiment_name)
    tracker.export_to_csv(output)

    click.echo(f"✅ Experiments exported to: {output}")


# Update the main CLI to include hyperopt commands
def register_hyperopt_commands(cli):
    """Register hyperopt commands with main CLI."""
    cli.add_command(hyperopt_group)
