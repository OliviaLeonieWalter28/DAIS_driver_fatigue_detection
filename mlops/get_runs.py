from config import logger, mlflow
import argparse
run_ids = []
def get_runs(experiment_name: str = "", metric: str = "", mode: str = "") -> str:
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    
    run_ids = sorted_runs['run_id'].tolist()
    
    return run_ids 



if __name__ == "__main__":  # pragma: no cover, application
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-name", help="--")
    parser.add_argument("--metric", help="--")
    parser.add_argument("--mode", help="--")

    args = parser.parse_args()

    runs = get_runs(args.experiment_name, args.metric, args.mode)
    print(runs)