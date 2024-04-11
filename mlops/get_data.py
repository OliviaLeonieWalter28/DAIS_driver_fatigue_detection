from pathlib import Path
import mlflow
import time
from mlflow.tracking import MlflowClient
import argparse
import subprocess
import os

old_runs = []
raw_data_path = 'mlops/train_data'
mlflow_artifacts_path = "mlops/artifacts"
senddata_tag = "ml/senddata.py"
participant = 0
def check_if_new_data(serveruri,experiment_name):
            
        if os.path.exists(raw_data_path) == False:
            os.makedirs(raw_data_path, exist_ok=True)
        if os.path.exists(mlflow_artifacts_path) == False:
            os.makedirs(mlflow_artifacts_path, exist_ok=True)
            
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(serveruri)
        
        # Set the experiment name
        mlflow.set_experiment(experiment_name)
        
        
        while True:
            try:
                new_data = False
                client = MlflowClient()
                runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiment_name).experiment_id)
                for _, run_info in runs.iterrows():
                    run_id = run_info.run_id
                        
                    # Get the rundetails to access tags
                    run = mlflow.get_run(run_id)
                    tags = run.data.tags
                    
                    run_name = run.info.run_name       
                        
                    # Check if run is new data run and also if data is old
                    if 'mlflow.source.name' in tags and tags['mlflow.source.name'] == 'ml/senddata.py':
                        if run_id not in old_runs:
                            old_runs.append(run_id)
                            mlflow.tracking.MlflowClient().download_artifacts(run_id, 'data', raw_data_path)
                            participant = tags['mlflow.runName']
                            new_data = True
                                
                # if new data then start training process
                if new_data == True:
                    commands_training_tuning = [
                        ". venv/bin/activate",
                        f"python3 mlops/run.py --participant {run_name}"
                        ]
                    command_string_training_tuning = "; ".join(commands_training_tuning)
                    process = subprocess.Popen([command_string_training_tuning ], shell=True).wait() 
                    
               
            except Exception as e:
                print(f"Error: {e}")
                
            # Check for new run every minute
            time.sleep(60)

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--serveruri", help="--")
    parser.add_argument("--experiment_name", help="--")
    args = parser.parse_args()
    
    check_if_new_data(args.serveruri, args.experiment_name)