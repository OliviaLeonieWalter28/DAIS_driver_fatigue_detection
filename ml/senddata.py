from pathlib import Path
import subprocess
import mlflow
import os
import argparse
import pandas as pd

folderpath = r"ml/trained_data/processed"
folderpath_preprocessing = r"ml/trained_data/for_processing"

# Checks so rows in CSV file doenst't overstep 5001, used to create evenly sized files
def check_data_size():
     for file_name in os.listdir("ml/trained_data/"):
        if file_name.endswith(r'.csv'):
            df = pd.read_csv(f"ml/trained_data/{file_name}")
            num_rows = df.shape[0]
            if num_rows > 5000:
                df.to_csv(f"{folderpath_preprocessing}/{file_name}", index=False, header=True)
# calls preprocessing through bash
def data_preprocessing(participant):

    commands_training_tuning = [
        ". venv/bin/activate",
        # Data Pre-Processing
        'export DATASET_LOC="ml/trained_data/for_processing"',
        'export OUTPUT_LOC="ml/trained_data/processed"',
        f'python ml/data_preprocessing.py --raw_data_path "$DATASET_LOC"  --dest_path  "$OUTPUT_LOC" --participant_test {participant}',
        'echo "new data processed"',]
    
    command_string_training_tuning = "; ".join(commands_training_tuning)
    # this will run training and tuning and will wait for it to finish
    process1 = subprocess.Popen([command_string_training_tuning ], shell=True).wait()
    
# send the data to a MLFLOW server
def send_data_to_server(serveruri, experiment_name,participant):
    check_data_size()
    data_preprocessing(participant)
    
    run_name = f"{participant}"
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(serveruri)
    
    # Set the experiment name
    mlflow.set_experiment(experiment_name)
    
    # starts a run that uploads csv files
    with mlflow.start_run(run_name=run_name):
        for filename in os.listdir(folderpath):
            if filename.endswith('.pkl'):
                csv_filepath = os.path.join(folderpath, filename)
                
                # This will store all csv files in seperate folders
                # artifact_path = f'{filename}'
                # mlflow.log_artifact(csv_filepath, artifact_path=artifact_path)
                
                # This will store all CSV files in one folder
                mlflow.log_artifact(csv_filepath, artifact_path='data')
                

    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--serveruri", help="--")
    parser.add_argument("--experiment_name", help="--")
    parser.add_argument("--participant", help="--")


    args = parser.parse_args()
    
    send_data_to_server(args.serveruri, args.experiment_name, args.participant)


    
    