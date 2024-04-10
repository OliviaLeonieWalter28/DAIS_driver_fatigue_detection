import subprocess
import argparse

def run_training(participant):
    commands_training_tuning = [
        ". venv/bin/activate",
        # Train
        'export EXPERIMENT_NAME="fatigue_with_CV"',
        'export DATASET_LOC="mlops/train_data/data"',
        'export TRAIN_LOOP_CONFIG=\'{"fc_size": 512, "lr": 1e-3, "lr_factor": 0.8, "lr_patience": 3}\'',
        'python mlops/train.py '
        f'--participant {participant} '
        '--experiment-name "$EXPERIMENT_NAME" '
        '--dataset-loc "$DATASET_LOC" '
        '--train-loop-config "$TRAIN_LOOP_CONFIG" '
        '--num-samples 1000 '
        '--num-workers 4 '
        '--cpu-per-worker 1 '
        '--gpu-per-worker 0 '
        '--num-epochs 15 '
        '--batch-size 16 '
        '--results-fp results/training_results.json',
        # Tune
        'export INITIAL_PARAMS="[{\\"train_loop_config\\": $TRAIN_LOOP_CONFIG}]"',
        'python mlops/tune.py '
        f'--participant {participant} ' 
        '--experiment-name "$EXPERIMENT_NAME" '
        '--dataset-loc "$DATASET_LOC" '
        '--initial-params "$INITIAL_PARAMS" '
        '--num-runs 2 ' #50
        '--num-workers 6 '
        '--cpu-per-worker 1 '
        '--gpu-per-worker 0 '
        '--num-epochs 2 '
        '--batch-size 16 '
        '--results-fp results/tuning_results.json',
        ]

    command_string_training_tuning = "; ".join(commands_training_tuning)
    # this will run training and tuning and will wait for it to finish
    process1 = subprocess.Popen([command_string_training_tuning ], shell=True).wait()
    commands_get_runs = [
    # get runs
    ". venv/bin/activate",
    'export EXPERIMENT_NAME="fatigue_with_CV"',
    "python mlops/get_runs.py "
    '--experiment-name "$EXPERIMENT_NAME" '
    '--metric val_loss '
    '--mode ASC '
    ]
        
    command_string_get_runs = "; ".join(commands_get_runs)
    return_runs = subprocess.Popen([command_string_get_runs], shell=True,stdout=subprocess.PIPE)
    output, _ = return_runs.communicate()

    runs = output.decode('utf-8')
    runslist = runs.split("'")
    # deletes the ',', '[' and ']' from the runlist
    filtered_runs = []
    for run in runslist:
        if "," in run or "[" in run or "]" in run:
            continue
        else:
            filtered_runs.append(run)
            
    print(filtered_runs)

    for run in filtered_runs:
        run_id = run
        commands_evaluation = [
        ". venv/bin/activate",
        # Evaluation
        'export EXPERIMENT_NAME="fatigue_with_CV"',
        'export HOLDOUT_LOC="mlops/train_data/data"',
        'python mlops/evaluate.py '
        f'--run-id {run_id}  '
        '--dataset-loc "$HOLDOUT_LOC" '
        f'--participant {participant} '
        '--subset "test" '
        '--results-fp results/evaluation_results.json',
        # Inference 
        'export DATASET_LOC="mlops/train_data/data"',
        'python mlops/predict.py predict '
        f'--run-id {run_id}  '
        '--dataset-loc "$DATASET_LOC" '
        f'--participant {participant} '
        '--subset "test"' 
        ]
        
        command_string_evaluation = "; ".join(commands_evaluation)
        #This will run evaluation 
        subprocess.Popen([command_string_evaluation], shell=True).wait()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--participant", help="--")
    args = parser.parse_args()
    
    run_training(args.participant)