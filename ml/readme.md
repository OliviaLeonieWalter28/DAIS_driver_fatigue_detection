# Setup
Follow this video to connect your WSL ubuntu setup with your Webcam
https://youtu.be/t_YnACEPmrM?si=lOrrU2Wrl2PyERZY

```
pip install opencv-python
pip install mediapipe
pip install tensorflow
pip install albumentations
sudo apt-get install libgtk2.0-dev
pip uninstall opencv-python
pip install opencv-python

```
# Pre-processing

```
export DATASET_LOC="/home/username/preprocessed_dataset/preprocessed_dataset/train"
export OUTPUT_LOC="/home/username/preprocessed_dataset/output"
python ml/data_preprocessing.py --raw_data_path "$DATASET_LOC"  --dest_path  "$OUTPUT_LOC" --participant_test 2
```
# Training Network
open cnn1d.py and edited lines 268-272 to fit your preprocessed data

```
python3 ml/cnn1d.py

```
# Running Live extraction
python3 ml/extract_features.py

# Experiment Tracking
Use the MLflow library to track our experiments and store our models and the MLflow Tracking UI to view our experiments. We have been saving our experiments to a local directory but note that in an actual production setting, we would have a central location to store all of our experiments. It's easy/inexpensive to spin up your own MLflow server for all of your team members to track their experiments on or use a managed solution like Weights & Biases, Comet, etc.

```
export MODEL_REGISTRY=/tmp/mlflow
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $MODEL_REGISTRY
```

Then look at http://localhost:8080/ to view your MLflow dashboard.

# Sending Data to MLflow
```
export PARTICIPANT=$(python -c 'from ml.savedata import get_device_id; print(get_device_id())')
python3 ml/senddata.py --serveruri "http://localhost:8080" --experiment_name "fatigue_with_CV" --participant $PARTICIPANT
```

