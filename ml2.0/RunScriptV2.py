import cv2
import dlib
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import csv
import time
import savedata as savetocsv


class DrowsinessNet(nn.Module):
    def __init__(self):
        super(DrowsinessNet, self).__init__()
        self.fc1 = nn.Linear(5, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x


def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.getcwd()


# Load the trained model and the scaler
model = DrowsinessNet()
base_path = get_base_path()
model_path = os.path.join(base_path, 'best_model_02-05_Medium.pth')
scaler_path = os.path.join(base_path, 'scaler_best_model_02-05_Medium.pkl')
predictor_path = os.path.join(base_path, 'shape_predictor_68_face_landmarks.dat')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load(scaler_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

ear_threshold = 0.21
max_drowsiness_score = 11
decay_rate = 0.1
frame_buffer = 0
drowsiness_score = 0
frame_count = 0


# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    horiz_dist = np.linalg.norm(mouth[0] - mouth[6])
    vert_dist_1 = np.linalg.norm(mouth[2] - mouth[10])
    vert_dist_2 = np.linalg.norm(mouth[4] - mouth[8])
    return (vert_dist_1 + vert_dist_2) / (2.0 * horiz_dist)


# Function to calculate Mouth Over Eye (MOE) ratio
def mouth_over_eye(mar, ear):
    return mar / ear if ear != 0 else float('inf')


def calculate_puc(ear, ear_threshold, frame_buffer, consecutive_frames_threshold=3):
    return 1 if (frame_buffer >= consecutive_frames_threshold and ear < ear_threshold) else 0


def calculate_moe(mar, ear):
    return mar / ear if ear != 0 else float('inf')


avg_ear_open = 0.26
avg_ear_half_open = 0.20
avg_ear_closed = 0.15

max_ear = avg_ear_open
min_ear = avg_ear_closed

frame_buffer_limit = 30
increase_rate = 0.5
decrease_rate = 0.1

consecutive_frames = 0

max_ear = avg_ear_open
min_ear = avg_ear_closed
frame_threshold_for_immediate_max_score = 30  # Approx 1 second if 30 FPS


def adjust_drowsiness_score(ear, mar, puc, moe):
    global drowsiness_score
    mar_threshold = 0.5
    mar_scaling_factor = 0.1
    some_moe_threshold = 0.75
    some_reduction_factor = 0.5
    smoothing_factor = 0.1
    decrease_rate = 0.1
    increase_rate = 0.2

    ear = max(min(ear, max_ear), min_ear)

    normalized_score = (ear - min_ear) / (max_ear - min_ear)
    ear_score = (1 - normalized_score) * (max_drowsiness_score - 1) + 1
    current_score = ear_score

    if ear > avg_ear_open:
        current_score -= decay_rate
        current_score = max(current_score, 0)

    current_score = min(current_score, max_drowsiness_score)

    if ear > avg_ear_open:
        decrease_rate = 0.05  # Decrease rate when eyes are open
    else:
        decrease_rate = 0.01  # Decrease rate when eyes are closed or half-closed

    mar_threshold = 0.5
    mar_scaling_factor = 0.2

    if mar > mar_threshold:
        excess_mar = mar - mar_threshold

        drowsiness_score += (excess_mar * mar_scaling_factor)

    drowsiness_score = min(drowsiness_score, max_drowsiness_score)

    smoothing_factor = 0.1
    drowsiness_score = (1 - smoothing_factor) * drowsiness_score + smoothing_factor * current_score

    drowsiness_score -= decrease_rate
    drowsiness_score = max(drowsiness_score, 0)


# Function to write data to CSV file
def write_to_csv(data, filename='drowsiness_data.csv'):
    fieldnames = ['Frame', 'EAR', 'MAR', 'MOE', 'Drowsy', 'Left Eye', 'Right Eye', 'Mouth']
    is_file_empty = not os.path.isfile(filename) or os.stat(filename).st_size == 0
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if is_file_empty:
            writer.writeheader()  # Write header only if file is empty
        writer.writerow(data)


def process_frame(frame):
    global frame_buffer, drowsiness_score, frame_count
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 60)])

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        puc = calculate_puc(ear, ear_threshold, frame_buffer)
        moe = calculate_moe(mar, ear)

        adjust_drowsiness_score(ear, mar, puc, moe)
        
        drowsy = True if drowsiness_score >= 10 else False
        savetocsv.createdata(drowsy,ear,mar,puc,moe,frame_count)
        # # Write data to CSV
        # drowsy = True if drowsiness_score >= 10 else False
        # data = {'Frame': frame_count, 'EAR': ear, 'MAR': mar, 'MOE': moe, 'Drowsy': drowsy,
        #         'Left Eye': leftEAR, 'Right Eye': rightEAR, 'Mouth': mar}
        # write_to_csv(data)

        for eye, ear_value, label in [(leftEye, leftEAR, '(L)'), (rightEye, rightEAR, '(R)')]:
            x1, y1 = np.min(eye, axis=0)
            x2, y2 = np.max(eye, axis=0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            if eye is leftEye:
                text = f"{label} {ear_value:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x1 - text_size[0] - 20
            else:
                text = f"{label} {ear_value:.2f}"
                text_x = x2 + 20

            cv2.putText(frame, text, (text_x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        mx1, my1 = np.min(mouth, axis=0)
        mx2, my2 = np.max(mouth, axis=0)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 0, 0), 1)
        mouth_text = f"MAR: {mar:.2f}"

        cv2.putText(frame, mouth_text, (mx1, my2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        drowsiness_level = int(drowsiness_score)
        cv2.putText(frame, f'Drowsiness Level: {drowsiness_level}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        dot_color = (0, 255 * (1 - drowsiness_level / 10), 255 * drowsiness_level / 10)
        cv2.circle(frame, (frame.shape[1] - 20, 20), 10, dot_color, -1)

    return frame, int(drowsiness_score)


# Camera loop
def run_camera_loop():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
    else:
        start_time = time.time()  # Get the start time
        while True:
            ret, frame = cap.read()
            processed_frame, drowsiness_level = process_frame(frame)
            cv2.imshow('Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 120:  # Check if 2 minutes have passed
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_loop()
