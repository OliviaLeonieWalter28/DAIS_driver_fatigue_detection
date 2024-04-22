from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from RunScript import process_frame

app = Flask(__name__)


@app.route('/process_frame', methods=['POST'])
def process_frame_web():
    data = request.get_json()
    image_data = data['image']
    header, encoded = image_data.split(",", 1)
    binary = base64.b64decode(encoded)
    image = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    processed_frame, drowsiness_level = process_frame(frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    response_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': 'data:image/jpeg;base64,' + response_frame, 'drowsinessLevel': drowsiness_level})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
