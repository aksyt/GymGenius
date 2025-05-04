from flask import Flask, render_template, Response
from bicep_curl import bicep_curl_detection
from squats import squat_detection  # Assumes squat_detection(cap) is defined in squats.py
from pushups import push_up_detection  # Import the push-up detection function
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bicep-curls')
def bicep_curls():
    return render_template('bicep-curls.html')

@app.route('/bicep-video')
def bicep_video():
    return Response(bicep_curl_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squats')
def squats():
    return render_template('squats.html')

@app.route('/squats-video')
def squats_video():
    return Response(squat_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

# New routes for push-ups
@app.route('/push-ups')
def push_ups():
    return render_template('push-ups.html')

@app.route('/push-ups-video')
def push_ups_video():
    return Response(push_up_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)