# app.py

from flask import Flask, render_template, Response
from bicep_curl import bicep_curl_detection
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bicep-curls')
def bicepCurls():
    return render_template('bicep-curls.html')

@app.route('/video')
def video():
    return Response(bicep_curl_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
