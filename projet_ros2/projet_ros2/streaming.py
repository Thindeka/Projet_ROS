import cv2
import flask
from flask import Response

app = flask.Flask(__name__)
cap = cv2.VideoCapture(2,cv2.CAP_AVFOUNDATION) #0,1,..... Id de la caméra.

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=8080, threaded=True)