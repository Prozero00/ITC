"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image

import torch
import subprocess
import mimetypes
from flask import Flask, render_template, request, redirect, send_file, jsonify, Response
from werkzeug.utils import secure_filename
from camera import VideoCamera
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
root_directory = 'runs/detect/'



@app.route("/", methods=["GET", "POST"])
def predict():
    uploaded_image = None
    result_filename = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if not file:
            return jsonify({'error': 'No file provided'})

        # Check if the uploaded file is an image or video
        file_type, _ = mimetypes.guess_type(file.filename)

        if file_type and file_type.startswith('image'):
            # Process image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            results = model([img])

            # Extract the processed image from the list
            processed_img = results.render()[0]

            result_filename = f"result_{secure_filename(file.filename)}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            # Convert the processed image to a PIL image before saving
            Image.fromarray(processed_img).save(result_path)

        elif file_type and file_type.startswith('video'):
            # Process 
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            video = request.files['file']
            video_filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            

            # Run detection script on the uploaded video and capture output
            try:
                result = subprocess.run(
                    ['python', 'detect.py', '--source', video_path, '--weights', 'best.pt'],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Assuming the detection script generates an output file (modify this part accordingly)
                result_filename = secure_filename(video.filename)

                for root, dirs, files in os.walk(root_directory):
                    for file in files:
                        if file == result_filename:
                            # File found, set the result path
                            result_path = os.path.join(root, file)


                            return send_file(result_path, as_attachment=True, download_name=result_filename)

            except subprocess.CalledProcessError as e:
                return jsonify({'error': f'Error running detection script: {e.stderr.decode()}'})

        else:
            return jsonify({'error': 'Unsupported file type'})

        return send_file(result_path, as_attachment=True)

    return render_template("index.html", uploaded_image=uploaded_image, result_filename=result_filename)

@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.run(['python', 'detect.py', '--source', '0', '--weights', 'best.pt'], check=True)
    return "done"

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
