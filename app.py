import base64
import os

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

from models.ensemble import attempt_load
from utils.general import (check_img_size, letterbox, non_max_suppression,
                           scale_coords)

app = Flask(__name__)

# Load model
device = torch.device('cpu')
weights = './yolov7.pt'
print("Loading model...")
model = attempt_load(weights, map_location=device)  # load FP32 model
print("Model loaded")

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>autocrop v0.1</title>
            <style>
                pre {
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                font-family: 'Courier New', Courier, monospace;
                }
            </style>
        </head>
        <body>
            <h1>autocrop v0.1 by reedless</h1>

            <p>
                A simple auto cropping app that takes in an image and a prompt 
                (currently only supports COCO classes), and returns a cropped image 
                512x512 with the subject in the centre. Created by https://github.com/reedless.
            </p>

            <h2>Code Example</h2>
            <pre>
                <code>
                    import base64

                    import cv2
                    import numpy as np
                    import requests

                    # URL to post the image
                    url = "https://autocrop-production.up.railway.app/autocrop"

                    # Prompt
                    prompt = 'cat'

                    # Path to the image file
                    image_path = f"./test_{prompt}.jpg"

                    # Send POST request
                    files = {'image': open(image_path, 'rb')}
                    params = {'prompt': prompt}
                    response = requests.post(url, files=files, data=params)

                    if response.status_code == 200:
                        # Get the returned image and decode from base64
                        cropped_image_data = response.json()['cropped_image_data']
                        jpg_original = base64.b64decode(cropped_image_data)
                        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                        img = cv2.imdecode(jpg_as_np, flags=1)

                        # Save the image using cv2
                        save_path = "./received_from_post.jpg"
                        cv2.imwrite(save_path, img)
                        print(f" The image with the result is saved in: {save_path}")
                    else:
                        print(response.status_code)
                        print(response.text)
                </code>
            </pre>
        </body>
    </html>
    """


@app.route("/autocrop", methods=["POST"])
def autocrop():
    # try:
    prompt = request.form.get('prompt')
    if prompt is None:
        return "No prompt provided"

    if 'image' not in request.files:
        return "No image file provided"
    image_file = request.files['image']

    # get some variables
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    imgsz = check_img_size(640, s=stride)  # check img_size

    # Read the image as numpy array
    image_data = np.frombuffer(image_file.read(), np.uint8)

    # Load the image as cv2 image
    im0 = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Padded resize
    img = letterbox(im0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img / 255  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    print("Starting prediction...")
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    print("Applying NMS...")
    conf_thres = 0.25
    iou_thres = 0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    # Process detections
    for _, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).int()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                if prompt in label:
                    x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    H, W, _ = im0.shape
                    if H == W:
                        cropped = im0 # no need to crop if is already a square
                    else:
                        y_midpoint = (y1+y0)/2
                        x_midpoint = (x1+x0)/2

                        if H > W:
                            x_left = 0
                            x_right = W
                            if y_midpoint < W/2:
                                y_top = 0
                                y_bottom = W
                            elif y_midpoint > (H - W/2):
                                y_top = H - W
                                y_bottom = H
                            else:
                                y_top = y_midpoint - W/2
                                y_bottom = y_midpoint + W/2

                        if W > H:
                            y_top = 0
                            y_bottom = H
                            if x_midpoint < H/2:
                                x_left = 0
                                x_right = H
                            elif y_midpoint > (W - H/2):
                                x_left = W - H
                                x_right = W
                            else:
                                x_left = x_midpoint - H/2
                                x_right = x_midpoint + H/2

                        y_top, y_bottom, x_left, x_right = int(y_top), int(y_bottom), int(x_left), int(x_right)
                        cropped = im0[y_top:y_bottom, x_left:x_right]
                    
                    resized = cv2.resize(cropped, (512,512))
                    # save_path = './cropped_and_resized.jpg'
                    # cv2.imwrite(save_path, resized)
                    # print(f" The image with the result is saved in: {save_path}")
                    break

    # Encode the cropped image as base64 string
    string = base64.b64encode(cv2.imencode('.jpg', resized)[1]).decode()
    
    # Return the cropped image as a JSON response
    response = {"cropped_image_data": string}
    return jsonify(response), 200

    # except Exception as e:
    #     print(e)
    #     response = {"error": str(e)}
    #     return jsonify(response), 400

if __name__ == "__main__":
    app.run(port=os.getenv("PORT", default=5000))
