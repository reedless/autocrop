import base64

import cv2
import numpy as np
import requests

# URL to post the image
# url = "http://127.0.0.1:5000/autocrop"
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
    # Get the returned image
    cropped_image_data = response.json()['cropped_image_data']
    jpg_original = base64.b64decode(cropped_image_data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    # Display the image using cv2
    save_path = "./received_from_post.jpg"
    cv2.imwrite(save_path, img)
    print(f" The image with the result is saved in: {save_path}")
else:
    print(response.status_code)
    print(response.text)