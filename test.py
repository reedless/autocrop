import base64

import cv2
import numpy as np
import requests


def post_image(url, image_path):
    files = {'image': open(image_path, 'rb')}
    params = {'prompt': 'cat'}
    
    # Send POST request
    response = requests.post(url, files=files, data=params)

    return response

# URL to post the image
url = "http://127.0.0.1:5000/autocrop"
# url = "https://autocrop-production.up.railway.app/autocrop"

# Path to the image file
image_path = "./test_cat.jpg"

# Call the function to post the image
response = post_image(url, image_path)

# Get the returned image
cropped_image_data = response.json()['cropped_image_data']
jpg_original = base64.b64decode(cropped_image_data)
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
img = cv2.imdecode(jpg_as_np, flags=1)

# Display the image using cv2
save_path = "./received_from_post.jpg"
cv2.imwrite(save_path, img)
print(f" The image with the result is saved in: {save_path}")
