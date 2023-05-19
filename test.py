import requests
import base64
import cv2
import numpy as np

def post_image(url, image_path):
    files = {'image': open(image_path, 'rb')}

    # Send POST request
    response = requests.post(url, files=files)

    return response

# URL to post the image
url = "http://127.0.0.1:5000/autocrop"

# Path to the image file
image_path = "./test.jpg"

# Call the function to post the image
response = post_image(url, image_path)

# Get the returned image
cropped_image_data = response.json()['cropped_image_data']
jpg_original = base64.b64decode(cropped_image_data)
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
img = cv2.imdecode(jpg_as_np, flags=1)

# Display the image using cv2
cv2.imwrite("./received_from_post.jpg", img)

