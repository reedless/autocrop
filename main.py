from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route("/")
def home():
    return "autocrop v0.1"

@app.route("/autocrop", methods=["POST"])
def autocrop():
    try:
        # Get the JSON data from the request
        data = request.json

        # Extract the base64 string, object_name, height, and width from the JSON data
        image_data = data.get("image_data")
        object_name = data.get("object_name")
        height = data.get("height")
        width = data.get("width")

        # Decode the base64 string and open it as an image using PIL
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Perform the autocrop operation on the image with the given object_name, height, and width

        # ... (Implement your autocrop logic here) ...

        # Encode the cropped image as base64 string
        cropped_image_data = base64.b64encode(image.tobytes()).decode("utf-8")

        # Return the cropped image as a JSON response
        response = {"cropped_image_data": cropped_image_data}
        return jsonify(response)

    except Exception as e:
        # Return an error message if there's any exception
        response = {"error": str(e)}
        return jsonify(response), 400

if __name__ == "__main__":
    app.run()

