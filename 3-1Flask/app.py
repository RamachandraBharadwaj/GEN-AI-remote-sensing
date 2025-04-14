import os
import onnxruntime
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from PIL import Image
from io import BytesIO
import gc  # Garbage collector for memory management
from patchify import patchify
import cv2

app = Flask(__name__)

# Load ONNX models
onnx_model_path = "sar2rgb.onnx"
onnx_sess = onnxruntime.InferenceSession(onnx_model_path)

# Load the ViT ONNX model
vit_onnx_model_path = "vit_model.onnx"
vit_onnx_sess = onnxruntime.InferenceSession(vit_onnx_model_path)

# Load the flood ONNX model
flood_onnx_model_path = "flood_model.onnx"
flood_onnx_sess = onnxruntime.InferenceSession(flood_onnx_model_path)

# ViT class names
class_names = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']

@app.route('/predict_vit', methods=['POST'])
def predict_vit():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    gc.collect()  # Clear previous memory usage

    try:
        # Read and preprocess the image for ViT using NumPy
        img = Image.open(BytesIO(img_file.read())).convert('RGB')  # Ensure RGB format
        img = img.resize((224, 224))  # Resize to ViT input size (224x224)
        img_array = np.array(img).astype(np.float32)  # Convert image to NumPy array

        # Normalize image (values in range [-1, 1])
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]

        # Convert HWC to CHW format and add batch dimension
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run ONNX ViT model inference
        inputs = {vit_onnx_sess.get_inputs()[0].name: img_array}
        outputs = vit_onnx_sess.run(None, inputs)

        # Get the predicted class index
        predicted_class_index = np.argmax(outputs[0], axis=1).item()  # Get index of the highest score
        predicted_class = class_names[predicted_class_index]  # Map to the class name
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/flood', methods=['POST'])
def flood_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']

    # Read the image file into memory
    img = Image.open(BytesIO(img_file.read()))
    img = img.convert("RGB")  # Ensure the image is in RGB mode

    # Preprocess the image for prediction
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0

    # Patchify the image for model input
    patch_shape = (16, 16, 3)
    patches = patchify(img_array, patch_shape, 16)
    patches = np.reshape(patches, (-1, patch_shape[0] * patch_shape[1] * patch_shape[2]))
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    # Run the ONNX flood model
    inputs = {flood_onnx_sess.get_inputs()[0].name: patches}
    pred = flood_onnx_sess.run(None, inputs)[0]

    # Post-process the flood prediction (threshold, edges, etc.)
    pred = np.reshape(pred, (256, 256, 1))
    pred = (pred > 0.5).astype(np.uint8)  # Threshold prediction

    # Find edges of the flood region using Canny edge detection
    pred_edges = cv2.Canny(pred[:, :, 0] * 255, 100, 200)

    # Make edges thicker using dilation
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel (3x3 for moderate thickness)
    thicker_edges = cv2.dilate(pred_edges, kernel, iterations=1)

    # Create a blank RGB image to draw the thicker edges
    outline_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    outline_mask[:, :, 2] = thicker_edges  # Set the thicker edges to blue

    # Overlay the outline onto the original image
    img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
    combined_image = cv2.addWeighted(img_array, 0.9, outline_mask, 0.3, 0)

    output = BytesIO()
    combined_pil_image = Image.fromarray(combined_image)
    combined_pil_image.save(output, format="PNG")
    output.seek(0)

    # Return the image as a response
    return Response(output.getvalue(), mimetype='image/png')
@app.route('/predict2', methods=['POST'])
def predict_onnx():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']

    # Clear any previous image and data before loading new one
    gc.collect()

    # Read and preprocess the image for the ONNX model
    img = Image.open(BytesIO(img_file.read()))
    img = img.resize((256, 256))  # Adjust size as needed
    print(np.array(img))
    img = np.array(img).transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run the ONNX model
    inputs = {onnx_sess.get_inputs()[0].name: img}
    output = onnx_sess.run(None, inputs)

    # Post-process the output image
    output_image = output[0].squeeze().transpose(1, 2, 0)  # CHW to HWC
    output_image = (output_image + 1) / 2  # Normalize to [0, 1]
    output_image = (output_image * 255).astype(np.uint8)  # Denormalize to [0, 255]

    # Convert to Image and return as response
    output_image = Image.fromarray(output_image)
    img_byte_arr = BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
