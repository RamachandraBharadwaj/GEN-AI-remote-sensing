FROM python:3.12-slim

# Create and set the working directory
WORKDIR /app

# Copy the entire application code
COPY app.py .
COPY flood_model.onnx .
COPY sar2rgb.onnx .
COPY vit_model.onnx .

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your application will run on
EXPOSE 5000

# Specify the command to run on container start
CMD ["python", "app.py"]