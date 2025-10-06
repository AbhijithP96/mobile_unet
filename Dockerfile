# ------------------------------------------------------------------------------
# Dockerfile: Model Inference API
#
# This Dockerfile builds a lightweight container for serving a trained model
# via a FastAPI + Uvicorn endpoint. The container exposes an HTTP inference
# API at http://0.0.0.0:8000, making it suitable for deployment in production.
# ------------------------------------------------------------------------------

# Use official Python 3.11 base image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install required dependencies for inference and API serving
RUN pip install numpy opencv-python-headless tensorflow mlflow fastapi uvicorn

# Copy source code, saved model, and inference scripts into the container
COPY ./src/ /app/src/
COPY ./saved_model /app/saved_model
COPY infer.py /app/
COPY main.py /app/

# Expose port 8000 for FastAPI server
EXPOSE 8000

# Start the FastAPI application with Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]