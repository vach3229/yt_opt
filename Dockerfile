FROM python:3.11-slim

# System packages for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]