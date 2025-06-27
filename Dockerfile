# Use a base image with Python 3.11 and Debian-based system tools
FROM python:3.11-slim

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code into container
COPY . /app
WORKDIR /app

# Set environment variable for Flask
ENV PYTHONUNBUFFERED=1

# Expose port (not strictly necessary for Render)
EXPOSE 8000

# Start the app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]