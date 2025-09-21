# Dockerfile for the main FastAPI API server
FROM python:3.11-slim
WORKDIR /app

# 0. Set environment variable for PIP timeout (The FIX)
ENV PIP_DEFAULT_TIMEOUT=100

# 1. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the application files
# These files are in the project root, so the path is clean
COPY main.py .
COPY database.py .
COPY models.py .
COPY .env .

# 3. Expose the port FastAPI uses
EXPOSE 8000

# 4. Command to run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
