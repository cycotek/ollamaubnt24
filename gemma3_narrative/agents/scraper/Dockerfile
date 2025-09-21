# Dockerfile for the Scraper Agent

FROM python:3.11-slim
WORKDIR /app

# 1. Set environment variable for PIP timeout
ENV PIP_DEFAULT_TIMEOUT=100

# 2. Copy dependencies file from project root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the application files that the agent needs to run
COPY models.py .
COPY database.py .
COPY .env .

# 4. Copy the specific agent code from its path
COPY agents/scraper/scraper.py .

# The command to run the agent script
CMD ["python", "scraper.py"]
