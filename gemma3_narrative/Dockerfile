FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=100
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py .
COPY database.py .
COPY .env .
COPY main.py .
COPY wait-for-postgres.sh .
RUN chmod +x ./wait-for-postgres.sh
RUN apt-get update && apt-get install -y netcat-traditional
CMD ["./wait-for-postgres.sh", "db", "5432", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
