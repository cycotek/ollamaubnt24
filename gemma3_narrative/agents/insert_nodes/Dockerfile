FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the actual agent code
COPY ./agents/insert_nodes/insert_nodes.py .
CMD ["python", "insert_nodes.py"]
