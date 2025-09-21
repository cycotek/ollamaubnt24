#!/bin/bash
# -------------------------------------------------------------
# Project Structure Builder: gemma3_narrative
# Creates directories and populates all core files with corrected content.
# This script is designed to be run after setup_server.sh on a fresh system.
# -------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="gemma3_narrative"

# --- 1. Create Directory Structure ---
echo "🏗️ Creating project directories..."
mkdir -p "$PROJECT_ROOT"/{agents/scraper,ollama,agents/insert_nodes}

# --- 2. Create Core Files in Project Root ---
echo "📝 Creating core files in $PROJECT_ROOT/..."
cd "$PROJECT_ROOT"

# .env file
tee ".env" > /dev/null <<'EOF'
# --- .env Configuration ---
# CRITICAL: CHANGE THESE VALUES! Use the strong password from setup_server.sh.
DATABASE_USER=agent_user
DATABASE_PASS=YOUR_STRONG_POSTGRES_PASSWORD
DATABASE_HOST=host.docker.internal
DATABASE_NAME=aiserver_db
API_KEY=YOUR_SECRET_API_KEY
EOF

# requirements.txt
tee "requirements.txt" > /dev/null <<'EOF'
fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
python-dotenv
aiohttp
newspaper3k
pydantic
lxml
lxml-html-clean
EOF

# main.py
tee "main.py" > /dev/null <<'EOF'
import asyncio, logging, random, os
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select, update
from models import Base, Node, Replica, ReplicaHistory
from database import engine, async_session

# --- Logging Setup ---
logger = logging.getLogger("gemma3")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# --- App Setup ---
app = FastAPI(title="Gemma3 Narrative Analysis API")
API_KEY = os.getenv("API_KEY", "YOUR_SECRET_KEY")
websockets = []

# --- Database Init ---
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("startup")
async def startup():
    await init_db()
    asyncio.create_task(periodic_update())
    logger.info("FastAPI server started and DB initialized.")

# --- API Key Verification Middleware ---
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- WebSocket Alerts ---
@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws.accept()
    websockets.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        websockets.remove(ws)
        logger.info("WebSocket disconnected.")

async def send_alert(message: str):
    logger.warning(f"Sending ALERT: {message}")
    for ws in websockets:
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")

# --- Update Metrics Endpoint ---
@app.post("/update_metrics")
async def update_metrics(replica_id: int, metrics: dict, x_api_key: str = Header(...)):
    await verify_api_key(x_api_key)
    try:
        async with async_session() as session:
            async with session.begin():
                stmt = (
                    update(Replica)
                    .where(Replica.id == replica_id)
                    .values(
                        accuracy=metrics.get("accuracy"),
                        distortion=metrics.get("distortion"),
                        reach=metrics.get("reach"),
                        sentiment=metrics.get("sentiment"),
                        source_credibility=metrics.get("source_credibility"),
                        audience_demographics=metrics.get("audience_demographics")
                    )
                )
                await session.execute(stmt)

                history = ReplicaHistory(
                    replica_id=replica_id,
                    accuracy=metrics.get("accuracy"),
                    distortion=metrics.get("distortion"),
                    reach=metrics.get("reach"),
                    sentiment=metrics.get("sentiment"),
                    source_credibility=metrics.get("source_credibility"),
                    audience_demographics=metrics.get("audience_demographics")
                )
                session.add(history)

        if metrics.get("distortion", 0) > 50:
            await send_alert(f"Distortion alert: Replica {replica_id} distortion={metrics.get('distortion')}")
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Failed to update metrics for replica {replica_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# --- Analyze Node Endpoint ---
@app.get("/analyze_node/{node_id}")
async def analyze_node(node_id: int, x_api_key: str = Header(...)):
    await verify_api_key(x_api_key)
    try:
        async with async_session() as session:
            node_result = await session.execute(select(Node).where(Node.id == node_id))
            node = node_result.scalar_one_or_none()
            if not node:
                raise HTTPException(status_code=404, detail="Node not found")

            replica_result = await session.execute(select(Replica).where(Replica.node_id == node.id))
            replicas = replica_result.scalars().all()

            return {
                "node": node.name,
                "replicas": [
                    {
                        "lens": r.lens,
                        "accuracy": r.accuracy,
                        "distortion": r.distortion,
                        "reach": r.reach,
                        "sentiment": r.sentiment,
                        "source_credibility": r.source_credibility,
                        "audience_demographics": r.audience_demographics,
                        "embedding": r.embedding
                    }
                    for r in replicas
                ]
            }
    except Exception as e:
        logger.error(f"Analyze node error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# --- Periodic Update Task ---
async def periodic_update():
    while True:
        await asyncio.sleep(10)
        try:
            async with async_session() as session:
                async with session.begin():
                    replicas_result = await session.execute(select(Replica))
                    replicas = replicas_result.scalars().all()
                    
                    for r in replicas:
                        r.distortion = min(100, max(0, (r.distortion or 0) + random.uniform(-2, 2)))
                        r.accuracy = min(100, max(0, (r.accuracy or 0) + random.uniform(-1, 1)))
                        
                        session.add(ReplicaHistory(
                            replica_id=r.id,
                            accuracy=r.accuracy,
                            distortion=r.distortion,
                            reach=r.reach,
                            sentiment=r.sentiment,
                            source_credibility=r.source_credibility,
                            audience_demographics=r.audience_demographics
                        ))
                    
                    logger.info(f"Periodic update simulated drift on {len(replicas)} replicas.")

        except Exception as e:
            logger.error(f"Periodic update error: {e}")
EOF

tee "database.py" > /dev/null <<'EOF'
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DATABASE_USER")
DB_PASS = os.getenv("DATABASE_PASS")
DB_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=10, 
    max_overflow=5
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
EOF

tee "models.py" > /dev/null <<'EOF'
from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator
import json

Base = declarative_base()

class Vector(TypeDecorator):
    impl = String
    cache_ok = True

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def get_col_spec(self, **kw):
        return f"VECTOR({self.dim})"
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return "[" + ",".join(map(str, value)) + "]"
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                return [float(x.strip()) for x in value.strip('[]').split(',')]
            except ValueError:
                return None
        return value

class Node(Base):
    __tablename__ = "nodes"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    pi = Column(Boolean, default=True)
    last_update = Column(TIMESTAMP, server_default=func.now())

class Replica(Base):
    __tablename__ = "replicas"
    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    lens = Column(String, nullable=False)
    accuracy = Column(Float)
    distortion = Column(Float)
    reach = Column(Float)
    sentiment = Column(Float, default=0)
    source_credibility = Column(Float, default=50)
    audience_demographics = Column(JSON, default={})
    embedding = Column(Vector(dim=768))

class ReplicaHistory(Base):
    __tablename__ = "replica_history"
    id = Column(Integer, primary_key=True)
    replica_id = Column(Integer, ForeignKey("replicas.id"), nullable=False)
    timestamp = Column(TIMESTAMP, server_default=func.now())
    accuracy = Column(Float)
    distortion = Column(Float)
    reach = Column(Float)
    sentiment = Column(Float)
    source_credibility = Column(Float)
    audience_demographics = Column(JSON)
EOF

tee "ollama/Dockerfile" > /dev/null <<'EOF'
# Base image from NVIDIA, guaranteed to have correct CUDA drivers and binaries
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for Ollama (curl, git) and common tools (jq)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    jq \
    # Clean up APT cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install Ollama from the official installer script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the startup script into the container
COPY ollama_startup.sh /usr/local/bin/ollama_startup.sh

# Give the script executable permissions inside the container as the root user
RUN chmod +x /usr/local/bin/ollama_startup.sh

# Set a non-root user (optional but good practice)
RUN useradd -ms /bin/bash ollama
USER ollama
WORKDIR /home/ollama

# Set the entrypoint to a script that will manage startup and model pulls
ENTRYPOINT ["/usr/local/bin/ollama_startup.sh"]
EOF

tee "ollama/ollama_startup.sh" > /dev/null <<'EOF'
#!/bin/bash
# Script to ensure models are available before starting the server

echo "Starting Ollama model pull and serve sequence..."

# Pull Gemma 3B
ollama pull gemma:3b

# Pull Deepseek 7B
ollama pull deepseek-llm:7b

# The gpt-oss:20b model may exceed your GPU's VRAM (8GB)
# Uncomment the line below if you want to attempt to download it.
# ollama pull gpt-oss:20b

echo "Model downloads complete. Starting Ollama server..."

# Start the Ollama server and keep it running
exec ollama serve
EOF

tee "agents/scraper/Dockerfile" > /dev/null <<'EOF'
FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=100
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py .
COPY database.py .
COPY .env .
COPY agents/scraper/scraper.py .
CMD ["python", "scraper.py"]
EOF

tee "agents/scraper/scraper.py" > /dev/null <<'EOF'
import aiohttp, asyncio, random, json, os, logging
from newspaper import Article
from sqlalchemy import select
from models import Node, Replica
from database import async_session

logger = logging.getLogger("scraper_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

async def get_llm_analysis(text: str, session: aiohttp.ClientSession):
    system_prompt = (
        "You are a narrative analysis engine. Analyze the text for sentiment, credibility, and distortion. "
        "Output a single JSON object with keys: sentiment (float -1.0 to 1.0), "
        "source_credibility (int 0 to 100), distortion (int 0 to 100), and "
        "audience_demographics (JSON object, e.g., {'age': '25-35', 'region': 'NA'})."
    )
    
    analysis_payload = {
        "model": "gemma:3b",
        "system": system_prompt,
        "prompt": text[:4000],
        "format": "json",
        "stream": False
    }

    try:
        async with session.post(f"{OLLAMA_HOST}/api/generate", json=analysis_payload) as resp:
            data = await resp.json()
            return json.loads(data.get("response", "{}"))
    except Exception as e:
        logger.error(f"Ollama Analysis Error: {e}")
        return {}

async def get_llm_embedding(text: str, session: aiohttp.ClientSession):
    embedding_payload = {
        "model": "deepseek-llm:7b",
        "prompt": text[:2000] 
    }
    try:
        async with session.post(f"{OLLAMA_HOST}/api/embeddings", json=embedding_payload) as resp:
            data = await resp.json()
            return data.get("embedding")
    except Exception as e:
        logger.error(f"Ollama Embedding Error: {e}")
        return None

async def scrape_and_analyze_article(url, node_name, lens):
    async with aiohttp.ClientSession() as session:
        try:
            article = Article(url, fetch_images=False, request_timeout=60, keep_article_html=False, headers={'User-Agent': 'Mozilla/5.0'})
            await asyncio.to_thread(article.download)
            await asyncio.to_thread(article.parse)
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return

        llm_metrics = await get_llm_analysis(article.text, session)
        embedding = await get_llm_embedding(article.text, session)

        async with async_session() as db_session:
            async with db_session.begin():
                node_result = await db_session.execute(select(Node).where(Node.name == node_name))
                node = node_result.scalar_one_or_none()
                if not node:
                    node = Node(name=node_name)
                    db_session.add(node)
                    await db_session.flush()

                replica = Replica(
                    node_id=node.id,
                    lens=lens,
                    accuracy=random.uniform(70,95),
                    distortion=llm_metrics.get("distortion", 0),
                    reach=random.uniform(50,90),
                    sentiment=llm_metrics.get("sentiment", 0),
                    source_credibility=llm_metrics.get("source_credibility", 50),
                    audience_demographics=llm_metrics.get("audience_demographics", {}),
                    embedding=embedding
                )
                db_session.add(replica)
                logger.info(f"Successfully inserted Replica for Node: {node_name} with lens: {lens}")

async def main_scraper_task():
    targets = [
        ("https://www.theguardian.com/technology/2025/sep/17/google-deepmind-claims-historic-ai-breakthrough-in-problem-solving", "Google DeepMind AI", "Guardian Technology"),
        ("https://developers.googleblog.com/en/updated-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/", "Gemini Model Updates", "Google Developers Blog"),
    ]
    
    tasks = [scrape_and_analyze_article(url, node, lens) for url, node, lens in targets]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    logger.info("Starting Scraper Agent...")
    asyncio.run(main_scraper_task())
    logger.info("Scraper Agent finished its task.")
EOF

tee "agents/insert_nodes/Dockerfile" > /dev/null <<'EOF'
FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=100
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py .
COPY database.py .
COPY .env .
COPY agents/insert_nodes/insert_nodes.py .
CMD ["python", "insert_nodes.py"]
EOF

tee "agents/insert_nodes/insert_nodes.py" > /dev/null <<'EOF'
from database import async_session
from models import Node, Replica
from sqlalchemy import select
import asyncio, random, logging, os

logger = logging.getLogger("node_inserter")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

async def insert_example_node(node_name="Ferguson Protests"):
    async with async_session() as session:
        node_result = await session.execute(select(Node).where(Node.name == node_name))
        if node_result.scalar_one_or_none():
            logger.info(f"Node '{node_name}' already exists. Skipping insertion.")
            return

        async with session.begin():
            node = Node(name=node_name)
            session.add(node)
            await session.flush()

            lenses = ["structural", "human", "technical", "comparative"]
            for lens in lenses:
                replica = Replica(
                    node_id=node.id,
                    lens=lens,
                    accuracy=random.uniform(70, 95),
                    distortion=random.uniform(0, 5),
                    reach=random.uniform(50, 90),
                    sentiment=random.uniform(-1, 1),
                    source_credibility=random.randint(40,100),
                    audience_demographics={"age": random.randint(18,65), "region":"NA"}
                )
                session.add(replica)
            
            logger.info(f"Successfully inserted new Node '{node_name}' and {len(lenses)} replicas.")

if __name__ == "__main__":
    logger.info("Starting Node Insertion Agent...")
    asyncio.run(insert_example_node())
    logger.info("Node Insertion Agent finished.")
EOF
```
eof

### Final Instructions

1.  **Remove the problematic `ollama_startup.sh` directory:**
    ```bash
    sudo rm -r ollama_startup.sh/
    ```
2.  **Run the `build_project_structure.sh` script.** This will create all the files with the correct content and permissions.
    ```bash
    ./build_project_structure.sh
    ```
3.  **Perform the final deployment.**
    ```bash
    docker compose up -d --build
