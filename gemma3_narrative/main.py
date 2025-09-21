import asyncio, logging, random, os
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select, update
from models import Base, Node, Replica, ReplicaHistory
from database import engine, async_session

# --- Logging Setup ---
logger = logging.getLogger("gemma3")
logger.setLevel(logging.INFO)
# Basic console handler for Docker logs
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
            await ws.receive_text() # Keep connection alive by waiting for data
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
            # Use session.begin() for atomic transaction
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
                        audience_demographics=metrics.get("audience_demographics"),
                        # Note: Embedding is updated by the scraper agent, not here.
                    )
                )
                await session.execute(stmt)

                # Log to history
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

        # Trigger alert if distortion > threshold
        if metrics.get("distortion", 0) > 50:
            await send_alert(f"Distortion alert: Replica {replica_id} distortion={metrics.get('distortion')}")
        
        return {"status": "ok", "replica_id": replica_id}

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
                        "embedding": r.embedding # Includes the new vector field
                    }
                    for r in replicas
                ]
            }
    except Exception as e:
        logger.error(f"Analyze node error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# --- Periodic Update Task (Simulates Data Drift) ---
async def periodic_update():
    while True:
        await asyncio.sleep(10)  # Every 10 seconds
        try:
            async with async_session() as session:
                async with session.begin():
                    replicas_result = await session.execute(select(Replica))
                    replicas = replicas_result.scalars().all()
                    
                    for r in replicas:
                        # Randomly drift distortion
                        r.distortion = min(100, max(0, (r.distortion or 0) + random.uniform(-2, 2)))
                        r.accuracy = min(100, max(0, (r.accuracy or 0) + random.uniform(-1, 1)))
                        
                        # Save historical snapshot
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
