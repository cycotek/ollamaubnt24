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
        # Check if node already exists
        node_result = await session.execute(select(Node).where(Node.name == node_name))
        if node_result.scalar_one_or_none():
            logger.info(f"Node '{node_name}' already exists. Skipping insertion.")
            return

        async with session.begin():
            node = Node(name=node_name)
            session.add(node)
            await session.flush()  # Get node.id

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
                    # Embedding is left as NULL here, as this agent is not LLM-powered
                )
                session.add(replica)
            
            logger.info(f"Successfully inserted new Node '{node_name}' and {len(lenses)} replicas.")

if __name__ == "__main__":
    logger.info("Starting Node Insertion Agent...")
    asyncio.run(insert_example_node())
    logger.info("Node Insertion Agent finished.")
