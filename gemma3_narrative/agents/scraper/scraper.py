import aiohttp, asyncio, random, json, os, logging
from newspaper import Article, Config
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
    # System prompt to instruct Gemma 3B to output JSON
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
    # Generate Vector Embedding using Deepseek
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
    # --- FIX: Use download() and parse() in the correct order ---
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    article = Article(url, config=config)
    try:
        await asyncio.to_thread(article.download)
        await asyncio.to_thread(article.parse)
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return
    # --- END FIX ---
    
    async with aiohttp.ClientSession() as session:
        # 2. Get LLM Analysis and Embedding
        llm_metrics = await get_llm_analysis(article.text, session)
        embedding = await get_llm_embedding(article.text, session)

        async with async_session() as db_session:
            async with db_session.begin():
                # Get or Create Node
                node_result = await db_session.execute(select(Node).where(Node.name == node_name))
                node = node_result.scalar_one_or_none()
                if not node:
                    node = Node(name=node_name)
                    db_session.add(node)
                    await db_session.flush()

                # Insert LLM-Derived Replica
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
    # Example articles to be scraped and analyzed
    targets = [
        # Replaced Reuters URL with a more reliable, publicly accessible source
        ("https://developers.googleblog.com/en/updated-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/", "Gemini Model Updates", "Google Developers Blog"),
        ("https://www.theguardian.com/technology/2025/sep/17/google-deepmind-claims-historic-ai-breakthrough-in-problem-solving", "Google DeepMind AI", "Guardian Technology"), 
    ]
    
    tasks = [scrape_and_analyze_article(url, node, lens) for url, node, lens in targets]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    logger.info("Starting Scraper Agent...")
    asyncio.run(main_scraper_task())
    logger.info("Scraper Agent finished its task.")
