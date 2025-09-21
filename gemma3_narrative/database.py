from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DATABASE_USER")
DB_PASS = os.getenv("DATABASE_PASS")
DB_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")

# Build the connection string
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

# Add pooling parameters for robust production use
engine = create_async_engine(
    DATABASE_URL, 
    echo=False, # Set to True for debugging SQL
    pool_size=10, 
    max_overflow=5
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
