from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
# We don't import TypeDecorator or SQLAlchemy-Utils anymore

Base = declarative_base()

# --- Custom Type for pgvector (Native SQLAlchemy 2.0 style) ---
# Define a class that is compatible with the PostgreSQL dialect
class Vector(String):
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim

    def get_col_spec(self, **kw):
        # This tells PostgreSQL to use the native VECTOR(dim) type
        return f"VECTOR({self.dim})"
    
    def process_bind_param(self, value, dialect):
        # Convert Python list/array into the Postgres string format: [x,y,z]
        if value is not None:
            return "[" + ",".join(map(str, value)) + "]"
        return value

    def process_result_value(self, value, dialect):
        # Convert Postgres string format: [x,y,z] back to Python list of floats
        if value is not None:
            try:
                # Remove brackets and split by comma
                return [float(x.strip()) for x in value.strip('[]').split(',')]
            except ValueError:
                return None
        return value
# -------------------------------------------------------------

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
    # Use the new, native Vector type
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
