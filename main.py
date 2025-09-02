import os
import uuid
from typing import List, Dict, Any, Literal, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from thefuzz import fuzz

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

from sentence_transformers import SentenceTransformer

# Initialize the model (this will download it the first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Get embedding for text using sentence-transformers"""
    try:
        # Encode the text and convert to list (Pinecone expects list format)
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")


# Create or connect to Pinecone serverless index (BYOV)
index_name = "project-detection"
# Delete existing index if it exists with wrong dimensions
try:
    if pc.has_index(index_name):
        pc.delete_index(index_name)
except Exception as e:
    print(f"Error deleting index: {e}")

# Create new index
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension for all-MiniLM-L6-v2
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        },
    )

# Connect to the index
index = pc.Index(index_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://project-detect-lovat.vercel.app", "http://localhost:3000"],  # No trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project topics with their categories
topics = [
    {"topic": "Design and implementation of a smart irrigation system", "category": "IoT"},
    {"topic": "Development of a mobile health monitoring app", "category": "Healthcare"},
    {"topic": "Automated attendance system using facial recognition", "category": "Computer Vision"},
    {"topic": "IoT-based home security system", "category": "IoT"},
    {"topic": "Real-time traffic prediction using machine learning", "category": "Machine Learning"},
    {"topic": "Blockchain-based voting system", "category": "Blockchain"},
    {"topic": "AI-powered chatbot for customer service", "category": "AI"},
    {"topic": "Energy consumption optimization in smart grids", "category": "IoT"},
    {"topic": "Sentiment analysis of social media posts", "category": "NLP"},
    {"topic": "E-commerce product recommendation engine", "category": "Machine Learning"}
]

class TopicCheckRequest(BaseModel):
    topic: str
    top_k: int = 5  # Number of similar topics to return
    similarity_type: Literal['semantic', 'text'] = 'semantic'  # Type of similarity to use

class SimilarityResult(BaseModel):
    topic: str
    category: str
    score: float

async def initialize_topics():
    """Initialize the Pinecone index with sample topics."""
    try:
        # Force reinitialization by clearing existing vectors
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            # Delete all vectors in the index
            index.delete(delete_all=True)
            print("Cleared existing vectors from index")
            
        # Generate embeddings for all topics
        vectors = []
        for topic in topics:
            # Get embedding for the topic text
            embedding = get_embedding(topic['topic'])
            
            vector = {
                'id': str(uuid.uuid4()),
                'values': embedding,
                'metadata': {
                    'text': topic['topic'],
                    'category': topic['category']
                }
            }
            vectors.append(vector)
        
        # Upsert in batches (Pinecone has a limit on batch size)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        print(f"Successfully initialized {len(vectors)} topics in Pinecone")
            
    except Exception as e:
        print(f"Error initializing topics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize topics: {str(e)}")

@app.on_event("startup")
async def startup_event():
    await initialize_topics()

@app.post("/check_similarity", response_model=List[SimilarityResult])
async def check_similarity(request: TopicCheckRequest):
    print(f"Received request - Topic: {request.topic}, Type: {request.similarity_type}")
    try:
        if request.similarity_type == 'semantic':
            print("Using semantic similarity...")
            # Semantic similarity using Pinecone
            query_embedding = get_embedding(request.topic)
            print("Generated embedding successfully")
            
            # Query Pinecone for similar topics
            results = index.query(
                vector=query_embedding,
                top_k=request.top_k,
                include_metadata=True
            )
            print(f"Pinecone query returned {len(results.matches)} matches")
            
            # Format the results
            similar_topics = []
            for match in results.matches:
                score = float(match.score)
                print(f"Match: {match.metadata['text']} - Score: {score}")
                similar_topics.append(SimilarityResult(
                    topic=match.metadata['text'],
                    category=match.metadata['category'],
                    score=score
                ))
                
            return similar_topics
            
        else:  # Text-based similarity using thefuzz
            print("Using text-based similarity...")
            print(f"Input topic: {request.topic}")
            scored_topics = []
            
            for topic in topics:
                topic_text = topic['topic'].lower()
                input_text = request.topic.lower()
                
                # Use token set ratio for best matching with word order independence
                score = fuzz.token_set_ratio(input_text, topic_text)
                
                print(f"Comparing with: {topic_text}")
                print(f"Score: {score}")
                
                # Add to results if score is above 0
                scored_topics.append(SimilarityResult(
                    topic=topic['topic'],
                    category=topic['category'],
                    score=float(score) / 100  # Convert to 0-1 range
                ))
            
            # Sort by score and return top_k
            return scored_topics[:request.top_k]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics")
async def get_topics():
    """Get all topics from the index."""
    try:
        # Get all vectors (limited to 1000 for demo purposes)
        dummy_embedding = [0] * 384  # Match the dimension of all-MiniLM-L6-v2
        results = index.query(
            vector=dummy_embedding,
            top_k=1000,
            include_metadata=True
        )
        
        topics_list = []
        for match in results.matches:
            topics_list.append({
                'topic': match.metadata['text'],
                'category': match.metadata['category']
            })
            
        return topics_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
