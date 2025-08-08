
import random
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from thefuzz import fuzz
from pydantic import BaseModel
from typing import List


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://project-detect-lovat.vercel.app"],  # No trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Artificial project topics
topics = [
    "Design and implementation of a smart irrigation system",
    "Development of a mobile health monitoring app",
    "Automated attendance system using facial recognition",
    "IoT-based home security system",
    "Real-time traffic prediction using machine learning",
    "Blockchain-based voting system",
    "AI-powered chatbot for customer service",
    "Energy consumption optimization in smart grids",
    "Sentiment analysis of social media posts",
    "E-commerce product recommendation engine",
    "Voice-controlled personal assistant",
    "Disease prediction using medical imaging",
    "Automated plagiarism detection for academic papers",
    "Smart waste management system",
    "Online exam proctoring using AI",
    "Weather forecasting using deep learning",
    "Cybersecurity threat detection platform",
    "Virtual reality for education",
    "Remote patient monitoring system",
    "Intelligent parking management solution",
    "crowd sourced reporting platform",
    "Web-based voice-email system for visually impaired",
    "Emergency Response System",
    "Online Department Payment System (PCI DSS Complaint)",
    "Installment Payment System (A Case Study of Hedge Leasing Limited)",
    "Biometric Attendance System",
    "Attendance Management System Using Biometrics",
    "Development of Online Based Tutorial for Computer Programming Language for QBasic",
    "Web Based SIWES System (A Case Study of Computer Science Department, Moshood Abiola Polytechnic, Ogun State)",
    "Automated Project Topic Registration System",
    "Mapoly Staff Location System Using Geographical Information System (GIS)",
    "Online Department Payment System (PCI DSS Complaint)",

]

class TopicCheckRequest(BaseModel):
    topic: str

class SimilarityResult(BaseModel):
    topic: str
    score: int

@app.post("/check_similarity", response_model=List[SimilarityResult])
def check_similarity(request: TopicCheckRequest):
    input_topic = request.topic
    results = []
    for t in topics:
        score = fuzz.token_set_ratio(input_topic, t)
        results.append(SimilarityResult(topic=t, score=score))
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    return results

@app.get("/topics")
def get_topics():
    return topics
