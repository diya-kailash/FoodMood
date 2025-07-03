from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from src.food_recommender import FoodRecommender
import os

app = FastAPI(
    title="Food Recommender API",
    description="AI-powered food recommendation service with constraint analysis",
    version="1.0.0"
)

recommender = None
@app.on_event("startup")
async def startup_event():
    global recommender
    try:
        recommender = FoodRecommender()
        print("✅ Food Recommender API started successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize Food Recommender: {e}")
        raise

class RecommendationRequest(BaseModel):
    user_input: str = Field(..., description="User's food preference input")
    top_k: int = Field(default=8, ge=1, le=20, description="Number of recommendations to return")
    use_cross_encoder: bool = Field(default=True, description="Whether to use cross-encoder for reranking")

class FoodRecommendation(BaseModel):
    food_name: str
    cuisine: str
    food_type: str
    taste_notes: str
    texture: str
    prep_time: str
    budget: str
    social_context: str
    user_mood: str
    similarity_score: float
    tradeoff_score: Optional[float] = None
    combined_score: float
    tradeoff_explanation: str
    detailed_tradeoffs: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    recommendations: List[FoodRecommendation]
    total_found: int
    processing_time: Optional[str] = None

class ExplanationRequest(BaseModel):
    user_input: str = Field(..., description="Original user input")
    food_name: str = Field(..., description="Name of the food to explain")

class ExplanationResponse(BaseModel):
    explanation: str
    food_name: str
    user_input: str

class DatasetStatsResponse(BaseModel):
    total_items: int
    cuisines: int
    food_types: int
    unique_cuisines: List[str]
    unique_food_types: List[str]


@app.get("/", summary="Health Check")
async def root():
    return {
        "message": "Food Recommender API is running!",
        "status": "healthy",
        "groq_available": recommender.groq_client is not None if recommender else False
    }

@app.post("/recommendations", response_model=RecommendationResponse, summary="Get Food Recommendations")
async def get_recommendations(request: RecommendationRequest):
    if not recommender:
        raise HTTPException(status_code=500, detail="Food recommender not initialized")
    
    try:
        recommendations = recommender.get_recommendations(
            user_input=request.user_input,
            top_k=request.top_k,
            use_cross_encoder=request.use_cross_encoder
        )
        
        # Convert to Pydantic models
        food_recommendations = [
            FoodRecommendation(**rec) for rec in recommendations
        ]
        
        return RecommendationResponse(
            recommendations=food_recommendations,
            total_found=len(food_recommendations)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/recommendations/intelligent", response_model=RecommendationResponse, summary="Get Intelligent Recommendations")
async def get_intelligent_recommendations(request: RecommendationRequest):
    if not recommender:
        raise HTTPException(status_code=500, detail="Food recommender not initialized")
    
    try:
        recommendations = recommender.get_intelligent_recommendations(
            user_input=request.user_input,
            top_k=request.top_k
        )
        
        food_recommendations = [
            FoodRecommendation(**rec) for rec in recommendations
        ]
        
        return RecommendationResponse(
            recommendations=food_recommendations,
            total_found=len(food_recommendations)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting intelligent recommendations: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse, summary="Explain Recommendation")
async def explain_recommendation(request: ExplanationRequest):
    if not recommender:
        raise HTTPException(status_code=500, detail="Food recommender not initialized")
    
    try:
        explanation = recommender.explain_recommendation_with_groq(
            user_input=request.user_input,
            food_name=request.food_name
        )
        
        return ExplanationResponse(
            explanation=explanation,
            food_name=request.food_name,
            user_input=request.user_input
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining recommendation: {str(e)}")

@app.get("/dataset/stats", response_model=DatasetStatsResponse, summary="Get Dataset Statistics")
async def get_dataset_stats():
    if not recommender:
        raise HTTPException(status_code=500, detail="Food recommender not initialized")
    
    try:
        stats = recommender.get_dataset_stats()
        return DatasetStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset stats: {str(e)}")

@app.get("/search", response_model=RecommendationResponse, summary="Quick Search")
async def quick_search(
    query: str = Query(..., description="Search query for food recommendations"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of results to return")
):
    if not recommender:
        raise HTTPException(status_code=500, detail="Food recommender not initialized")
    
    try:
        recommendations = recommender.get_recommendations(
            user_input=query,
            top_k=limit,
            use_cross_encoder=True
        )
        
        food_recommendations = [
            FoodRecommendation(**rec) for rec in recommendations
        ]
        
        return RecommendationResponse(
            recommendations=food_recommendations,
            total_found=len(food_recommendations)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in quick search: {str(e)}")

@app.get("/health", summary="Detailed Health Check")
async def health_check():
    if not recommender:
        return {
            "status": "unhealthy",
            "recommender_initialized": False,
            "error": "Food recommender not initialized"
        }
    
    try:
        stats = recommender.get_dataset_stats()
        return {
            "status": "healthy",
            "recommender_initialized": True,
            "groq_available": recommender.groq_client is not None,
            "dataset_loaded": len(stats) > 0,
            "total_food_items": stats.get("total_items", 0),
            "model_loaded": recommender.model is not None,
            "embeddings_created": recommender.food_embeddings is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "recommender_initialized": True,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
