import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from typing import List, Dict, Any, Tuple
import warnings
import json
from dataclasses import dataclass, asdict
import re
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial, lru_cache
import hashlib
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

warnings.filterwarnings('ignore')

def performance_monitor(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⚡ {func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@dataclass
class UserConstraints:
    time_urgency: float = 0.0 
    budget_constraint: float = 0.0  
    health_consciousness: float = 0.0
    comfort_seeking: float = 0.0
    mood_state: str = ""
    social_requirements: List[str] = None
    cuisine_preferences: List[str] = None
    texture_preferences: List[str] = None
    taste_preferences: List[str] = None
    food_type_preferences: List[str] = None

    def __post_init__(self):
        if self.social_requirements is None:
            self.social_requirements = []
        if self.cuisine_preferences is None:
            self.cuisine_preferences = []
        if self.texture_preferences is None:
            self.texture_preferences = []
        if self.taste_preferences is None:
            self.taste_preferences = []
        if self.food_type_preferences is None:
            self.food_type_preferences = []

class FoodRecommender:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), 'Data', 'Dataset.csv')
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.cross_encoder = None
        self.food_embeddings = None
        self.combined_features = None
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.groq_client = None
        self.max_workers = min(8, os.cpu_count() or 4)  
        self._recommendation_cache = {}  
        self._cache_max_size = 100  
        
        if self.groq_api_key and self.groq_api_key != 'your-groq-api-key-here':
            self._setup_groq()
        else:
            print("⚠️  Warning: Groq API key not found in .env file.")
            print("   Please add your API key to .env file for AI-powered recommendations.")
        
        self._load_dataset()
        self._prepare_features()
        self._initialize_models()
        self._create_food_embeddings()
    
    def _setup_groq(self):
        if not GROQ_AVAILABLE:
            print("Warning: groq not available. Enhanced verification will be disabled.")
            self.groq_client = None
            return   
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("Groq AI client configured successfully")
        except Exception as e:
            print(f"Warning: Failed to setup Groq: {str(e)}. Enhanced verification will be disabled.")
            self.groq_client = None
    
    def _load_dataset(self):
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.df = self._preprocess_dataset()
            print(f"Dataset loaded successfully with {len(self.df)} food items")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _preprocess_dataset(self):
        df = self.df.copy()
        prep_time_mapping = {
            'Quick': 1.0, 'Moderate': 2.0, 'Elaborate': 3.0
        }
        budget_mapping = {
            'Cheap': 1.0, 'Cheap–Moderate': 1.5, 'Moderate': 2.0,
            'Moderate‑Expensive': 2.5, 'Expensive': 3.0
        }
        df['Prep Time Numeric'] = df['Prep Time'].map(prep_time_mapping).fillna(2.0)
        df['Budget Numeric'] = df['Budget'].map(budget_mapping).fillna(2.0)
        return df
    
    def _prepare_features(self):
        feature_columns = [
            'Food Item', 'Food Type', 'Cuisine', 'Taste Notes', 
            'Texture', 'Social Context', 'User Mood',
            'Prep Time', 'Budget' 
        ]
        self.combined_features = []
        for _, row in self.df.iterrows():
            features = []
            for col in feature_columns:
                if pd.notna(row.get(col, None)):
                    text = str(row[col]).strip().replace(',', ' ')
                    features.append(text)
            combined_text = ' '.join(features)
            self.combined_features.append(combined_text)
        print(f"Features prepared for {len(self.combined_features)} food items")
    
    def _initialize_models(self):
        print("Loading models...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Models loaded successfully")
    
    def _create_food_embeddings(self):
        print("Creating embeddings for food items using parallel processing...")
        batch_size = max(1, len(self.combined_features) // self.max_workers)
        feature_batches = [
            self.combined_features[i:i + batch_size] 
            for i in range(0, len(self.combined_features), batch_size)
        ]
        def encode_batch(batch):
            return self.model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        
        all_embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(encode_batch, batch): i 
                for i, batch in enumerate(feature_batches)
            }
            batch_results = [None] * len(feature_batches)
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    batch_results[batch_idx] = result
                    print(f"✓ Encoded batch {batch_idx + 1}/{len(feature_batches)}")
                except Exception as e:
                    print(f"✗ Error encoding batch {batch_idx}: {e}")
                    batch_results[batch_idx] = self.model.encode(
                        feature_batches[batch_idx], 
                        convert_to_tensor=False, 
                        show_progress_bar=False
                    )
        for batch_result in batch_results:
            if batch_result is not None:
                if len(all_embeddings) == 0:
                    all_embeddings = batch_result
                else:
                    all_embeddings = np.vstack([all_embeddings, batch_result])
        self.food_embeddings = all_embeddings
        print(f"Embeddings created with shape: {np.array(self.food_embeddings).shape}")
    
    def _extract_constraints_with_groq(self, user_input: str) -> UserConstraints: 
        if not self.groq_client:
            return UserConstraints()  
        constraint_extraction_prompt = f"""
        You are an expert food preference analyzer. Analyze the following user input for food preferences and extract structured information.

        User Input: "{user_input}"

        Extract the following information and provide scores from 0.0 to 1.0 where applicable:

        1. Time Urgency (0.0 = no rush, 1.0 = very urgent): Look for words like "quick", "fast", "hurry", "rush", "instant"
        2. Budget Consciousness (0.0 = money no object, 1.0 = very budget conscious): Look for words like "cheap", "affordable", "tight budget"
        3. Health Consciousness (0.0 = not important, 1.0 = very health focused): Look for words like "healthy", "light", "nutritious", "diet"
        4. Comfort Seeking (0.0 = adventurous, 1.0 = seeking comfort): Look for words like "comfort", "cozy", "familiar", "stressed", "tired"
        5. Mood State: Describe the user's mood in 1-2 words based on the input
        6. Social Context: List any mentioned contexts like "alone", "family", "friends", "party", "date", "group"
        7. Cuisine Preferences: List any specific cuisines mentioned
        8. Texture Preferences: List any texture preferences mentioned like "crispy", "soft", "crunchy", "fried", "roasted", "mushy", "creamy"
        9. Taste Preferences: List any taste preferences mentioned like "spicy", "sweet", "mild", "tangy", "savoury", "umami"
        10. Food Type Preferences: List any food type preferences mentioned like "snack", "meal", "light meal", "dessert", "beverage", "main course"

        Respond ONLY with a valid JSON object in this exact format:
        {{
            "time_urgency": 0.0,
            "budget_constraint": 0.0,
            "health_consciousness": 0.0,
            "comfort_seeking": 0.0,
            "mood_state": "",
            "social_requirements": [],
            "cuisine_preferences": [],
            "texture_preferences": [],
            "taste_preferences": [],
            "food_type_preferences": []
        }}

        Make sure to provide a valid JSON response with no additional text or explanation.
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": constraint_extraction_prompt,
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.1,
                max_tokens=1024,
            )
            response_text = chat_completion.choices[0].message.content.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                constraints_dict = json.loads(json_match.group())
                return UserConstraints(**constraints_dict)
            else:
                constraints_dict = json.loads(response_text)
                return UserConstraints(**constraints_dict)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Groq response: {response_text}")
            return UserConstraints()
        except Exception as e:
            print(f"Groq API error in constraint extraction: {e}")
            return UserConstraints()
    
    def _analyze_single_candidate(self, args):
        idx, constraints, similarity_score = args
        try:
            food_item = self.df.iloc[idx]
            tradeoff_analysis = self._analyze_tradeoffs_with_groq(constraints, food_item)
            
            return {
                'index': idx,
                'similarity_score': similarity_score,
                'tradeoff_analysis': tradeoff_analysis,
                'combined_score': (similarity_score * 0.4 + tradeoff_analysis.get('overall_score', 0) * 0.6)
            }
        except Exception as e:
            print(f"Error analyzing candidate {idx}: {e}")
            return {
                'index': idx,
                'similarity_score': similarity_score,
                'tradeoff_analysis': {"overall_score": 0.5, "overall_explanation": "Error in analysis"},
                'combined_score': similarity_score * 0.4 + 0.5 * 0.6
            }
    
    def _analyze_tradeoffs_with_groq(self, constraints: UserConstraints, food_item: pd.Series) -> Dict[str, Any]:
        if not self.groq_client:
            return {
                "overall_score": 0.5,
                "time_tradeoff": {"score": 0.5, "explanation": "Moderate match"},
                "budget_tradeoff": {"score": 0.5, "explanation": "Reasonable option"},
                "health_tradeoff": {"score": 0.5, "explanation": "Balanced choice"},
                "comfort_tradeoff": {"score": 0.5, "explanation": "Satisfying option"},
                "social_tradeoff": {"score": 0.5, "explanation": "Suitable choice"},
                "cuisine_tradeoff": {"score": 0.5, "explanation": "Good cuisine match"},
                "taste_tradeoff": {"score": 0.5, "explanation": "Decent taste match"},
                "texture_tradeoff": {"score": 0.5, "explanation": "Suitable texture"},
                "mood_tradeoff": {"score": 0.5, "explanation": "Mood appropriate"},
                "food_type_tradeoff": {"score": 0.5, "explanation": "Type suitable"},
                "overall_explanation": "Balanced recommendation"
            }
        
        food_description = {
            'name': food_item.get('Food Item', ''),
            'type': food_item.get('Food Type', ''),
            'cuisine': food_item.get('Cuisine', ''),
            'taste': food_item.get('Taste Notes', ''),
            'texture': food_item.get('Texture', ''),
            'prep_time': food_item.get('Prep Time', ''),
            'budget': food_item.get('Budget', ''),
            'social_context': food_item.get('Social Context', ''),
            'mood': food_item.get('User Mood', '')
        }

        tradeoff_analysis_prompt = f"""
        Analyze food-user match. Score each tradeoff 0.0-1.0 (1.0=perfect).

        User: {json.dumps(asdict(constraints))}
        Food: {json.dumps(food_description)}

        Score these matches:
        1. Time: prep time vs urgency
        2. Budget: cost vs budget consciousness  
        3. Health: healthiness vs health focus
        4. Comfort: comfort level vs comfort seeking
        5. Social: appropriateness for social context
        6. Cuisine: cuisine match vs preferences
        7. Taste: taste match vs preferences
        8. Texture: texture match vs preferences
        9. Mood: mood appropriateness
        10. Type: food type vs preferences

        JSON only:
        {{
            "overall_score": 0.0,
            "time_tradeoff": {{"score": 0.0, "explanation": ""}},
            "budget_tradeoff": {{"score": 0.0, "explanation": ""}},
            "health_tradeoff": {{"score": 0.0, "explanation": ""}},
            "comfort_tradeoff": {{"score": 0.0, "explanation": ""}},
            "social_tradeoff": {{"score": 0.0, "explanation": ""}},
            "cuisine_tradeoff": {{"score": 0.0, "explanation": ""}},
            "taste_tradeoff": {{"score": 0.0, "explanation": ""}},
            "texture_tradeoff": {{"score": 0.0, "explanation": ""}},
            "mood_tradeoff": {{"score": 0.0, "explanation": ""}},
            "food_type_tradeoff": {{"score": 0.0, "explanation": ""}},
            "overall_explanation": ""
        }}
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": tradeoff_analysis_prompt,
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.1,
                max_tokens=1024,
            )
            response_text = chat_completion.choices[0].message.content.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in tradeoff analysis: {e}")
            print(f"Groq response: {response_text}")
            return {
                "overall_score": 0.5,
                "time_tradeoff": {"score": 0.5, "explanation": "Moderate match"},
                "budget_tradeoff": {"score": 0.5, "explanation": "Reasonable option"},
                "health_tradeoff": {"score": 0.5, "explanation": "Balanced choice"},
                "comfort_tradeoff": {"score": 0.5, "explanation": "Satisfying option"},
                "social_tradeoff": {"score": 0.5, "explanation": "Suitable choice"},
                "cuisine_tradeoff": {"score": 0.5, "explanation": "Good cuisine match"},
                "taste_tradeoff": {"score": 0.5, "explanation": "Decent taste match"},
                "texture_tradeoff": {"score": 0.5, "explanation": "Suitable texture"},
                "mood_tradeoff": {"score": 0.5, "explanation": "Mood appropriate"},
                "food_type_tradeoff": {"score": 0.5, "explanation": "Type suitable"},
                "overall_explanation": "Balanced recommendation"
            }
        except Exception as e:
            print(f"Groq API error in tradeoff analysis: {e}")
            return {
                "overall_score": 0.5,
                "time_tradeoff": {"score": 0.5, "explanation": "Moderate match"},
                "budget_tradeoff": {"score": 0.5, "explanation": "Reasonable option"},
                "health_tradeoff": {"score": 0.5, "explanation": "Balanced choice"},
                "comfort_tradeoff": {"score": 0.5, "explanation": "Satisfying option"},
                "social_tradeoff": {"score": 0.5, "explanation": "Suitable choice"},
                "cuisine_tradeoff": {"score": 0.5, "explanation": "Good cuisine match"},
                "taste_tradeoff": {"score": 0.5, "explanation": "Decent taste match"},
                "texture_tradeoff": {"score": 0.5, "explanation": "Suitable texture"},
                "mood_tradeoff": {"score": 0.5, "explanation": "Mood appropriate"},
                "food_type_tradeoff": {"score": 0.5, "explanation": "Type suitable"},
                "overall_explanation": "Balanced recommendation"
            }
    
    def _get_cache_key(self, user_input: str, top_k: int) -> str:
        content = f"{user_input.lower().strip()}_{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_recommendations(self, cache_key: str, recommendations: List[Dict[str, Any]]):
        if len(self._recommendation_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._recommendation_cache))
            del self._recommendation_cache[oldest_key]
        
        self._recommendation_cache[cache_key] = {
            'recommendations': recommendations,
            'timestamp': time.time()
        }
    
    @performance_monitor
    def get_recommendations(self, user_input: str, top_k: int = 8, use_cross_encoder: bool = True) -> List[Dict[str, Any]]:
        cache_key = self._get_cache_key(user_input, top_k)
        if cache_key in self._recommendation_cache:
            cached_data = self._recommendation_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 300:
                print("🚀 Retrieved from cache (instant)")
                return cached_data['recommendations']
            else:
                del self._recommendation_cache[cache_key]
        if self.groq_client:
            try:
                recommendations = self.get_intelligent_recommendations(user_input, top_k)
            except Exception as e:
                print(f"⚠️  Groq analysis failed, falling back to similarity matching: {str(e)}")
                recommendations = self._get_similarity_recommendations(user_input, top_k, use_cross_encoder)
        else:
            recommendations = self._get_similarity_recommendations(user_input, top_k, use_cross_encoder)
        self._cache_recommendations(cache_key, recommendations)
        
        return recommendations
    
    def _get_similarity_recommendations(self, user_input: str, top_k: int = 8, use_cross_encoder: bool = True) -> List[Dict[str, Any]]:
        try:
            user_embedding = self.model.encode([user_input], convert_to_tensor=False)
            similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]
            top_indices = np.argsort(similarity_scores)[::-1][:top_k * 3]  
            if use_cross_encoder and self.cross_encoder:
                cross_inputs = [[user_input, self.combined_features[idx]] for idx in top_indices]
                batch_size = max(1, len(cross_inputs) // self.max_workers)
                input_batches = [
                    cross_inputs[i:i + batch_size] 
                    for i in range(0, len(cross_inputs), batch_size)
                ]
                
                def process_cross_batch(batch):
                    return self.cross_encoder.predict(batch)
                all_scores = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(process_cross_batch, batch): i 
                        for i, batch in enumerate(input_batches)
                    }
                    batch_results = [None] * len(input_batches)
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_results[batch_idx] = future.result()
                        except Exception as e:
                            print(f"Cross-encoder batch {batch_idx} failed: {e}")
                            batch_size = len(input_batches[batch_idx])
                            start_idx = batch_idx * len(input_batches[0]) if batch_idx < len(input_batches) - 1 else batch_idx * batch_size
                            batch_results[batch_idx] = similarity_scores[top_indices[start_idx:start_idx + batch_size]]
                for batch_result in batch_results:
                    if batch_result is not None:
                        if len(all_scores) == 0:
                            all_scores = batch_result
                        else:
                            all_scores = np.concatenate([all_scores, batch_result])
                cross_scores = all_scores
                reranked = sorted(zip(top_indices, cross_scores), key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in reranked[:top_k]]
            else:
                top_indices = top_indices[:top_k]
            
            def build_recommendation(idx):
                food_item = self.df.iloc[idx]
                score = similarity_scores[idx]
                return {
                    'food_name': food_item.get('Food Item', ''),
                    'cuisine': food_item.get('Cuisine', ''),
                    'food_type': food_item.get('Food Type', ''),
                    'taste_notes': food_item.get('Taste Notes', ''),
                    'texture': food_item.get('Texture', ''),
                    'prep_time': food_item.get('Prep Time', ''),
                    'budget': food_item.get('Budget', ''),
                    'social_context': food_item.get('Social Context', ''),
                    'user_mood': food_item.get('User Mood', ''),
                    'similarity_score': round(score, 3),
                    'tradeoff_score': round(score, 3),
                    'combined_score': round(score, 3),
                    'tradeoff_explanation': 'Based on similarity matching (fast mode)',
                    'detailed_tradeoffs': {
                        'overall_score': score,
                        'overall_explanation': 'Based on similarity matching'
                    }
                }
            
            recommendations = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(build_recommendation, idx): idx 
                    for idx in top_indices
                }
                for future in as_completed(future_to_idx):
                    try:
                        recommendation = future.result()
                        recommendations.append(recommendation)
                    except Exception as e:
                        idx = future_to_idx[future]
                        print(f"Error building recommendation for {idx}: {e}")
            recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
            return recommendations
        
        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")
    
    @performance_monitor
    def get_intelligent_recommendations(self, user_input: str, top_k: int = 8) -> List[Dict[str, Any]]:
        try:
            if self.groq_client:
                print("Extracting constraints with Groq...")
                constraints = self._extract_constraints_with_groq(user_input)
                print(f"Extracted constraints: {asdict(constraints)}")
            else:
                constraints = UserConstraints()
            user_embedding = self.model.encode([user_input], convert_to_tensor=False)
            similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]
            top_candidates = np.argsort(similarity_scores)[::-1][:top_k * 2]
            if self.groq_client:
                print(f"Analyzing tradeoffs for top {len(top_candidates)} candidates using parallel processing...")
                analysis_args = [
                    (idx, constraints, similarity_scores[idx]) 
                    for idx in top_candidates
                ]
                candidate_analyses = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._analyze_single_candidate, args): args[0] 
                        for args in analysis_args
                    }
                    for i, future in enumerate(as_completed(future_to_idx)):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            candidate_analyses.append(result)
                            print(f"✓ Completed {i+1}/{len(top_candidates)}: {self.df.iloc[idx]['Food Item']}")
                        except Exception as e:
                            print(f"✗ Error with candidate {idx}: {e}")
                            candidate_analyses.append({
                                'index': idx,
                                'similarity_score': similarity_scores[idx],
                                'tradeoff_analysis': {"overall_score": 0.5, "overall_explanation": "Analysis failed"},
                                'combined_score': similarity_scores[idx] * 0.4 + 0.5 * 0.6
                            })
                candidate_analyses.sort(key=lambda x: x['combined_score'], reverse=True)
            else:
                candidate_analyses = []
                for idx in top_candidates[:top_k]:
                    candidate_analyses.append({
                        'index': idx,
                        'similarity_score': similarity_scores[idx],
                        'tradeoff_analysis': {
                            "overall_score": similarity_scores[idx],
                            "overall_explanation": "Based on similarity matching"
                        },
                        'combined_score': similarity_scores[idx]
                    })
            recommendations = []
            for analysis in candidate_analyses[:top_k]:
                idx = analysis['index']
                food_item = self.df.iloc[idx]
                recommendation = {
                    'food_name': food_item.get('Food Item', ''),
                    'cuisine': food_item.get('Cuisine', ''),
                    'food_type': food_item.get('Food Type', ''),
                    'taste_notes': food_item.get('Taste Notes', ''),
                    'texture': food_item.get('Texture', ''),
                    'prep_time': food_item.get('Prep Time', ''),
                    'budget': food_item.get('Budget', ''),
                    'social_context': food_item.get('Social Context', ''),
                    'user_mood': food_item.get('User Mood', ''),
                    'similarity_score': round(analysis['similarity_score'], 3),
                    'tradeoff_score': round(analysis['tradeoff_analysis'].get('overall_score', 0), 3),
                    'combined_score': round(analysis['combined_score'], 3),
                    'tradeoff_explanation': analysis['tradeoff_analysis'].get('overall_explanation', ''),
                    'detailed_tradeoffs': analysis['tradeoff_analysis']
                }
                recommendations.append(recommendation)
            return recommendations

        except Exception as e:
            raise Exception(f"Error getting intelligent recommendations: {str(e)}")
    
    def explain_recommendation_with_groq(self, user_input: str, food_name: str) -> str:        
        if not self.groq_client:
            return f"Enhanced explanations require Groq API key. {food_name} was recommended based on similarity matching."
        food_item = self.df[self.df['Food Item'] == food_name].iloc[0] if len(self.df[self.df['Food Item'] == food_name]) > 0 else None
        if food_item is None:
            return f"Food item '{food_name}' not found in dataset."
        explanation_prompt = f"""
        Explain why "{food_name}" is a good recommendation for this user request in a conversational, friendly way.

        User Request: "{user_input}"

        Food Details:
        - Name: {food_item.get('Food Item', '')}
        - Type: {food_item.get('Food Type', '')}
        - Cuisine: {food_item.get('Cuisine', '')}
        - Taste: {food_item.get('Taste Notes', '')}
        - Texture: {food_item.get('Texture', '')}
        - Prep Time: {food_item.get('Prep Time', '')}
        - Budget: {food_item.get('Budget', '')}
        - Social Context: {food_item.get('Social Context', '')}
        - Mood: {food_item.get('User Mood', '')}

        Provide a friendly, conversational explanation (2-3 sentences) about why this food matches the user's request, mentioning specific aspects that align with their needs.
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": explanation_prompt,
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=512,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        stats = {
            'total_items': len(self.df),
            'cuisines': self.df['Cuisine'].nunique(),
            'food_types': self.df['Food Type'].nunique(),
            'unique_cuisines': sorted(self.df['Cuisine'].dropna().unique().tolist()),
            'unique_food_types': sorted(self.df['Food Type'].dropna().unique().tolist())
        }
        return stats


if __name__ == "__main__":
    try:
        recommender = FoodRecommender()
        test_input = "I want something quick to make, comforting, warm and under a tight budget"
        print("=== AI-POWERED RECOMMENDATIONS ===")
        recommendations = recommender.get_recommendations(test_input, top_k=5)
        print(f"\nRecommendations for: '{test_input}'")
        print("-" * 60)      
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['food_name']} ({rec['cuisine']})")
            print(f"   Type: {rec['food_type']}")
            print(f"   Taste: {rec['taste_notes']}")
            print(f"   Prep Time: {rec['prep_time']}, Budget: {rec['budget']}")
            if 'combined_score' in rec:
                print(f"   Scores - Similarity: {rec['similarity_score']}, AI: {rec['tradeoff_score']}, Combined: {rec['combined_score']}")
                print(f"   AI Explanation: {rec['tradeoff_explanation']}")
            else:
                print(f"   Similarity Score: {rec['similarity_score']}")
            print()  

        print("\n=== INTELLIGENT RECOMMENDATIONS (No Groq) ===")
        intelligent_recs = recommender.get_intelligent_recommendations(test_input, top_k=5)
        for i, rec in enumerate(intelligent_recs, 1):
            print(f"{i}. {rec['food_name']} ({rec['cuisine']})")
            print(f"   Type: {rec['food_type']} | Prep: {rec['prep_time']} | Budget: {rec['budget']}")
            print(f"   Scores - Similarity: {rec['similarity_score']}, Combined: {rec['combined_score']}")
            print(f"   Explanation: {rec['tradeoff_explanation']}")
            print()
        print("\n=== GROQ INTEGRATION ===")
        if recommender.groq_client:
            print("✅ Groq AI is enabled and working!")
            print("🧠 All recommendations are AI-powered with constraint analysis")
        else:
            print("⚠️  Groq AI not available - using similarity matching")
            print("To enable enhanced analysis with Groq AI:")
            print("1. Get a Groq API key from https://console.groq.com/keys")
            print("2. Add GROQ_API_KEY=your-api-key to .env file")
            print("3. Restart the application")
    except Exception as e:
        print(f"Error testing recommender: {str(e)}")
