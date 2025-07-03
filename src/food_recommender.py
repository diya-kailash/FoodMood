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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Optional Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

warnings.filterwarnings('ignore')

@dataclass
class UserConstraints:
    """Structured representation of user constraints extracted by AI"""
    time_urgency: float = 0.0  # 0-1 scale
    budget_constraint: float = 0.0  # 0-1 scale (1 = very budget conscious)
    health_consciousness: float = 0.0
    comfort_seeking: float = 0.0
    mood_state: str = ""
    social_requirements: List[str] = None
    cuisine_preferences: List[str] = None
    texture_preferences: List[str] = None
    taste_preferences: List[str] = None

    def __post_init__(self):
        if self.social_requirements is None:
            self.social_requirements = []
        if self.cuisine_preferences is None:
            self.cuisine_preferences = []
        if self.texture_preferences is None:
            self.texture_preferences = []
        if self.taste_preferences is None:
            self.taste_preferences = []

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
        
        # Groq components - load API key from environment
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.groq_client = None
        
        # Always try to setup Groq if API key is available
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
        """Setup Groq AI client"""
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
        """Preprocess dataset with numerical mappings"""
        df = self.df.copy()
        
        # Create numerical mappings
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
        print("Creating embeddings for food items...")
        self.food_embeddings = self.model.encode(
            self.combined_features,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        print(f"Embeddings created with shape: {np.array(self.food_embeddings).shape}")
    
    def _extract_constraints_with_groq(self, user_input: str) -> UserConstraints:
        """Extract user constraints using Groq AI"""
        
        if not self.groq_client:
            return UserConstraints()  # Return default constraints if Groq not available
        
        constraint_extraction_prompt = f"""
        You are an expert food preference analyzer. Analyze the following user input for food preferences and extract structured information.

        User Input: "{user_input}"

        Extract the following information and provide scores from 0.0 to 1.0 where applicable:

        1. Time Urgency (0.0 = no rush, 1.0 = very urgent): Look for words like "quick", "fast", "hurry", "rush", "instant"
        2. Budget Consciousness (0.0 = money no object, 1.0 = very budget conscious): Look for words like "cheap", "affordable", "tight budget"
        3. Health Consciousness (0.0 = not important, 1.0 = very health focused): Look for words like "healthy", "light", "nutritious", "diet"
        4. Comfort Seeking (0.0 = adventurous, 1.0 = seeking comfort): Look for words like "comfort", "cozy", "familiar", "stressed", "tired"
        5. Social Context: List any mentioned contexts like "alone", "family", "friends", "party", "date", "group"
        6. Mood State: Describe the user's mood in 1-2 words based on the input
        7. Cuisine Preferences: List any specific cuisines mentioned
        8. Texture Preferences: List any texture preferences mentioned like "crispy", "soft", "crunchy", "fried", "roasted", "mushy"
        9. Taste Preferences: List any taste preferences mentioned like "spicy", "sweet", "mild", "tangy", "savoury"

        Respond ONLY with a valid JSON object in this exact format:
        {{
            "time_urgency": 0.0,
            "budget_constraint": 0.0,
            "health_consciousness": 0.0,
            "comfort_seeking": 0.0,
            "social_requirements": [],
            "mood_state": "",
            "cuisine_preferences": [],
            "texture_preferences": [],
            "taste_preferences": []
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

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                constraints_dict = json.loads(json_match.group())
                return UserConstraints(**constraints_dict)
            else:
                # If no JSON found, try parsing the entire response
                constraints_dict = json.loads(response_text)
                return UserConstraints(**constraints_dict)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Groq response: {response_text}")
            # Return default constraints if parsing fails
            return UserConstraints()
        except Exception as e:
            print(f"Groq API error in constraint extraction: {e}")
            return UserConstraints()
    
    def _analyze_tradeoffs_with_groq(self, constraints: UserConstraints, food_item: pd.Series) -> Dict[str, Any]:
        """Analyze tradeoffs using Groq AI"""
        
        if not self.groq_client:
            # Return default analysis if Groq not available
            return {
                "overall_score": 0.5,
                "time_tradeoff": {"score": 0.5, "explanation": "Moderate match"},
                "budget_tradeoff": {"score": 0.5, "explanation": "Reasonable option"},
                "health_tradeoff": {"score": 0.5, "explanation": "Balanced choice"},
                "comfort_tradeoff": {"score": 0.5, "explanation": "Satisfying option"},
                "social_tradeoff": {"score": 0.5, "explanation": "Suitable choice"},
                "overall_explanation": "Balanced recommendation"
            }
        
        # Prepare food item description
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
        You are an expert food recommendation analyst. Analyze how well this food item satisfies the user's needs and what tradeoffs are involved.

        User Constraints: {json.dumps(asdict(constraints), indent=2)}

        Food Item Details: {json.dumps(food_description, indent=2)}

        Analyze the following tradeoffs and provide scores from 0.0 to 1.0:

        1. Time Tradeoff: How well does the prep time match the user's time urgency?
        2. Budget Tradeoff: How well does the cost match the user's budget constraints?
        3. Health Tradeoff: How well does this match the user's health consciousness?
        4. Comfort Tradeoff: How well does this match the user's comfort seeking needs?
        5. Social Tradeoff: How appropriate is this for the user's social context?

        For each tradeoff, provide:
        - A score from 0.0 to 1.0 (1.0 = perfect match, 0.0 = poor match)
        - A brief explanation (max 10 words)

        Calculate an overall score as the weighted average of relevant tradeoffs.

        Respond ONLY with a valid JSON object in this exact format:
        {{
            "overall_score": 0.0,
            "time_tradeoff": {{"score": 0.0, "explanation": ""}},
            "budget_tradeoff": {{"score": 0.0, "explanation": ""}},
            "health_tradeoff": {{"score": 0.0, "explanation": ""}},
            "comfort_tradeoff": {{"score": 0.0, "explanation": ""}},
            "social_tradeoff": {{"score": 0.0, "explanation": ""}},
            "overall_explanation": ""
        }}

        Make sure to provide a valid JSON response with no additional text.
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

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in tradeoff analysis: {e}")
            print(f"Groq response: {response_text}")
            # Return default analysis if parsing fails
            return {
                "overall_score": 0.5,
                "time_tradeoff": {"score": 0.5, "explanation": "Moderate match"},
                "budget_tradeoff": {"score": 0.5, "explanation": "Reasonable option"},
                "health_tradeoff": {"score": 0.5, "explanation": "Balanced choice"},
                "comfort_tradeoff": {"score": 0.5, "explanation": "Satisfying option"},
                "social_tradeoff": {"score": 0.5, "explanation": "Suitable choice"},
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
                "overall_explanation": "Balanced recommendation"
            }
    
    def get_recommendations(self, user_input: str, top_k: int = 8, use_cross_encoder: bool = True) -> List[Dict[str, Any]]:
        """Get AI-powered recommendations using Groq analysis (primary method)"""
        # Always try to use intelligent recommendations first
        if self.groq_client:
            try:
                return self.get_intelligent_recommendations(user_input, top_k)
            except Exception as e:
                print(f"⚠️  Groq analysis failed, falling back to similarity matching: {str(e)}")
        
        # Fallback to similarity-based recommendations
        return self._get_similarity_recommendations(user_input, top_k, use_cross_encoder)
    
    def _get_similarity_recommendations(self, user_input: str, top_k: int = 8, use_cross_encoder: bool = True) -> List[Dict[str, Any]]:
        """Fallback method for similarity-based recommendations"""
        try:
            user_embedding = self.model.encode([user_input], convert_to_tensor=False)
            similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]
            
            top_indices = np.argsort(similarity_scores)[::-1][:top_k * 3]  
            
            if use_cross_encoder and self.cross_encoder:
                cross_inputs = [[user_input, self.combined_features[idx]] for idx in top_indices]
                cross_scores = self.cross_encoder.predict(cross_inputs)
                reranked = sorted(zip(top_indices, cross_scores), key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in reranked[:top_k]]
            else:
                top_indices = top_indices[:top_k]
            
            recommendations = []
            for idx in top_indices:
                food_item = self.df.iloc[idx]
                score = similarity_scores[idx]
                
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
                    'similarity_score': round(score, 3),
                    'tradeoff_score': round(score, 3),  # Use similarity as fallback
                    'combined_score': round(score, 3),
                    'tradeoff_explanation': 'Based on similarity matching (Gemini not available)',
                    'detailed_tradeoffs': {
                        'overall_score': score,
                        'overall_explanation': 'Based on similarity matching'
                    }
                }
                recommendations.append(recommendation)
            
            return recommendations
        
        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")
    
    def get_intelligent_recommendations(self, user_input: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Get recommendations using Groq-based constraint extraction and tradeoff analysis"""

        try:
            # Extract constraints using Groq (if available)
            if self.groq_client:
                print("Extracting constraints with Groq...")
                constraints = self._extract_constraints_with_groq(user_input)
                print(f"Extracted constraints: {asdict(constraints)}")
            else:
                constraints = UserConstraints()

            # Get semantic similarity scores
            user_embedding = self.model.encode([user_input], convert_to_tensor=False)
            similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]

            # Get top candidates for detailed analysis
            top_candidates = np.argsort(similarity_scores)[::-1][:top_k * 2]

            if self.groq_client:
                print(f"Analyzing tradeoffs for top {len(top_candidates)} candidates...")
                # Analyze tradeoffs for each candidate using Groq
                candidate_analyses = []
                for i, idx in enumerate(top_candidates):
                    print(f"Analyzing candidate {i+1}/{len(top_candidates)}: {self.df.iloc[idx]['Food Item']}")
                    food_item = self.df.iloc[idx]
                    tradeoff_analysis = self._analyze_tradeoffs_with_groq(constraints, food_item)

                    candidate_analyses.append({
                        'index': idx,
                        'similarity_score': similarity_scores[idx],
                        'tradeoff_analysis': tradeoff_analysis,
                        'combined_score': (similarity_scores[idx] * 0.4 +
                                         tradeoff_analysis.get('overall_score', 0) * 0.6)
                    })

                # Sort by combined score
                candidate_analyses.sort(key=lambda x: x['combined_score'], reverse=True)
            else:
                # Fallback to similarity-only ranking
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

            # Build final recommendations
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
        """Get detailed explanation for a specific recommendation using Groq"""
        
        if not self.groq_client:
            return f"Enhanced explanations require Groq API key. {food_name} was recommended based on similarity matching."

        # Find the food item
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
        # Initialize recommender (without Groq for basic testing)
        recommender = FoodRecommender()
        
        test_input = "I want something quick to make, comforting, warm and under a tight budget"
        
        # Test basic recommendations (now AI-powered if API key available)
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
        
        # Test intelligent recommendations (works without Groq API but with enhanced features)
        print("\n=== INTELLIGENT RECOMMENDATIONS (No Groq) ===")
        intelligent_recs = recommender.get_intelligent_recommendations(test_input, top_k=5)
        
        for i, rec in enumerate(intelligent_recs, 1):
            print(f"{i}. {rec['food_name']} ({rec['cuisine']})")
            print(f"   Type: {rec['food_type']} | Prep: {rec['prep_time']} | Budget: {rec['budget']}")
            print(f"   Scores - Similarity: {rec['similarity_score']}, Combined: {rec['combined_score']}")
            print(f"   Explanation: {rec['tradeoff_explanation']}")
            print()
        
        # Instructions for Groq usage
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
