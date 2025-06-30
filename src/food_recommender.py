import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

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
        
        # Load and prepare the dataset
        self._load_dataset()
        self._prepare_features()
        
        # Initialize models
        self._initialize_models()
        
        # Create embeddings for all food items
        self._create_food_embeddings()
    
    def _load_dataset(self):
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully with {len(self.df)} food items")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
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
    
    def get_recommendations(self, user_input: str, top_k: int = 8, use_cross_encoder: bool = True) -> List[Dict[str, Any]]:
        try:
            user_embedding = self.model.encode([user_input], convert_to_tensor=False)
            similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]
            
            top_indices = np.argsort(similarity_scores)[::-1][:top_k * 3]  # Retrieve more for reranking
            
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
                    'similarity_score': round(score, 3)
                }
                recommendations.append(recommendation)
            
            return recommendations
        
        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")
    
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
        recommendations = recommender.get_recommendations(test_input, top_k=5)
        
        print(f"\nRecommendations for: '{test_input}'")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['food_name']} ({rec['cuisine']})")
            print(f"   Type: {rec['food_type']}")
            print(f"   Taste: {rec['taste_notes']}")
            print(f"   Prep Time: {rec['prep_time']}, Budget: {rec['budget']}")
            print(f"   Similarity Score: {rec['similarity_score']}")
            print()
    
    except Exception as e:
        print(f"Error testing recommender: {str(e)}")
