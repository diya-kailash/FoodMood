"""
Food Recommender System using Sentence Transformers
Recommends food based on user mood and preferences using semantic similarity
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers, use TF-IDF as fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Sentence Transformers not available, using TF-IDF as fallback")


class FoodRecommender:
    def __init__(self, dataset_path: str = None):
        """
        Initialize the Food Recommender System
        
        Args:
            dataset_path: Path to the CSV dataset file
        """
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), 'Data', 'Dataset.csv')
        
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.vectorizer = None
        self.food_embeddings = None
        self.combined_features = None
        self.use_sentence_transformers = SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Load and prepare the dataset
        self._load_dataset()
        self._prepare_features()
        
        # Initialize the model (either sentence transformers or TF-IDF)
        self._initialize_model()
        
        # Create embeddings for all food items
        self._create_food_embeddings()
    
    def _load_dataset(self):
        """Load the food dataset from CSV"""
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully with {len(self.df)} food items")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _prepare_features(self):
        """Prepare combined text features for each food item"""
        # Clean and combine relevant text features
        feature_columns = [
            'Food Item', 'Food Type', 'Cuisine', 'Taste Notes', 
            'Texture', 'Social Context', 'User Mood'
        ]
        
        # Create combined feature text for each food item
        self.combined_features = []
        
        for _, row in self.df.iterrows():
            features = []
            for col in feature_columns:
                if pd.notna(row[col]):
                    # Clean the text and add to features
                    text = str(row[col]).strip().replace(',', ' ')
                    features.append(text)
            
            # Combine all features into a single string
            combined_text = ' '.join(features)
            self.combined_features.append(combined_text)
        
        print(f"Features prepared for {len(self.combined_features)} food items")
    
    def _initialize_model(self):
        """Initialize the model (sentence transformers or TF-IDF fallback)"""
        if self.use_sentence_transformers:
            try:
                # Using a lightweight but effective model for food/text similarity
                model_name = 'all-MiniLM-L6-v2'
                print(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                print("Sentence transformer model loaded successfully")
                return
            except Exception as e:
                print(f"Error loading sentence transformer model: {str(e)}")
                print("Falling back to TF-IDF vectorizer...")
                self.use_sentence_transformers = False
        
        # Initialize TF-IDF vectorizer as fallback
        print("Using TF-IDF vectorizer for text similarity")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        print("TF-IDF vectorizer initialized successfully")
    
    def _create_food_embeddings(self):
        """Create embeddings for all food items"""
        try:
            print("Creating embeddings for food items...")
            
            if self.use_sentence_transformers and self.model:
                # Use sentence transformers
                self.food_embeddings = self.model.encode(
                    self.combined_features,
                    convert_to_tensor=False,
                    show_progress_bar=True
                )
                print(f"Sentence transformer embeddings created with shape: {self.food_embeddings.shape}")
            else:
                # Use TF-IDF vectorizer
                self.food_embeddings = self.vectorizer.fit_transform(self.combined_features)
                print(f"TF-IDF embeddings created with shape: {self.food_embeddings.shape}")
                
        except Exception as e:
            raise Exception(f"Error creating food embeddings: {str(e)}")
    
    def get_recommendations(self, user_input: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Get food recommendations based on user input
        
        Args:
            user_input: Natural language description of mood/preferences
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended food items with details
        """
        try:
            if self.use_sentence_transformers and self.model:
                # Create embedding for user input using sentence transformers
                user_embedding = self.model.encode([user_input], convert_to_tensor=False)
                # Calculate similarity scores
                similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]
            else:
                # Create embedding for user input using TF-IDF
                user_embedding = self.vectorizer.transform([user_input])
                # Calculate similarity scores
                similarity_scores = cosine_similarity(user_embedding, self.food_embeddings)[0]
            
            # Get top-k recommendations
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]
            
            recommendations = []
            for idx in top_indices:
                food_item = self.df.iloc[idx]
                score = similarity_scores[idx]
                
                recommendation = {
                    'food_name': food_item['Food Item'],
                    'cuisine': food_item['Cuisine'],
                    'food_type': food_item['Food Type'],
                    'taste_notes': food_item['Taste Notes'],
                    'texture': food_item['Texture'],
                    'prep_time': food_item['Prep Time'],
                    'budget': food_item['Budget'],
                    'social_context': food_item['Social Context'],
                    'user_mood': food_item['User Mood'],
                    'similarity_score': round(score, 3)
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        if self.df is None:
            return {}
        
        stats = {
            'total_items': len(self.df),
            'cuisines': self.df['Cuisine'].nunique(),
            'food_types': self.df['Food Type'].nunique(),
            'unique_cuisines': sorted(self.df['Cuisine'].unique().tolist()),
            'unique_food_types': sorted(self.df['Food Type'].unique().tolist())
        }
        
        return stats
    
    def search_by_cuisine(self, cuisine: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search food items by specific cuisine"""
        filtered_df = self.df[self.df['Cuisine'].str.contains(cuisine, case=False, na=False)]
        
        results = []
        for _, row in filtered_df.head(limit).iterrows():
            results.append({
                'food_name': row['Food Item'],
                'cuisine': row['Cuisine'],
                'food_type': row['Food Type'],
                'taste_notes': row['Taste Notes'],
                'texture': row['Texture']
            })
        
        return results


if __name__ == "__main__":
    # Test the recommender system
    try:
        recommender = FoodRecommender()
        
        # Test recommendation
        test_input = "I'm feeling sad and want some comfort food that's sweet and creamy"
        recommendations = recommender.get_recommendations(test_input, top_k=5)
        
        print(f"\nRecommendations for: '{test_input}'")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['food_name']} ({rec['cuisine']})")
            print(f"   Type: {rec['food_type']}")
            print(f"   Taste: {rec['taste_notes']}")
            print(f"   Similarity Score: {rec['similarity_score']}")
            print()
            
    except Exception as e:
        print(f"Error testing recommender: {str(e)}")
