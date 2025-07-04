"""
Anime Recommendation System

Hybrid recommendation system combining collaborative filtering and content-based
filtering using neural networks and Word2Vec for anime recommendations.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class AnimeRecommendationSystem:
    """Hybrid anime recommendation system using content and collaborative filtering."""
    
    def __init__(self):
        """Initialize the recommendation system."""
        self.anime_data = None
        self.ratings_data = None
        self.anime_synopsis = None
        self.model = None
        self.w2v_model = None
        self.anime_similarity = None
        self.user_mapping = {}
        self.anime_mapping = {}
        self.scaler = MinMaxScaler()
        
    def load_data(self, anime_path: str, ratings_path: str, 
                  synopsis_path: Optional[str] = None):
        """Load anime and ratings data."""
        print("Loading anime recommendation data...")
        
        self.anime_data = pd.read_csv(anime_path)
        self.ratings_data = pd.read_csv(ratings_path)
        
        if synopsis_path:
            self.anime_synopsis = pd.read_csv(synopsis_path)
            self._preprocess_synopsis_data()
        
        self._create_mappings()
        
        print(f"Loaded {len(self.anime_data)} anime and {len(self.ratings_data)} ratings")
        
    def _create_mappings(self):
        """Create user and anime ID mappings for model training."""
        user_ids = self.ratings_data['user_id'].unique()
        anime_ids = self.ratings_data['anime_id'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.anime_mapping = {anime_id: idx for idx, anime_id in enumerate(anime_ids)}
        
        self.ratings_data['user_idx'] = self.ratings_data['user_id'].map(self.user_mapping)
        self.ratings_data['anime_idx'] = self.ratings_data['anime_id'].map(self.anime_mapping)
        
        self.num_users = len(user_ids)
        self.num_animes = len(anime_ids)
        
    def _preprocess_synopsis_data(self):
        """Preprocess anime synopsis data for content-based filtering."""
        if self.anime_synopsis is None:
            return
            
        # Handle scores
        self.anime_synopsis['Score'] = pd.to_numeric(
            self.anime_synopsis['Score'], errors='coerce'
        )
        self.anime_synopsis['Score'].fillna(
            self.anime_synopsis['Score'].mean(), inplace=True
        )
        
        # Normalize scores
        self.anime_synopsis['Score'] = self.scaler.fit_transform(
            self.anime_synopsis[['Score']]
        )
        
        # Handle synopsis
        if 'sypnopsis' in self.anime_synopsis.columns:
            self.anime_synopsis['sypnopsis'] = self.anime_synopsis['sypnopsis'].fillna('')
            self.anime_synopsis['sypnopsis'] = self.anime_synopsis['sypnopsis'].apply(
                lambda x: x.lower().split()
            )
        
        # Process genres
        if 'Genres' in self.anime_synopsis.columns:
            self.anime_synopsis['Genres'] = self.anime_synopsis['Genres'].apply(
                lambda x: x.split(', ') if isinstance(x, str) else []
            )
    
    def train_content_model(self):
        """Train content-based filtering model using Word2Vec."""
        if self.anime_synopsis is None:
            print("No synopsis data available for content-based filtering")
            return
            
        print("Training content-based model...")
        
        # Prepare sentences for Word2Vec
        sentences = []
        if 'Genres' in self.anime_synopsis.columns:
            sentences.extend(self.anime_synopsis['Genres'].tolist())
        if 'sypnopsis' in self.anime_synopsis.columns:
            sentences.extend(self.anime_synopsis['sypnopsis'].tolist())
        
        # Train Word2Vec model
        self.w2v_model = Word2Vec(sentences, min_count=1, vector_size=50, workers=4)
        
        # Create vectors for each anime
        self._create_anime_vectors()
        
        # Calculate similarity matrix
        vectors = np.array(self.anime_synopsis['Vector'].tolist())
        self.anime_similarity = cosine_similarity(vectors)
        
        print("Content-based model training completed")
    
    def _create_anime_vectors(self):
        """Create feature vectors for each anime."""
        def get_vector(words):
            if not words or not isinstance(words, list):
                return np.zeros(50)
            
            valid_words = [word for word in words if word in self.w2v_model.wv]
            if not valid_words:
                return np.zeros(50)
            
            return np.mean([self.w2v_model.wv[word] for word in valid_words], axis=0)
        
        # Create vectors from genres and synopsis
        genre_vectors = self.anime_synopsis['Genres'].apply(get_vector)
        
        if 'sypnopsis' in self.anime_synopsis.columns:
            synopsis_vectors = self.anime_synopsis['sypnopsis'].apply(get_vector)
            self.anime_synopsis['Vector'] = (genre_vectors + synopsis_vectors).apply(
                lambda x: x / 2 if not np.isnan(x).any() else np.zeros(50)
            )
        else:
            self.anime_synopsis['Vector'] = genre_vectors
    
    def train_collaborative_model(self, epochs: int = 5, batch_size: int = 1024):
        """Train collaborative filtering model using neural networks."""
        print("Training collaborative filtering model...")
        
        # Prepare training data
        if self.anime_synopsis is not None:
            data = self.ratings_data.merge(
                self.anime_synopsis[['MAL_ID', 'Score']], 
                left_on='anime_id', 
                right_on='MAL_ID',
                how='left'
            )
            data['Score'].fillna(data['Score'].mean(), inplace=True)
        else:
            data = self.ratings_data.copy()
            data['Score'] = 0
        
        # Split data
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # Build model
        self._build_collaborative_model()
        
        # Train model
        if 'Score' in data.columns:
            self.model.fit(
                [train_data['user_idx'], train_data['anime_idx'], train_data['Score']],
                train_data['rating'],
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                verbose=1
            )
        
        print("Collaborative model training completed")
    
    def _build_collaborative_model(self):
        """Build neural network model for collaborative filtering."""
        user_input = Input(shape=(1,), dtype='int64', name='user_input')
        anime_input = Input(shape=(1,), dtype='int64', name='anime_input')
        
        user_embedding = Embedding(self.num_users, 50, input_length=1)(user_input)
        anime_embedding = Embedding(self.num_animes, 50, input_length=1)(anime_input)
        
        user_vec = Flatten()(user_embedding)
        anime_vec = Flatten()(anime_embedding)
        
        if self.anime_synopsis is not None:
            score_input = Input(shape=(1,), name='score_input')
            input_vecs = Concatenate()([user_vec, anime_vec, score_input])
            inputs = [user_input, anime_input, score_input]
        else:
            input_vecs = Concatenate()([user_vec, anime_vec])
            inputs = [user_input, anime_input]
        
        x = Dense(128, activation='relu')(input_vecs)
        x = Dense(64, activation='relu')(x)
        output = Dense(1)(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    def recommend_hybrid(self, user_id: int, num_recommendations: int = 10, 
                        content_weight: float = 0.6, collab_weight: float = 0.4) -> List[Tuple[int, float]]:
        """Generate hybrid recommendations."""
        content_recs = self.recommend_content_based(user_id, num_recommendations * 2)
        collab_recs = self.recommend_collaborative(user_id, num_recommendations * 2)
        
        # Normalize and combine scores
        def normalize_scores(recommendations):
            if not recommendations:
                return {}
            scores = [score for _, score in recommendations]
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return {anime_id: 0.5 for anime_id, _ in recommendations}
            return {
                anime_id: (score - min_score) / (max_score - min_score)
                for anime_id, score in recommendations
            }
        
        content_dict = normalize_scores(content_recs)
        collab_dict = normalize_scores(collab_recs)
        
        # Combine scores
        all_anime_ids = set(content_dict.keys()) | set(collab_dict.keys())
        combined_scores = {}
        
        for anime_id in all_anime_ids:
            content_score = content_dict.get(anime_id, 0)
            collab_score = collab_dict.get(anime_id, 0)
            combined_score = content_weight * content_score + collab_weight * collab_score
            combined_scores[anime_id] = combined_score
        
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def recommend_content_based(self, user_id: int, num_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate content-based recommendations."""
        if self.anime_similarity is None:
            return []
        
        user_ratings = self.ratings_data[self.ratings_data['user_id'] == user_id]
        user_ratings = user_ratings.sort_values('rating', ascending=False)
        top_anime_ids = user_ratings['anime_id'].values
        
        similar_animes = []
        
        for anime_id in top_anime_ids[:10]:
            if anime_id in self.anime_synopsis['MAL_ID'].values:
                idx = self.anime_synopsis[self.anime_synopsis['MAL_ID'] == anime_id].index[0]
                similarities = self.anime_similarity[idx]
                
                similar_indices = np.argsort(similarities)[::-1]
                
                for i, sim_idx in enumerate(similar_indices):
                    if i >= 50:
                        break
                    
                    sim_anime_id = self.anime_synopsis.iloc[sim_idx]['MAL_ID']
                    if sim_anime_id not in top_anime_ids:
                        similar_animes.append((sim_anime_id, similarities[sim_idx]))
        
        # Remove duplicates
        unique_animes = {}
        for anime_id, score in similar_animes:
            if anime_id not in unique_animes or unique_animes[anime_id] < score:
                unique_animes[anime_id] = score
        
        recommendations = sorted(unique_animes.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def recommend_collaborative(self, user_id: int, num_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate collaborative filtering recommendations."""
        if self.model is None or user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        anime_ids = self.ratings_data['anime_id'].unique()
        user_watched = set(self.ratings_data[self.ratings_data['user_id'] == user_id]['anime_id'])
        
        predictions = []
        
        for anime_id in anime_ids:
            if anime_id in user_watched or anime_id not in self.anime_mapping:
                continue
                
            anime_idx = self.anime_mapping[anime_id]
            
            if self.anime_synopsis is not None and anime_id in self.anime_synopsis['MAL_ID'].values:
                score = self.anime_synopsis[self.anime_synopsis['MAL_ID'] == anime_id]['Score'].values[0]
                pred_input = [np.array([user_idx]), np.array([anime_idx]), np.array([score])]
            else:
                pred_input = [np.array([user_idx]), np.array([anime_idx])]
            
            predicted_rating = self.model.predict(pred_input, verbose=0)[0][0]
            predictions.append((anime_id, predicted_rating))
        
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def get_anime_details(self, anime_ids: List[int]) -> List[Dict]:
        """Get detailed information for anime IDs."""
        details = []
        
        for anime_id in anime_ids:
            anime_info = self.anime_data[self.anime_data['MAL_ID'] == anime_id]
            
            if anime_info.empty:
                continue
            
            anime_dict = anime_info.iloc[0].to_dict()
            
            if self.anime_synopsis is not None:
                synopsis_info = self.anime_synopsis[self.anime_synopsis['MAL_ID'] == anime_id]
                if not synopsis_info.empty:
                    anime_dict.update(synopsis_info.iloc[0].to_dict())
            
            details.append(anime_dict)
        
        return details


def main():
    """Example usage of the recommendation system."""
    print("Anime Recommendation System initialized")
    print("Load data and train models to begin generating recommendations")


if __name__ == "__main__":
    main()