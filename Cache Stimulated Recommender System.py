import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from datetime import datetime

class MovieRecommenderCache:
    def __init__(self, cache_size=1000):
        """
        Initialize the recommender system with caching
        
        Parameters:
        cache_size (int): Maximum number of recommendations to store in cache
        """
        self.cache_size = cache_size
        self.cache = OrderedDict()  # Using OrderedDict for LRU implementation
        self.matrix_norm = None
        self.user_similarity = None
        
    def load_and_prepare_data(self, ratings_path, movies_path):
        """Load and prepare the movie and ratings data"""
        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
        df = pd.merge(ratings, movies, on='movieId', how='inner')
        
        # Filter movies with more than 100 ratings
        agg_ratings = df.groupby('title').agg(
            mean_rating=('rating', 'mean'),
            number_of_ratings=('rating', 'count')
        ).reset_index()
        agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 100]
        df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
        
        # Create and normalize user-item matrix
        matrix = df_GT100.pivot_table(index='userId', columns='title', values='rating')
        self.matrix_norm = matrix.subtract(matrix.mean(axis=1), axis='rows')
        
        # Calculate user similarity matrix
        self.user_similarity = self.matrix_norm.T.corr()
        
        return "Data prepared successfully"
    
    def get_cache_key(self, user_id, n_recommendations):
        """Generate a unique cache key for the user and number of recommendations"""
        return f"user_{user_id}_rec_{n_recommendations}"
    
    def get_from_cache(self, user_id, n_recommendations):
        """
        Retrieve recommendations from cache if they exist
        
        Returns: None if cache miss, recommendations if cache hit
        """
        cache_key = self.get_cache_key(user_id, n_recommendations)
        if cache_key in self.cache:
            # Move the accessed item to the end (most recently used)
            recommendations = self.cache.pop(cache_key)
            self.cache[cache_key] = recommendations
            return recommendations
        return None
    
    def add_to_cache(self, user_id, n_recommendations, recommendations):
        """Add recommendations to cache with LRU replacement if needed"""
        cache_key = self.get_cache_key(user_id, n_recommendations)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
            
        self.cache[cache_key] = recommendations
    
    def get_recommendations(self, user_id, n_similar_users=10, n_recommendations=10, 
                          similarity_threshold=0.3, use_cache=True):
        """
        Get movie recommendations for a user with caching
        
        Parameters:
        user_id (int): Target user ID
        n_similar_users (int): Number of similar users to consider
        n_recommendations (int): Number of recommendations to return
        similarity_threshold (float): Minimum similarity score for users
        use_cache (bool): Whether to use cache or force recalculation
        
        Returns:
        DataFrame: Ranked movie recommendations
        """
        if use_cache:
            cached_recommendations = self.get_from_cache(user_id, n_recommendations)
            if cached_recommendations is not None:
                return cached_recommendations
        
        # Calculate recommendations if not in cache
        try:
            # Remove target user from candidate list
            user_similarity = self.user_similarity.copy()
            user_similarity.drop(index=user_id, inplace=True)
            
            # Get top n similar users
            similar_users = user_similarity[user_similarity[user_id] > similarity_threshold][user_id]\
                .sort_values(ascending=False)[:n_similar_users]
            
            # Get movies watched by target user and similar users
            picked_userid_watched = self.matrix_norm[self.matrix_norm.index == user_id]\
                .dropna(axis=1, how='all')
            similar_user_movies = self.matrix_norm[self.matrix_norm.index.isin(similar_users.index)]\
                .dropna(axis=1, how='all')
            
            # Remove watched movies
            similar_user_movies.drop(picked_userid_watched.columns, axis=1, 
                                  inplace=True, errors='ignore')
            
            # Calculate movie scores
            item_score = {}
            for movie in similar_user_movies.columns:
                movie_rating = similar_user_movies[movie]
                total = 0
                count = 0
                for u in similar_users.index:
                    if pd.isna(movie_rating[u]) == False:
                        score = similar_users[u] * movie_rating[u]
                        total += score
                        count += 1
                if count > 0:
                    item_score[movie] = total / count
            
            # Create and rank recommendations
            recommendations = pd.DataFrame(item_score.items(), 
                                        columns=['movie', 'movie_score'])\
                .sort_values(by='movie_score', ascending=False)\
                .head(n_recommendations)
            
            # Add timestamp for cache management
            recommendations['timestamp'] = datetime.now()
            
            # Cache the results
            self.add_to_cache(user_id, n_recommendations, recommendations)
            
            return recommendations
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def clear_cache(self):
        """Clear the entire cache"""
        self.cache.clear()
        
    def remove_from_cache(self, user_id, n_recommendations):
        """Remove specific recommendations from cache"""
        cache_key = self.get_cache_key(user_id, n_recommendations)
        if cache_key in self.cache:
            del self.cache[cache_key]
            
    def get_cache_stats(self):
        """Get current cache statistics"""
        return {
            'cache_size': self.cache_size,
            'current_items': len(self.cache),
            'utilization': len(self.cache) / self.cache_size * 100
        }

# Example usage
def main():
    # Initialize recommender with cache
    recommender = MovieRecommenderCache(cache_size=1000)
    
    # Load and prepare data
    recommender.load_and_prepare_data('ratings.csv', 'movies.csv')
    
    # Get recommendations for user 1
    recommendations = recommender.get_recommendations(
        user_id=1,
        n_similar_users=10,
        n_recommendations=10,
        similarity_threshold=0.3
    )
    
    # Print recommendations
    print("\nRecommendations for user 1:")
    print(recommendations)
    
    # Get cache statistics
    print("\nCache statistics:")
    print(recommender.get_cache_stats())
    
    # Get recommendations again (should be from cache)
    cached_recommendations = recommender.get_recommendations(
        user_id=1,
        n_similar_users=10,
        n_recommendations=10
    )
    
    print("\nRecommendations from cache:")
    print(cached_recommendations)

if __name__ == "__main__":
    main()