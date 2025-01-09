import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from datetime import datetime,timedelta
import faiss
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, List
import random

class MovieDataset(Dataset):
    """Custom Dataset for loading movie ratings data"""
    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'rating': self.ratings[idx]
        }
    
class CacheSimulator:
    def __init__(self, recommender, num_users=1000, simulation_days=7):
        self.recommender = recommender
        self.num_users = num_users
        self.simulation_days = simulation_days
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': [],
            'cache_size_over_time': [],
            'eviction_counts': {
                'lru': 0,
                'relevance': 0,
                'hybrid': 0
            }
        }
        
    def simulate_user_activity(self):
        """Simulate user activity patterns over time"""
        # Create synthetic user activity patterns
        start_time = datetime.now()
        current_time = start_time
        end_time = start_time + timedelta(days=self.simulation_days)

        # Get valid user IDs
        valid_user_ids = self.recommender.user_encoder.classes_

        # Simulate different time periods
        while current_time < end_time:
            # Simulate peak hours (9 AM - 5 PM)
            is_peak = 9 <= current_time.hour <= 17
            num_requests = random.randint(10, 30) if is_peak else random.randint(1, 10)

            # Process requests for this time period
            for _ in range(num_requests):
                # Simulate user request
                user_id = random.choice(valid_user_ids)  # Choose from valid user IDs
                self._process_user_request(user_id, current_time)

            # Update cache statistics
            self.stats['cache_size_over_time'].append({
                'timestamp': current_time,
                'size': len(self.recommender.cache.cache_mapping)
            })

            # Advance time by 1 hour
            current_time += timedelta(hours=1)

            
    def _process_user_request(self, user_id, timestamp):
        """Process a single user request"""
        if user_id not in self.recommender.user_encoder.classes_:
            print(f"Skipping unseen user_id: {user_id}")
            return

        start_time = datetime.now()
        
        # Get user embedding
        user_tensor = torch.LongTensor([
            self.recommender.user_encoder.transform([user_id])[0]]).to(self.recommender.device)
        user_vector = self.recommender.model.user_embedding(user_tensor).cpu().detach().numpy().flatten()
        
        # Try to get recommendations from cache
        cached_result, similarity = self.recommender.cache.get_from_cache(user_vector)
        
        if cached_result is not None:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
            # Generate new recommendations
            recommendations = self.recommender.get_recommendations(
                user_id, n_recommendations=10, use_cache=False
            )
            
            # Calculate relevance based on predicted ratings
            relevance = float(recommendations['predicted_rating'].mean())
            
            # Add to cache
            self.recommender.cache.add_to_cache(f"user_{user_id}_rec_10",
                recommendations,
                user_vector,
                relevance=relevance
            )
            
        # Record response time
        end_time = datetime.now()
        self.stats['average_response_time'].append(
            (end_time - start_time).total_seconds())

                
            # Record response time
        end_time = datetime.now()
        self.stats['average_response_time'].append(
                (end_time - start_time).total_seconds())
        
    def get_simulation_results(self):
        """Generate comprehensive simulation results"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests) * 100 if total_requests > 0 else 0
        
        cache_sizes = pd.DataFrame(self.stats['cache_size_over_time'])
        
        results = {
            'total_requests': total_requests,
            'cache_hit_rate': hit_rate,
            'average_response_time': np.mean(self.stats['average_response_time']),
            'max_cache_size': cache_sizes['size'].max(),
            'avg_cache_size': cache_sizes['size'].mean(),
            'eviction_stats': self.stats['eviction_counts']
        }
        
        return results

def run_cache_simulation(recommender_system, simulation_config=None):
    """Run a complete cache simulation with the given recommender system"""
    if simulation_config is None:
        simulation_config = {
            'num_users': 1000,
            'simulation_days': 7,
            'cache_strategies': ['lru', 'relevance', 'hybrid']
        }
    
    results = {}
    
    # Run simulation for each cache strategy
    for strategy in simulation_config['cache_strategies']:
        print(f"\nRunning simulation with {strategy} strategy...")
        
        # Update cache strategy
        recommender_system.cache.strategy = strategy
        
        # Create and run simulator
        simulator = CacheSimulator(
            recommender_system,
            num_users=simulation_config['num_users'],
            simulation_days=simulation_config['simulation_days']
        )
        
        # Run simulation
        simulator.simulate_user_activity()
        
        # Store results
        results[strategy] = simulator.get_simulation_results()
        
    return results

class TextEncoder(nn.Module):
    """Project text embeddings to the same dimension as NCF embeddings"""
    def __init__(self, input_dim=384, output_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, x):
        return self.projection(x)
    
class RecommenderCache:
    """Advanced caching system for recommender systems with multiple eviction strategies"""
    
    def __init__(self, dimensions: int = 64, cache_size: int = 1000, 
                 strategy: str = "hybrid", time_window: int = 24):
        """
        Initialize the cache with specified parameters
        
        Args:
            dimensions: Size of the embedding vectors
            cache_size: Maximum number of items in cache
            strategy: Caching strategy ('lru', 'relevance', or 'hybrid')
            time_window: Time window for relevance calculation (hours)
        """
        self.dimensions = dimensions
        self.cache_size = cache_size
        self.strategy = strategy
        self.time_window = timedelta(hours=time_window)
        
        # Initialize FAISS index for similarity search
        self.index = faiss.IndexFlatL2(dimensions)
        
        # Main cache storage
        self.cache_mapping: Dict[str, Dict[str, Any]] = OrderedDict()
        
        # Access statistics
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}
        
        # Relevance scores for items
        self.relevance_scores: Dict[str, float] = {}
        
    def _calculate_item_score(self, key: str) -> float:
        """Calculate composite score for an item based on strategy"""
        current_time = datetime.now()
        
        # Time decay factor (exponential decay)
        time_diff = (current_time - self.last_access[key]).total_seconds() / 3600  # hours
        time_score = np.exp(-time_diff / self.time_window.total_seconds() * 3600)
        
        # Access frequency score (normalized)
        max_count = max(self.access_counts.values()) if self.access_counts else 1
        frequency_score = self.access_counts[key] / max_count
        
        # Relevance score (from recommendations)
        relevance_score = self.relevance_scores.get(key, 0.0)
        
        if self.strategy == "lru":
            return time_score
        elif self.strategy == "relevance":
            return 0.3 * frequency_score + 0.7 * relevance_score
        else:  # hybrid
            return 0.4 * time_score + 0.3 * frequency_score + 0.3 * relevance_score
            
    def _update_statistics(self, key: str, relevance: float = None):
        """Update access statistics for a cache item"""
        current_time = datetime.now()
        
        self.last_access[key] = current_time
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        if relevance is not None:
            self.relevance_scores[key] = relevance
            
    def _evict_items(self):
        """Evict items based on chosen strategy"""
        if len(self.cache_mapping) < self.cache_size:
            return
            
        # Calculate scores for all items
        scores = [(key, self._calculate_item_score(key)) 
                 for key in self.cache_mapping.keys()]
        
        # Sort by score and identify items to evict
        sorted_items = sorted(scores, key=lambda x: x[1])
        items_to_evict = sorted_items[:len(sorted_items) - self.cache_size + 1]
        
        # Evict items
        for key, _ in items_to_evict:
            self._remove_item(key)
            
    def _remove_item(self, key: str):
        """Remove an item from cache and all associated data structures"""
        del self.cache_mapping[key]
        del self.access_counts[key]
        del self.last_access[key]
        del self.relevance_scores[key]
        
        # Rebuild FAISS index
        self.index.reset()
        if self.cache_mapping:
            vectors = np.array([item['vector'] for item in self.cache_mapping.values()],
                             dtype=np.float32)
            self.index.add(vectors)
            
    def add_to_cache(self, key: str, value: Any, vector: np.ndarray,
                    relevance: float = None):
        """
        Add item to cache with relevance score
        
        Args:
            key: Unique identifier for the cache entry
            value: The actual data to cache
            vector: Embedding vector for similarity search
            relevance: Relevance score from recommender (optional)
        """
        # Ensure vector is the correct shape
        vector = vector.astype(np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError(f"Vector must have shape ({self.dimensions},)")
            
        # Check if we need to evict items
        self._evict_items()
        
        # Add new item
        self.cache_mapping[key] = {
            'value': value,
            'vector': vector,
            'timestamp': datetime.now()
        }
        
        # Update index
        self.index.add(vector.reshape(1, -1))
        
        # Update statistics
        self._update_statistics(key, relevance)
        
    def get_from_cache(self, vector: np.ndarray, similarity_threshold: float = 0.9, k: int = 1) -> Tuple[Any, float]:
        """
        Retrieve items from cache based on vector similarity.

        Args:
            vector: Query vector.
            similarity_threshold: Minimum similarity for a cache hit.
            k: Number of nearest neighbors to consider.

        Returns:
            Tuple of (cached_value, similarity_score) or (None, 0.0).
        """
        if not self.cache_mapping:
            # Cache is empty
            return None, 0.0

        vector = vector.astype(np.float32)
        #print(f"Cache size: {len(self.cache_mapping)}, FAISS index size: {self.index.ntotal}")
        D, I = self.index.search(vector.reshape(1, -1), k)

        # Check if any valid results are found
        if I[0][0] == -1:
            return None, 0.0

        # Retrieve the key and its cached value
        key = list(self.cache_mapping.keys())[I[0][0]]
        self._update_statistics(key)
        return self.cache_mapping[key]['value'], 1.0 - D[0][0]

        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        return {
            'size': len(self.cache_mapping),
            'max_size': self.cache_size,
            'strategy': self.strategy,
            'hit_rates': {key: self.access_counts[key] for key in self.cache_mapping},
            'avg_relevance': np.mean(list(self.relevance_scores.values()))
            if self.relevance_scores else 0.0
        }

# Example usage with the recommender system
def integrate_with_recommender(recommender_system):
    # Initialize cache with appropriate dimensions
    cache = RecommenderCache(
        dimensions=recommender_system.embedding_dim,
        cache_size=1000,
        strategy="hybrid"
    )
    
    def get_recommendations_cached(user_id: int, n_recommendations: int = 10):
        # Get user embedding
        user_tensor = torch.LongTensor([
            recommender_system.user_encoder.transform([user_id])[0]
        ]).to(recommender_system.device)
        user_vector = recommender_system.model.user_embedding(user_tensor).cpu().numpy().flatten()
        
        # Try to get from cache
        cached_recommendations, similarity = cache.get_from_cache(user_vector)
        
        if cached_recommendations is not None:
            return cached_recommendations
            
        # Cache miss - generate new recommendations
        recommendations = recommender_system.get_recommendations(
            user_id, n_recommendations, use_cache=False
        )
        
        # Calculate relevance score based on prediction confidence
        relevance = float(recommendations['predicted_rating'].mean())
        
        # Add to cache
        cache.add_to_cache(
            f"user_{user_id}_rec_{n_recommendations}",
            recommendations,
            user_vector,
            relevance=relevance
        )
        
        return recommendations
        
    return get_recommendations_cached

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32]):
        super().__init__()
        
        # User and Item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.2))  # Added dropout for regularization
            input_dim = layer_size
            
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # Concatenate user and item embeddings
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        
        # Forward pass through MLP
        for layer in self.fc_layers:
            vector = layer(vector)
            
        output = self.sigmoid(self.output_layer(vector))
        return output
    

class AIEnhancedRecommender:
    def __init__(self, cache_size=1000, embedding_dim=64, device=None):
        self.embedding_dim = embedding_dim
        self.cache = RecommenderCache(dimensions=embedding_dim, cache_size=cache_size)
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Set device (GPU if available, else CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load text model and create projection layer
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.text_projection = TextEncoder(input_dim=384, output_dim=embedding_dim).to(self.device)
        
    def prepare_data(self, ratings_df, movies_df):
        # Encode users and items
        self.user_encoder.fit(ratings_df['userId'].unique())
        self.item_encoder.fit(movies_df['movieId'].unique())
        
        # Create NCF model
        self.model = NeuralCollaborativeFiltering(
            num_users=len(self.user_encoder.classes_),
            num_items=len(self.item_encoder.classes_),
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Train both NCF and text projection
        return self._train_model(ratings_df, movies_df)
    
    def _train_model(self, ratings_df, movies_df, epochs=10, batch_size=1024, learning_rate=0.001):
        # Prepare training data for NCF
        users = self.user_encoder.transform(ratings_df['userId'])
        items = self.item_encoder.transform(ratings_df['movieId'])
        ratings = ratings_df['rating'].values
        
        # Create DataLoader
        dataset = MovieDataset(
            torch.LongTensor(users),
            torch.LongTensor(items),
            torch.FloatTensor(ratings)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize optimizers
        ncf_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        text_optimizer = optim.Adam(self.text_projection.parameters(), lr=learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            ncf_optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        self.text_projection.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                # Move batch to device
                user_batch = batch['user'].to(self.device)
                item_batch = batch['item'].to(self.device)
                rating_batch = batch['rating'].to(self.device)
                
                # Forward pass
                ncf_optimizer.zero_grad()
                text_optimizer.zero_grad()
                
                predictions = self.model(user_batch, item_batch).squeeze()
                loss = criterion(predictions, rating_batch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.text_projection.parameters(), max_norm=1.0)
                
                ncf_optimizer.step()
                text_optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch statistics
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Update learning rate
            scheduler.step(avg_loss)
    
    @torch.no_grad()
    def get_recommendations(self, user_id, n_recommendations=10, use_cache=True):
        # ... (previous get_recommendations method remains the same) ...
        # Generate cache key vector based on user context
        user_tensor = torch.LongTensor([self.user_encoder.transform([user_id])[0]]).to(self.device)
        user_vector = self.model.user_embedding(user_tensor).cpu().numpy().flatten()
        
        if use_cache:
            cached_recommendations = self.cache.get_from_cache(user_vector)
            if cached_recommendations is not None:
                return cached_recommendations
        
        # Generate recommendations using the neural model
        self.model.eval()
        all_items = torch.LongTensor(range(len(self.item_encoder.classes_))).to(self.device)
        user_tensor = user_tensor.repeat(len(all_items))
        
        # Process in batches to avoid memory issues
        batch_size = 1024
        predictions = []
        
        for i in range(0, len(all_items), batch_size):
            batch_users = user_tensor[i:i+batch_size]
            batch_items = all_items[i:i+batch_size]
            batch_pred = self.model(batch_users, batch_items).squeeze()
            predictions.append(batch_pred)
        
        predictions = torch.cat(predictions)
        top_items = torch.topk(predictions, n_recommendations)
        
        recommendations = pd.DataFrame({
            'movieId': self.item_encoder.inverse_transform(top_items.indices.cpu().numpy()),
            'predicted_rating': top_items.values.cpu().numpy()
        })
        
        # Cache the results
        self.cache.add_to_cache(
            f"user_{user_id}_rec_{n_recommendations}",
            recommendations,
            user_vector
        )
        
        return recommendations

    @torch.no_grad()
    def handle_cold_start(self, movie_title, movie_description):
        """Handle cold start items using text embeddings"""
        self.model.eval()
        self.text_projection.eval()
        
        # Generate text embedding for the new item
        text_input = self.text_tokenizer(
            movie_title + " " + movie_description,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get text embeddings and project to NCF embedding space
        text_features = self.text_model(**text_input).last_hidden_state.mean(dim=1)
        item_embedding = self.text_projection(text_features)
            
        # Find similar items using text embeddings
        existing_items = torch.LongTensor(range(len(self.item_encoder.classes_))).to(self.device)
        
        # Process in batches to avoid memory issues
        batch_size = 1024
        similarities = []
        
        for i in range(0, len(existing_items), batch_size):
            batch_items = existing_items[i:i+batch_size]
            batch_embeddings = self.model.item_embedding(batch_items)
            batch_sim = torch.cosine_similarity(item_embedding, batch_embeddings)
            similarities.append(batch_sim)
        
        similarities = torch.cat(similarities)
        top_similar = torch.topk(similarities, 5)
        
        return {
            'similar_items': self.item_encoder.inverse_transform(top_similar.indices.cpu().numpy()),
            'similarity_scores': top_similar.values.cpu().numpy()
        }
    
def get_recommendations_for_user(recommender, user_id, n_recommendations=10):
    """
    Get recommendations for a specific user ID.
    
    Args:
        recommender: Instance of the AIEnhancedRecommender class.
        user_id: The user ID to fetch recommendations for.
        n_recommendations: Number of recommendations to fetch (default is 10).
    
    Returns:
        A DataFrame containing recommended movie IDs and predicted ratings.
    """
    # Check if the user ID is valid
    if user_id not in recommender.user_encoder.classes_:
        print(f"User ID {user_id} is not recognized. Unable to generate recommendations.")
        return None
    
    # Generate recommendations
    recommendations = recommender.get_recommendations(user_id, n_recommendations)
    print(f"Recommendations for User ID {user_id}:\n{recommendations}")
    return recommendations

def main():
    # Initialize recommender system
    recommender = AIEnhancedRecommender(cache_size=1000, embedding_dim=64)

    # Load data
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    
    # Prepare data for recommender
    recommender.prepare_data(ratings_df, movies_df)
    
    # Configure simulation
    simulation_config = {
        'num_users': 1000,
        'simulation_days': 7,
        'cache_strategies': ['lru', 'relevance', 'hybrid']
    }
    
    # Run simulation
    results = run_cache_simulation(recommender, simulation_config)
    
    # Print results
    print("\nSimulation Results:")
    print("==================")
    for strategy, stats in results.items():
        print(f"\nStrategy: {strategy}")
        print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2f}%")
        print(f"Average Response Time: {stats['average_response_time']*1000:.2f}ms")
        print(f"Average Cache Size: {stats['avg_cache_size']:.0f}")
        print(f"Total Requests: {stats['total_requests']}")
    
    user_id = 123
    n_recommendations = 10
    print("\nFetching recommendations...")
    get_recommendations_for_user(recommender, user_id, n_recommendations)

if __name__ == "__main__":
    main()