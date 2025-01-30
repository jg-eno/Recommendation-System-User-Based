# Recommender System with Cache Replacement

## Overview
This Python script implements a **recommender system** that leverages **Neural Collaborative Filtering (NCF)** and **text-based embeddings** to provide personalized movie recommendations. The system is enhanced with a **caching mechanism** that uses multiple eviction strategies (LRU, Relevance, and Hybrid) to optimize performance and ensure relevant recommendations are delivered efficiently. The code is designed to simulate user activity and evaluate the effectiveness of different caching strategies.

---

## Key Components and Functionalities

### 1. **MovieDataset Class**
   - **Purpose**: A custom dataset class for loading movie ratings data.
   - **Methods**:
     - `__init__(self, users, items, ratings)`: Initializes the dataset with user IDs, item IDs, and ratings.
     - `__len__(self)`: Returns the number of ratings in the dataset.
     - `__getitem__(self, idx)`: Retrieves a specific data point (user, item, and rating) by index.

   **Usage**: This class is used to prepare the data for training the NCF model.

---

### 2. **CacheSimulator Class**
   - **Purpose**: Simulates user activity and evaluates the performance of the caching system.
   - **Key Features**:
     - Simulates user requests over a specified period (e.g., 7 days).
     - Tracks cache hits, misses, response times, and cache size over time.
     - Supports multiple caching strategies (LRU, Relevance, Hybrid).
   - **Methods**:
     - `simulate_user_activity(self)`: Simulates user requests over time, with peak and off-peak activity patterns.
     - `_process_user_request(self, user_id, timestamp)`: Processes a single user request, checking the cache for recommendations and updating cache statistics.
     - `get_simulation_results(self)`: Generates a summary of simulation results, including cache hit rate, average response time, and cache size statistics.

   **Usage**: This class is used to test the performance of the caching system under different strategies.

---

### 3. **RecommenderCache Class**
   - **Purpose**: Implements an advanced caching system for the recommender system.
   - **Key Features**:
     - Supports multiple eviction strategies: **LRU (Least Recently Used)**, **Relevance**, and **Hybrid**.
     - Uses **FAISS (Facebook AI Similarity Search)** for efficient similarity search in high-dimensional spaces.
     - Tracks access statistics, relevance scores, and timestamps for cached items.
   - **Methods**:
     - `add_to_cache(self, key, value, vector, relevance)`: Adds an item to the cache with its embedding vector and relevance score.
     - `get_from_cache(self, vector, similarity_threshold, k)`: Retrieves items from the cache based on vector similarity.
     - `_evict_items(self)`: Evicts items from the cache based on the selected strategy.
     - `_calculate_item_score(self, key)`: Computes a composite score for an item based on the caching strategy.
     - `_break_score_ties(self, items_with_scores)`: Breaks ties between items with the same score using additional factors like age, diversity, and popularity.

   **Usage**: This class manages the caching of recommendations and ensures efficient retrieval of relevant items.

---

### 4. **NeuralCollaborativeFiltering (NCF) Class**
   - **Purpose**: Implements a neural collaborative filtering model for recommendation tasks.
   - **Key Features**:
     - Uses **embedding layers** for users and items to capture latent features.
     - Combines user and item embeddings and passes them through a **multi-layer perceptron (MLP)** to predict ratings.
     - Supports dropout for regularization and Xavier initialization for weights.
   - **Methods**:
     - `forward(self, user_input, item_input)`: Performs the forward pass of the model, predicting ratings for user-item pairs.
     - `_init_weights(self)`: Initializes model weights using Xavier initialization.

   **Usage**: This class is the core of the recommendation engine, generating personalized recommendations based on user-item interactions.

---

### 5. **AIEnhancedRecommender Class**
   - **Purpose**: Integrates the NCF model, text embeddings, and caching system into a complete recommender system.
   - **Key Features**:
     - Uses **pre-trained text embeddings** (from `sentence-transformers/all-MiniLM-L6-v2`) to handle cold-start scenarios.
     - Projects text embeddings into the same space as NCF embeddings using a **TextEncoder**.
     - Supports caching of recommendations to improve response times.
   - **Methods**:
     - `prepare_data(self, ratings_df, movies_df)`: Prepares the data for training by encoding users and items.
     - `_train_model(self, ratings_df, movies_df, epochs, batch_size, learning_rate)`: Trains the NCF model and text projection layer.
     - `get_recommendations(self, user_id, n_recommendations, use_cache)`: Generates recommendations for a user, optionally using the cache.
     - `handle_cold_start(self, movie_title, movie_description)`: Handles cold-start items by generating recommendations based on text embeddings.

   **Usage**: This class provides the main interface for generating recommendations and handling cold-start scenarios.

---

### 6. **TextEncoder Class**
   - **Purpose**: Projects text embeddings into the same dimension as NCF embeddings.
   - **Methods**:
     - `forward(self, x)`: Projects input text embeddings into the target dimension using a simple MLP.

   **Usage**: This class is used to align text embeddings with the NCF embedding space, enabling the integration of text-based features into the recommendation system.

---

### 7. **Simulation and Evaluation**
   - **Purpose**: Evaluates the performance of the caching system under different strategies.
   - **Key Functions**:
     - `run_cache_simulation(recommender_system, simulation_config)`: Runs a simulation of user activity and evaluates the caching system.
     - `get_recommendations_for_user(recommender, user_id, n_recommendations)`: Retrieves recommendations for a specific user and prints the results.

   **Usage**: These functions are used to test the recommender system and analyze the effectiveness of different caching strategies.

---

## Methodologies Used

### 1. **Neural Collaborative Filtering (NCF)**
   - Combines user and item embeddings to predict ratings.
   - Uses a multi-layer perceptron (MLP) to model complex interactions between users and items.
   - Trained using a mean squared error (MSE) loss function.

### 2. **Text-Based Embeddings**
   - Uses pre-trained sentence transformers (`all-MiniLM-L6-v2`) to generate embeddings for movie titles and descriptions.
   - Projects these embeddings into the NCF embedding space to handle cold-start items.

### 3. **Caching Strategies**
   - **LRU (Least Recently Used)**: Evicts the least recently accessed items.
   - **Relevance**: Evicts items with the lowest relevance scores.
   - **Hybrid**: Combines LRU and relevance scores to make eviction decisions.

### 4. **FAISS for Similarity Search**
   - Efficiently searches for similar items in high-dimensional embedding spaces.
   - Used to retrieve cached recommendations based on user embeddings.

---

## Workflow

1. **Data Preparation**:
   - Encode users and items using `LabelEncoder`.
   - Prepare training data for the NCF model.

2. **Model Training**:
   - Train the NCF model and text projection layer using user-item interactions and text embeddings.

3. **Recommendation Generation**:
   - Generate recommendations for users using the NCF model.
   - Cache recommendations to improve response times.

4. **Simulation**:
   - Simulate user activity to evaluate the caching system.
   - Analyze cache hit rates, response times, and eviction statistics.

---

## Example Usage

```python
# Initialize the recommender system
recommender = AIEnhancedRecommender(cache_size=1000, embedding_dim=64)

# Load data
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# Prepare data and train the model
recommender.prepare_data(ratings_df, movies_df)

# Run cache simulation
simulation_config = {
    'num_users': 1000,
    'simulation_days': 7,
    'cache_strategies': ['lru', 'relevance', 'hybrid']
}
results = run_cache_simulation(recommender, simulation_config)

# Get recommendations for a specific user
get_recommendations_for_user(recommender, user_id=123, n_recommendations=10)
```

---

## Conclusion
This code provides a robust framework for building and evaluating a recommender system with advanced caching capabilities. By combining neural collaborative filtering, text-based embeddings, and multiple caching strategies, the system delivers personalized recommendations efficiently, even in cold-start scenarios. The simulation functionality allows for thorough testing and optimization of the caching system under different conditions.
