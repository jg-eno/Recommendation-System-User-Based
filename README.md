# Recommender System with Neural Collaborative Filtering and Caching

## Overview
This project implements a **Recommender System** that uses **Neural Collaborative Filtering (NCF)** to generate personalized recommendations for users. It also incorporates an advanced caching mechanism to improve the efficiency and scalability of the recommendation process. The caching system integrates strategies like **Least Recently Used (LRU)**, **Relevance-Based**, and **Hybrid** eviction policies to handle frequent and large-scale user requests.

---

## Features
1. **Neural Collaborative Filtering (NCF):**
   - Deep learning-based model to predict user ratings for items.
   - Embedding layers for users and items.
   - Fully connected layers (MLP) for modeling non-linear user-item interactions.

2. **Recommender Cache:**
   - Uses embeddings and FAISS for similarity-based searches.
   - Supports multiple eviction strategies:
     - **LRU (Least Recently Used):** Removes least accessed items.
     - **Relevance-Based:** Prioritizes items with higher predicted ratings.
     - **Hybrid:** Combines time-decay, access frequency, and relevance.

3. **Cold-Start Problem Handling:**
   - Uses text embeddings (via `sentence-transformers`) for new items.
   - Fallback mechanisms for new users with no prior data.

4. **Performance Simulation:**
   - Simulates user activity over a configurable period.
   - Tracks metrics such as cache hit rate, response time, and cache size.

---

## Workflow
### 1. **Data Loading**
   - **Files:** `movies.csv` (movie metadata) and `ratings.csv` (user-item interactions).
   - Loaded into Pandas DataFrames for preprocessing.

### 2. **Model Training**
   - The NCF model is trained using the `ratings.csv` data.
   - Embedding layers learn latent factors for users and items.
   - Fully connected layers predict ratings based on user-item embeddings.

### 3. **Recommendation Generation**
   - For each user, the model predicts ratings for all items.
   - Top `N` items (based on predicted ratings) are returned as recommendations.

### 4. **Recommender Cache**
   - Stores recommendations for frequent user requests.
   - Uses FAISS to perform fast similarity searches on user embeddings.
   - Handles cache eviction based on the chosen strategy (e.g., LRU, Relevance).

### 5. **Cold-Start Handling**
   - For new users: Uses default embeddings or synthetic data.
   - For new items: Text embeddings are generated from metadata and mapped to the NCF embedding space.

### 6. **Performance Simulation**
   - Simulates user activity over a specified time period.
   - Measures cache performance metrics, including:
     - **Cache Hit Rate:** Percentage of requests served from cache.
     - **Response Time:** Average time to generate recommendations.
     - **Cache Size:** Memory usage over time.

---

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jg-eno/Recommendation-System-User-Based.git
   cd Recommendation-System-User-Based
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data:**
   Place `movies.csv` and `ratings.csv` in the project directory.

4. **Set Up GPU (Optional):**
   If you have a CUDA-compatible GPU, ensure PyTorch is installed with GPU support.
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Usage
### 1. **Train the Model and Cache Recommendations**
Run the main script to train the model, simulate user activity, and evaluate the cache:
```bash
python3 Recommender_Cache_Replacement.py
```

### 2. **Fetch Recommendations for a Specific User**
To get recommendations for a particular user ID:
```python
from Cahce import get_recommendations_for_user
user_id = 123  # Replace with desired user ID
n_recommendations = 10
get_recommendations_for_user(recommender, user_id, n_recommendations)
```

### 3. **Simulation Results**
After running the simulation, performance metrics (e.g., cache hit rate, response time) are displayed for each cache strategy.

---

## File Structure
```
.
├── Recommender_Cache_Replacement.py                 # Main script for training, caching, and simulation
├── movies.csv               # Movie metadata (input file)
├── ratings.csv              # User-item interactions (input file)
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## Performance Metrics
During simulation, the following metrics are measured:
1. **Cache Hit Rate:**
   - Percentage of user requests served from the cache.

2. **Response Time:**
   - Time taken to generate recommendations.

3. **Cache Size:**
   - Monitors memory usage over time.

4. **Eviction Stats:**
   - Tracks the number of evictions for each strategy.

---

## Customization
- **Change Cache Strategy:**
  Modify the `simulation_config` dictionary in the `main()` function:
  ```python
  simulation_config = {
      'num_users': 1000,
      'simulation_days': 7,
      'cache_strategies': ['lru', 'relevance', 'hybrid']
  }
  ```

- **Adjust Model Parameters:**
  Update the `embedding_dim`, `layers`, or `learning_rate` in the `NeuralCollaborativeFiltering` class.

---

## Dependencies
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- FAISS
- scikit-learn
- Pandas
- NumPy

Install dependencies via `pip install -r requirements.txt`.

---

## Future Improvements
1. **Dynamic Caching:**
   - Adjust cache strategies dynamically based on workload.

2. **Hybrid Models:**
   - Combine NCF with content-based filtering for better cold-start handling.

3. **Distributed Caching:**
   - Implement distributed caching for large-scale systems.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- PyTorch for deep learning tools.
- HuggingFace for pre-trained text models.
- FAISS for efficient similarity search.
- scikit-learn for preprocessing utilities.

