# Cached Movie Recommender System

A Python-based movie recommendation system with LRU (Least Recently Used) caching for optimized performance. This system uses collaborative filtering to generate personalized movie recommendations while implementing an efficient caching mechanism to reduce computation time for repeated requests.

## Features

- **Collaborative Filtering**: Uses user-user similarity to generate personalized recommendations
- **LRU Cache Implementation**: Efficient caching system for faster repeated recommendations
- **Configurable Cache Size**: Adjustable cache capacity based on system requirements
- **Cache Statistics**: Real-time monitoring of cache utilization
- **Data Preprocessing**: Automated handling of movie and ratings data
- **Flexible Configuration**: Customizable similarity thresholds and recommendation parameters

## Prerequisites

- Python 3.7+
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cached-movie-recommender.git
cd cached-movie-recommender
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn
```

## Usage

### Basic Usage

```python
from movie_recommender import MovieRecommenderCache

# Initialize the recommender
recommender = MovieRecommenderCache(cache_size=1000)

# Load and prepare data
recommender.load_and_prepare_data('ratings.csv', 'movies.csv')

# Get recommendations for a user
recommendations = recommender.get_recommendations(
    user_id=1,
    n_similar_users=10,
    n_recommendations=10,
    similarity_threshold=0.3
)

# Print recommendations
print(recommendations)
```

### Advanced Usage

```python
# Get cache statistics
stats = recommender.get_cache_stats()
print(stats)

# Clear cache
recommender.clear_cache()

# Remove specific recommendations from cache
recommender.remove_from_cache(user_id=1, n_recommendations=10)

# Force recalculation without using cache
recommendations = recommender.get_recommendations(
    user_id=1,
    use_cache=False
)
```

## Sample Output

```
Recommendations for user 1:
                                             movie  movie_score                  timestamp
16  Harry Potter and the Chamber of Secrets (2002)     1.888889 2024-12-29 13:11:51.268443
13    Eternal Sunshine of the Spotless Mind (2004)     1.888889 2024-12-29 13:11:51.268443
6                      Bourne Identity, The (2002)     0.888889 2024-12-29 13:11:51.268443
...

Cache statistics:
{'cache_size': 1000, 'current_items': 1, 'utilization': 0.1}
```

## Data Format

The system expects two CSV files:

### ratings.csv
```
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
...
```

### movies.csv
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
...
```

## Configuration Options

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| cache_size | Maximum number of recommendations to store | 1000 |
| n_similar_users | Number of similar users to consider | 10 |
| n_recommendations | Number of movies to recommend | 10 |
| similarity_threshold | Minimum similarity score for users | 0.3 |

## Performance

The caching system significantly improves performance for repeated recommendations:
- First request: Computational complexity O(n*m) where n is number of users and m is number of movies
- Cached requests: O(1) retrieval time

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Based on collaborative filtering techniques
- Implements LRU cache design pattern
- Uses the MovieLens dataset format

