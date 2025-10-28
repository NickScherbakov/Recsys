# KION Movie Recommendation System

An advanced recommendation system for KION movies, implementing ensemble-based collaborative filtering with weighted interaction models.

## 📋 Project Overview

This project implements a sophisticated recommendation system designed to provide personalized movie recommendations for KION users. The system leverages multiple collaborative filtering algorithms and combines them through an ensemble approach to achieve superior prediction accuracy.

## 🎯 Task Description

**Objective:** Develop a recommendation system for KION movies that achieves high performance on the MAP@10 (Mean Average Precision at 10) metric.

**Key Requirements:**
- The solution must be fully encapsulated within a single `solution()` function
- Quality is evaluated using MAP@10 metric on a held-out week of data
- The system processes 10% of total users for computational efficiency
- All code must execute within 20 minutes on 4 CPU cores with 16GB RAM
- Random states must be fixed to ensure reproducible results
- No hardcoding of test data elements is allowed

**Dataset:** The system works with KION movie interaction data, including:
- User viewing history
- Watch percentage completion rates
- Interaction timestamps
- Movie metadata

**Evaluation:** Performance is measured by how accurately the system can predict which movies a user will watch in the next week, ranking recommendations by relevance.

## 🔧 Technology Stack

### Core Libraries
- **rectools (v0.17.0)**: Primary recommendation framework providing unified interfaces
- **implicit (v0.7.2)**: Matrix factorization algorithms (ALS)
- **pandas (v2.3.3)**: Data manipulation and preprocessing
- **numpy (v2.3.3)**: Numerical computations
- **scipy (v1.16.2)**: Sparse matrix operations
- **scikit-learn (v1.7.2)**: Machine learning utilities

### Environment
- Python 3.13.0
- CPU-only execution (no GPU required)
- Multi-threading support for parallel model training

## 🚀 Solution Architecture

### 1. Data Preprocessing
The system implements sophisticated interaction weighting based on viewing behavior:
- **High weight (3x)**: Movies watched ≥80% completion
- **Medium weight (2x)**: Movies watched 50-79% completion  
- **Low weight (0.5x)**: Movies watched <50% completion

This weighting scheme ensures that completed movies have stronger influence on recommendations than abandoned content.

### 2. Ensemble Model Approach

The solution employs a multi-model ensemble strategy combining three complementary algorithms:

#### **EASE Model (Weight: 50%)**
- Embarrassingly Shallow Autoencoders for Sparse data
- Efficient linear autoencoder for collaborative filtering
- Regularization parameter: 500
- Primary model due to superior performance on this dataset

#### **ALS Model (Weight: 35%)**
- Alternating Least Squares matrix factorization
- Configuration: 128 factors, 20 iterations, regularization 0.01
- Captures latent user-item relationships
- Single-threaded for reproducibility

#### **Popular Model (Weight: 15%)**
- Recommends globally popular items
- Provides diversity and handles cold-start scenarios
- Ensures coverage for users with limited history

### 3. Candidate Generation
Each model generates up to 50 candidate recommendations per user, which are then merged and re-ranked.

### 4. Score Fusion & Ranking
The system combines model predictions using weighted rank-based scoring:
```
final_score = (ease_score × 0.5) + (als_score × 0.35) + (popular_score × 0.15) + popularity_boost
```

Additional popularity boosting is applied based on:
- Item view count normalization
- Average watch completion rate

### 5. Final Recommendation Selection
For each user:
- Candidates are sorted by final score
- Top 10 items are selected
- If fewer than 10 candidates exist, popular items are used to fill remaining slots
- Results are validated to ensure exactly 10 recommendations per user

## 📊 Key Features

✅ **Multi-Model Ensemble**: Combines strengths of different algorithms  
✅ **Weighted Interactions**: Uses viewing completion as quality signal  
✅ **Popularity Boosting**: Balances personalization with trending content  
✅ **Cold-Start Handling**: Fallback to popular items for new users  
✅ **Reproducible Results**: Fixed random seeds for consistent evaluation  
✅ **Efficient Processing**: Optimized for quick execution on standard hardware  

## 🎓 Performance Considerations

- **Metric**: MAP@10 (Mean Average Precision at 10)
- **Validation**: Held-out week evaluation
- **Processing Time**: <20 minutes for full pipeline
- **Scalability**: Handles 10% user sample efficiently (~thousands of users)

## 📦 Installation

```bash
pip install implicit==0.7.2 "rectools[all]==0.17.0" pandas==2.3.3 numpy==2.3.3 scipy==1.16.2 requests==2.32.5 catboost==1.2.8 scikit-learn==1.7.2
```

## 🔬 Usage

The complete solution is contained in the `solution()` function which accepts:
- `train`: Training DataFrame with user-item interactions
- `users`: User metadata DataFrame
- `items`: Item (movie) metadata DataFrame

Returns a DataFrame with columns: `user_id`, `item_id`, `rank` (0-9)

## 📈 Future Improvements

Potential enhancements to further boost performance:
- Content-based filtering using movie metadata (genres, directors, actors)
- Deep learning models (neural collaborative filtering)
- Sequential pattern mining for temporal dynamics
- Context-aware recommendations (time of day, device type)
- A/B testing framework for online evaluation

## 👥 Development

This recommendation system was developed as part of a course project on Recommendation Systems, demonstrating advanced techniques in collaborative filtering and ensemble learning.

## 📄 License

See LICENSE file for details.
