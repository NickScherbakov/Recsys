# Design Document

## Overview

This design outlines the integration of an advanced ALS + LightGBM recommendation system into the KION assignment notebook. The solution will replace the current EASE baseline while adhering to strict academic constraints including function signature, import restrictions, and execution time limits.

## Architecture

### High-Level Flow
```
Input: train, users, items DataFrames
↓
Data Preprocessing & User-Item Matrix Creation
↓
ALS Model Training (Candidate Generation)
↓
Feature Engineering from ALS Embeddings
↓
LightGBM Training (Ranking Model)
↓
Final Recommendation Generation
↓
Output: Recommendations DataFrame
```

### Constraint-Compliant Design
- **Single Function Scope**: All logic contained within `solution()` function
- **Library Restrictions**: Use only implicit, sklearn, catboost, scipy, numpy, pandas
- **No External Dependencies**: All helper functions defined within solution scope
- **Memory Efficient**: Handle 10% user subset efficiently within 16GB RAM limit

## Components and Interfaces

### 1. Data Preprocessing Component
**Purpose**: Transform input data into ALS-compatible format
**Interface**:
```python
def preprocess_data(train_df):
    # Create user-item mappings
    # Build interaction matrix
    # Return matrix, mappings
```

### 2. ALS Candidate Generator
**Purpose**: Generate initial candidate recommendations
**Interface**:
```python
def generate_candidates(interaction_matrix, user_mappings):
    # Train ALS model
    # Generate top-N candidates per user
    # Return candidates with scores
```

### 3. Feature Engineering Component
**Purpose**: Extract ranking features from ALS embeddings
**Interface**:
```python
def extract_ranking_features(user_factors, item_factors, interactions):
    # Compute dot products, cosine similarities
    # Create positive/negative examples
    # Return feature matrix and labels
```

### 4. LightGBM Ranking Component
**Purpose**: Train and apply ranking model
**Interface**:
```python
def train_ranker(features, labels):
    # Train LightGBM/CatBoost classifier
    # Return trained model
```

### 5. Recommendation Assembly Component
**Purpose**: Generate final top-10 recommendations
**Interface**:
```python
def assemble_recommendations(candidates, ranker, hot_users):
    # Apply ranker to candidates
    # Select top-10 per user
    # Format output DataFrame
```

## Data Models

### User-Item Interaction Matrix
```python
# Sparse CSR matrix format
interaction_matrix: scipy.sparse.csr_matrix
# Shape: (n_items, n_users)
# Values: implicit feedback (1.0 for interactions)
```

### ALS Model Outputs
```python
user_factors: np.ndarray  # Shape: (n_users, n_factors)
item_factors: np.ndarray  # Shape: (n_items, n_factors)
```

### Ranking Features
```python
features = {
    'dot_product': float,      # User-item embedding dot product
    'cosine_similarity': float, # Normalized similarity
    'user_norm': float,        # User embedding magnitude
    'item_norm': float         # Item embedding magnitude
}
```

### Output Format
```python
recommendations = pd.DataFrame({
    'user_id': int,
    'item_id': int, 
    'rank': int     # 0-9 for top-10
})
```

## Error Handling

### Cold Start Users
- **Strategy**: Skip users not in training data
- **Fallback**: Could implement popularity-based recommendations (optional)

### Empty Recommendations
- **Detection**: Check if ALS generates candidates for user
- **Handling**: Skip user if no candidates available

### Memory Management
- **Sparse Matrices**: Use scipy.sparse for memory efficiency
- **Batch Processing**: Process users in batches if needed
- **Garbage Collection**: Explicit cleanup of large objects

### Numerical Stability
- **Division by Zero**: Add epsilon (1e-8) to denominators
- **Feature Scaling**: Use MinMaxScaler for ranking features
- **Random Seed**: Fix all random states to 42

## Testing Strategy

### Unit Testing Approach
1. **Data Preprocessing**: Verify matrix dimensions and mappings
2. **ALS Training**: Check model convergence and factor shapes
3. **Feature Engineering**: Validate feature value ranges
4. **Ranking Model**: Verify training completion and predictions
5. **Output Format**: Ensure correct DataFrame structure

### Integration Testing
1. **End-to-End Flow**: Test complete pipeline with sample data
2. **Performance Validation**: Measure MAP@10 on validation set
3. **Constraint Compliance**: Verify no unauthorized imports or external calls
4. **Execution Time**: Ensure completion within 20-minute limit

### Performance Benchmarks
- **Target MAP@10**: > 0.071 (minimum for passing grade)
- **Stretch Goal**: > 0.080 (for higher score)
- **Execution Time**: < 15 minutes (buffer for system variation)
- **Memory Usage**: < 12GB (buffer within 16GB limit)

## Implementation Considerations

### Hyperparameter Selection
- **ALS Parameters**: factors=64, regularization=0.1, iterations=15
- **LightGBM Parameters**: n_estimators=100, learning_rate=0.1
- **Candidate Pool Size**: 50-100 candidates per user for ranking

### Library Usage Strategy
- **Primary Ranking**: Try CatBoost first (better performance)
- **Fallback**: Use sklearn.ensemble.GradientBoostingClassifier if needed
- **Feature Scaling**: sklearn.preprocessing.MinMaxScaler

### Optimization Techniques
- **Negative Sampling**: Limit negative examples to 50 per user
- **Feature Selection**: Use only most informative embedding-based features
- **Early Stopping**: Implement for ranking model if time permits

### Academic Compliance
- **Code Placement**: All logic within solution() function scope
- **Import Verification**: Double-check only approved libraries used
- **Test Data Isolation**: Ensure no test data leakage in training
- **Reproducibility**: Fix all random seeds consistently