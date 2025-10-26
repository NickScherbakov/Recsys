# Implementation Plan

- [x] 1. Set up data preprocessing and user-item matrix creation


  - Create user and item ID mappings within solution function scope
  - Build sparse interaction matrix using scipy.sparse.coo_matrix
  - Handle the specific column names from the assignment (user_id, item_id, watched_pct)
  - _Requirements: 1.1, 2.1_






- [ ] 2. Implement ALS candidate generation system
  - [ ] 2.1 Configure and train ALS model using implicit library
    - Set up AlternatingLeastSquares with factors=64, regularization=0.1, iterations=15

    - Fix random_state=42 for reproducibility
    - Train on the interaction matrix
    - _Requirements: 2.2, 4.3_



  
  - [ ] 2.2 Generate candidate recommendations for hot users
    - Extract candidates for each user in hot_users list
    - Filter out already-viewed items
    - Generate 50-100 candidates per user for ranking

    - _Requirements: 2.3, 2.4_

- [ ] 3. Create feature engineering for ranking model
  - [x] 3.1 Extract embedding-based features from ALS model



    - Compute dot products between user and item embeddings
    - Calculate cosine similarities with numerical stability (epsilon=1e-8)
    - Extract user and item embedding norms
    - _Requirements: 3.1_
  

  - [ ] 3.2 Generate positive and negative training examples
    - Create positive examples from training interactions
    - Sample negative examples from ALS candidates not in training
    - Limit to 50 negative examples per user for efficiency



    - _Requirements: 3.2_

- [ ] 4. Implement ranking model training and prediction
  - [ ] 4.1 Train ranking classifier using available libraries
    - Try CatBoostClassifier first (n_estimators=100, random_state=42)

    - Fallback to sklearn GradientBoostingClassifier if needed
    - Apply MinMaxScaler to features before training
    - _Requirements: 3.3, 4.3_
  



  - [ ] 4.2 Apply trained ranker to reorder candidates
    - Extract features for all user-candidate pairs
    - Predict relevance scores using trained model
    - Rank candidates by predicted scores
    - _Requirements: 3.4_


- [ ] 5. Generate final recommendations in correct format
  - [ ] 5.1 Select top-10 recommendations per user
    - Sort candidates by ranking scores in descending order
    - Take top 10 items per user




    - Handle users with fewer than 10 candidates
    - _Requirements: 3.5, 1.4_
  
  - [ ] 5.2 Format output DataFrame for scorer compatibility
    - Create DataFrame with columns: user_id, item_id, rank

    - Ensure rank values are 0-9 for top-10 items
    - Match the exact format expected by rectools.metrics.MAP
    - _Requirements: 1.4_

- [ ] 6. Optimize performance and ensure constraint compliance
  - [ ] 6.1 Implement memory and time optimizations
    - Use sparse matrices throughout to save memory
    - Optimize batch processing for large user sets
    - Add explicit garbage collection for large objects
    - _Requirements: 1.5, 4.4_
  
  - [ ] 6.2 Verify academic constraint compliance
    - Ensure all code is within solution() function scope
    - Verify only approved imports are used (implicit, sklearn, catboost, scipy, numpy, pandas)
    - Confirm no test data is used during training
    - Test that function signature matches assignment requirements
    - _Requirements: 1.1, 1.2, 1.3, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 7. Performance validation and testing
  - [ ] 7.1 Validate MAP@10 performance on test set
    - Run complete solution and measure MAP@10 score
    - Ensure score exceeds 0.071 threshold
    - Test reproducibility across multiple runs
    - _Requirements: 4.1, 4.2_
  
  - [ ] 7.2 Verify execution time and resource constraints
    - Measure total execution time (target < 15 minutes)
    - Monitor memory usage (target < 12GB)
    - Test on the 10% user subset efficiently
    - _Requirements: 1.5, 4.5_