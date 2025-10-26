# Requirements Document

## Introduction

This feature involves integrating an advanced recommendation system solution (ALS + LightGBM) into the KION movie recommendation assignment notebook while adhering to strict academic constraints. The goal is to improve the MAP@10 score from the current baseline of 0.033 to above 0.071 to achieve a passing grade.

## Glossary

- **Assignment_Notebook**: The main `recsys_project_upd.ipynb` file containing the course assignment
- **Solution_Function**: The `solution()` function that must contain all recommendation logic
- **ALS_Model**: Alternating Least Squares collaborative filtering model for candidate generation
- **LightGBM_Ranker**: Gradient boosting model for ranking candidates
- **MAP_Score**: Mean Average Precision at 10 metric used for evaluation
- **Academic_Constraints**: Strict rules about imports, function signatures, and code placement

## Requirements

### Requirement 1

**User Story:** As a student, I want to integrate my advanced recommendation system into the assignment notebook, so that I can achieve a passing grade while following all academic constraints.

#### Acceptance Criteria

1. THE Solution_Function SHALL accept exactly three parameters: train, users, and items DataFrames
2. THE Solution_Function SHALL contain all logic within the function scope without external dependencies
3. THE Solution_Function SHALL use only the pre-approved imports from the Assignment_Notebook
4. THE Solution_Function SHALL return recommendations in the exact format expected by the scorer
5. THE Solution_Function SHALL execute within the 20-minute time limit on specified hardware

### Requirement 2

**User Story:** As a student, I want to implement ALS candidate generation within the constraints, so that I can generate high-quality movie recommendations.

#### Acceptance Criteria

1. THE Solution_Function SHALL create user-item interaction matrices using only scipy.sparse
2. THE Solution_Function SHALL implement ALS model using the implicit library
3. THE Solution_Function SHALL generate candidate recommendations for each user
4. THE Solution_Function SHALL filter out already-viewed items from candidates
5. THE Solution_Function SHALL handle cold-start users appropriately

### Requirement 3

**User Story:** As a student, I want to implement LightGBM ranking within the constraints, so that I can improve recommendation quality through learned ranking.

#### Acceptance Criteria

1. THE Solution_Function SHALL extract features from ALS embeddings for ranking
2. THE Solution_Function SHALL create positive and negative training examples
3. THE Solution_Function SHALL train a ranking model using sklearn or catboost libraries
4. THE Solution_Function SHALL apply the trained ranker to reorder candidates
5. THE Solution_Function SHALL return top-10 ranked recommendations per user

### Requirement 4

**User Story:** As a student, I want to achieve a MAP@10 score above 0.071, so that I can pass the assignment with a non-zero grade.

#### Acceptance Criteria

1. THE Solution_Function SHALL generate recommendations that achieve MAP@10 > 0.071
2. THE Solution_Function SHALL maintain consistent performance across multiple runs
3. THE Solution_Function SHALL fix all random seeds for reproducibility
4. THE Solution_Function SHALL optimize hyperparameters within execution time limits
5. THE Solution_Function SHALL handle the 10% user subset efficiently

### Requirement 5

**User Story:** As a student, I want to ensure my solution follows all academic integrity rules, so that my work is not disqualified.

#### Acceptance Criteria

1. THE Solution_Function SHALL not use any test data during training
2. THE Solution_Function SHALL not hardcode any test user or item IDs
3. THE Solution_Function SHALL place all code between CODE BEGIN and CODE END markers
4. THE Solution_Function SHALL not modify any instructor-provided code
5. THE Solution_Function SHALL not add new imports or external libraries