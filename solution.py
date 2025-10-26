import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

def solution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)

    # === Шаг 1: Индексация пользователей и объектов ===
    user_ids = train_df['user_id'].unique()
    item_ids = train_df['item_id'].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}
    idx2item = {j: i for i, j in item2idx.items()}

    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    # === Шаг 2: Матрица взаимодействий ===
    interaction_matrix = coo_matrix(
        (np.ones(len(train_df))),
        (train_df['item_idx'], train_df['user_idx'])
    ).tocsr()

    # === Шаг 3: ALS-модель для генерации кандидатов ===
    als_model = AlternatingLeastSquares(
        factors=64,
        regularization=0.1,
        iterations=15,
        random_state=42
    )
    als_model.fit(interaction_matrix)

    # === Шаг 4: Подготовка обучающего датасета для ранжирования ===
    positives = train_df[['user_idx', 'item_idx']].drop_duplicates()
    positives['label'] = 1

    # Генерация негативных примеров
    negatives = []
    for u in positives['user_idx'].unique():
        seen = set(positives[positives['user_idx'] == u]['item_idx'])
        recs = als_model.recommend(u, interaction_matrix.T, N=50, filter_already_liked_items=True)
        for item_idx, _ in recs:
            if item_idx not in seen:
                negatives.append({'user_idx': u, 'item_idx': item_idx, 'label': 0})
    negatives = pd.DataFrame(negatives)

    # Объединение
    ranking_df = pd.concat([positives, negatives], ignore_index=True)

    # === Шаг 5: Извлечение признаков ===
    def extract_features(df):
        user_vecs = als_model.user_factors[df['user_idx']]
        item_vecs = als_model.item_factors[df['item_idx']]
        dot_product = np.sum(user_vecs * item_vecs, axis=1)
        return pd.DataFrame({'dot': dot_product})

    X = extract_features(ranking_df)
    y = ranking_df['label']

    # === Шаг 6: Обучение модели ранжирования ===
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)

    # === Шаг 7: Генерация финальных рекомендаций ===
    recommendations = []
    for uid in test_df['user_id'].unique():
        if uid not in user2idx:
            continue
        uidx = user2idx[uid]
        recs = als_model.recommend(uidx, interaction_matrix.T, N=50, filter_already_liked_items=True)
        items = [r[0] for r in recs]
        df = pd.DataFrame({'user_idx': [uidx]*len(items), 'item_idx': items})
        feats = extract_features(df)
        feats_scaled = scaler.transform(feats)
        scores = clf.predict_proba(feats_scaled)[:, 1]
        top_indices = np.argsort(scores)[::-1][:10]
        for rank, idx in enumerate(top_indices):
            recommendations.append({
                'user_id': uid,
                'item_id': idx2item[items[idx]],
                'rank': rank
            })

    return pd.DataFrame(recommendations)
