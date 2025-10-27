def solution(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    # CODE BEGIN
    # Импорты внутри функции (как требуется)
    import rectools
    from rectools import models, dataset
    import numpy as np

    # Фиксация random state
    np.random.seed(42)

    # 1. ПРЕПРОЦЕССИНГ ДАННЫХ
    # Взвешивание взаимодействий на основе watched_pct
    train_processed = train.copy()

    # Создаем веса: комбинация просмотренного процента и длительности
    # Более высокий вес для полностью просмотренных фильмов
    train_processed[rectools.Columns.Weight] = np.where(
        train_processed['watched_pct'] >= 80,
        train_processed['weight'] * 3.0,  # Высокий вес для досмотренных
        np.where(
            train_processed['watched_pct'] >= 50,
            train_processed['weight'] * 2.0,  # Средний вес
            train_processed['weight'] * 0.5   # Низкий вес для недосмотренных
        )
    )

    # Определяем пользователей для рекомендаций (hot users)
    hot_users = train_processed[rectools.Columns.User].unique()

    # 2. СОЗДАНИЕ ДАТАСЕТА через rectools
    dataset_train = rectools.dataset.Dataset.construct(train_processed)

    # 3. ОБУЧЕНИЕ НЕСКОЛЬКИХ МОДЕЛЕЙ (Ensemble подход)

    # Модель 1: EASE (быстрая и эффективная)
    ease_model = rectools.models.EASEModel(regularization=500)
    ease_model.fit(dataset_train)

    # Модель 2: ALS через rectools
    als_model = rectools.models.ImplicitALSWrapperModel(
        factors=128,
        regularization=0.01,
        iterations=20,
        random_state=42,
        num_threads=1
    )
    als_model.fit(dataset_train)

    # Модель 3: Popular (для холодных пользователей и разнообразия)
    popular_model = rectools.models.PopularModel()
    popular_model.fit(dataset_train)

    # 4. ГЕНЕРАЦИЯ КАНДИДАТОВ от разных моделей
    k_candidates = 50  # Больше кандидатов для лучшего покрытия

    # Получаем рекомендации от каждой модели
    ease_recs = ease_model.recommend(
        users=hot_users,
        dataset=dataset_train,
        k=k_candidates,
        filter_viewed=True
    )

    als_recs = als_model.recommend(
        users=hot_users,
        dataset=dataset_train,
        k=k_candidates,
        filter_viewed=True
    )

    popular_recs = popular_model.recommend(
        users=hot_users,
        dataset=dataset_train,
        k=k_candidates,
        filter_viewed=True
    )

    # 5. ОБЪЕДИНЕНИЕ И РАНЖИРОВАНИЕ КАНДИДАТОВ
    # Присваиваем веса каждой модели на основе позиции в рекомендациях
    ease_recs['ease_score'] = 1.0 / (ease_recs['rank'] + 1)
    als_recs['als_score'] = 1.0 / (als_recs['rank'] + 1)
    popular_recs['popular_score'] = 1.0 / (popular_recs['rank'] + 1)

    # Объединяем все рекомендации
    all_recs = pd.concat([
        ease_recs[['user_id', 'item_id', 'ease_score']],
        als_recs[['user_id', 'item_id', 'als_score']],
        popular_recs[['user_id', 'item_id', 'popular_score']]
    ], axis=0)

    # Группируем по (user_id, item_id) и суммируем скоры
    combined = all_recs.groupby(['user_id', 'item_id']).agg({
        'ease_score': 'sum',
        'als_score': 'sum', 
        'popular_score': 'sum'
    }).reset_index()

    # Заполняем пропуски нулями
    combined = combined.fillna(0)

    # 6. ВЫЧИСЛЕНИЕ ФИНАЛЬНОГО СКОРА с весами моделей
    # EASE обычно лучше работает, даем ему больший вес
    combined['final_score'] = (
        combined['ease_score'] * 0.5 +      # EASE - основная модель
        combined['als_score'] * 0.35 +       # ALS - вторая по важности
        combined['popular_score'] * 0.15     # Popular - для разнообразия
    )

    # 7. ДОПОЛНИТЕЛЬНЫЕ УЛУЧШЕНИЯ
    # Добавим бустинг для популярных айтемов среди активных пользователей
    item_popularity = train_processed.groupby(rectools.Columns.Item).agg({
        rectools.Columns.User: 'count',
        'watched_pct': 'mean'
    }).reset_index()
    item_popularity.columns = ['item_id', 'user_count', 'avg_watched_pct']

    # Нормализуем популярность
    item_popularity['popularity_boost'] = (
        (item_popularity['user_count'] / item_popularity['user_count'].max()) * 0.3 +
        (item_popularity['avg_watched_pct'] / 100.0) * 0.2
    )

    # Объединяем с рекомендациями
    combined = combined.merge(
        item_popularity[['item_id', 'popularity_boost']], 
        on='item_id', 
        how='left'
    )
    combined['popularity_boost'] = combined['popularity_boost'].fillna(0)

    # Применяем бустинг
    combined['final_score'] = combined['final_score'] + combined['popularity_boost']

    # 8. ФОРМИРОВАНИЕ ФИНАЛЬНЫХ РЕКОМЕНДАЦИЙ ТОП-10
    final_recommendations = []

    for user_id in hot_users:
        user_recs = combined[combined['user_id'] == user_id].copy()

        # Сортируем по финальному скору
        user_recs = user_recs.sort_values('final_score', ascending=False)

        # Берем топ-10
        top_10 = user_recs.head(10).copy()

        # Если меньше 10 рекомендаций, дополняем популярными
        if len(top_10) < 10:
            # Получаем уже рекомендованные айтемы
            recommended_items = set(top_10['item_id'].values)

            # Берем топ популярных, которых нет в рекомендациях
            popular_items = item_popularity.sort_values('popularity_boost', ascending=False)
            additional_items = popular_items[
                ~popular_items['item_id'].isin(recommended_items)
            ]['item_id'].head(10 - len(top_10)).values

            # Добавляем недостающие
            for item_id in additional_items:
                top_10 = pd.concat([
                    top_10,
                    pd.DataFrame([{
                        'user_id': user_id,
                        'item_id': item_id,
                        'final_score': 0.0
                    }])
                ], ignore_index=True)

        # Убеждаемся, что берем ровно 10
        top_10 = top_10.head(10)

        # Добавляем ранг (0-9)
        top_10['rank'] = range(len(top_10))

        final_recommendations.append(top_10[['user_id', 'item_id', 'rank']])

    # Объединяем все рекомендации
    result = pd.concat(final_recommendations, ignore_index=True)

    # 9. ФИНАЛЬНАЯ ПРОВЕРКА И ФОРМАТИРОВАНИЕ
    # Убеждаемся, что типы данных правильные
    result['user_id'] = result['user_id'].astype(train[rectools.Columns.User].dtype)
    result['item_id'] = result['item_id'].astype(train[rectools.Columns.Item].dtype)
    result['rank'] = result['rank'].astype(int)

    # Проверка: каждый пользователь должен иметь ровно 10 рекомендаций
    recs_per_user = result.groupby('user_id').size()
    assert all(recs_per_user == 10), "Не все пользователи имеют 10 рекомендаций"

    # CODE END
    return result
