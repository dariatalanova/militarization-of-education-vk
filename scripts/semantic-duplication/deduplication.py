import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# ============================================
# НАСТРОЙКИ
# ============================================

DATA_PATH = '../../data/filtered-data/svo_events.csv'
OUTPUT_PATH = '../../data/filtered-data/svo_events_deduplicated.csv'
SIMILARITY_THRESHOLD = 0.95
TIME_WINDOW_DAYS = 2

# ============================================
# ДЕДУПЛИКАЦИЯ
# ============================================

def deduplicate_similar_posts(df, text_column='post_text', similarity_threshold=0.95, time_window_days=2):

    df_sorted = df.sort_values(['city_id', 'post_date']).reset_index(drop=True)
    to_remove = set()

    print(f"\nПоиск дубликатов...")
    print(f"Порог похожести: {similarity_threshold * 100:.0f}%")
    print(f"Временное окно: {time_window_days} дня/дней\n")

    duplicates_found = 0
    df_with_city = df_sorted[df_sorted['city_id'].notna()].copy()

    if len(df_with_city) > 0:
        df_with_city['month'] = df_with_city['post_date'].dt.to_period('M')
        grouped = df_with_city.groupby(['city_id', 'month'])

        for (city_id, month), group in tqdm(grouped, desc="Обработка город-месяц", total=len(grouped)):
            if len(group) < 2:
                continue

            group = group.reset_index(drop=True)
            texts = group[text_column].fillna('').astype(str).tolist()

            if len(texts) < 2 or any(len(t.strip()) == 0 for t in texts):
                continue

            try:
                vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarities = cosine_similarity(tfidf_matrix)

                for i in range(len(group)):
                    if i in to_remove:
                        continue

                    current_date = group.loc[i, 'post_date']

                    for j in range(i + 1, len(group)):
                        if j in to_remove:
                            continue

                        other_date = group.loc[j, 'post_date']
                        if abs((other_date - current_date).days) > time_window_days:
                            continue

                        if similarities[i][j] >= similarity_threshold:
                            if other_date > current_date:
                                to_remove.add(j)
                            else:
                                to_remove.add(i)
                                break
                            duplicates_found += 1

            except Exception:
                continue

    nan_count = df_sorted['city_id'].isna().sum()
    if nan_count > 0:
        print(f"\nПропущено {nan_count} постов с city_id=NaN")

    print(f"\nНайдено дубликатов: {duplicates_found}")

    df_deduplicated = df_sorted[~df_sorted.index.isin(to_remove)].copy()
    return df_deduplicated, list(to_remove)


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates('id').reset_index(drop=True)
    df = df[df['is_event'] == True].reset_index(drop=True)
    df['post_date'] = pd.to_datetime(df['post_date'])

    print(f"Загружено постов: {len(df)}")
    print(f"Период: с {df['post_date'].min()} по {df['post_date'].max()}")
    print(f"Уникальных источников: {df['owner_id'].nunique()}")

    df_clean, removed_indices = deduplicate_similar_posts(
        df,
        text_column='post_text',
        similarity_threshold=SIMILARITY_THRESHOLD,
        time_window_days=TIME_WINDOW_DAYS
    )

    print(f"\nБыло постов: {len(df)}")
    print(f"Удалено дубликатов: {len(removed_indices)}")
    print(f"Осталось постов: {len(df_clean)}")
    print(f"Удалено: {len(removed_indices) / len(df) * 100:.1f}%")

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nСохранено в: {OUTPUT_PATH}")