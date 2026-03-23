# Militarization of Education in Russia

Detecting war-related events in Russian educational institutions through VK social network data.

In December 2022, the Russian government mandated all state institutions — including schools, kindergartens, colleges, and universities — to maintain active VK accounts. This made VK a valuable archive for social research, as educational institutions regularly report on their activities on these pages.

This project collects posts from over 65,000 educational institutions across Russia and identifies reports of events related to the war in Ukraine held with student participation.

## Pipeline

```
VK API → PostgreSQL → SQL filtering → LLM filtering → Semantic deduplication
```

### 1. Scraping (`scripts/scraping-vk/`)
- `get_cities.py` — fetches a list of cities to scope the search
- `get_groups.py` — retrieves VK group IDs for educational institutions by city
- `get_groups_info.py` — collects metadata for each group (location, description)
- `get_posts.py` — scrapes all posts from collected groups via VK API

All group IDs used for scraping are available in `data/auxiliary/id_groups.csv`.

### 2. Database entry (`scripts/database/`)
- `insert_posts.py` — batch inserts posts into the `posts` table (PostgreSQL)
- `insert_groups.py` — inserts public metadata into the `publics` table

### 3. SQL filtering (`scripts/database/`)
- `filtering_keywords.py` — queries the database for posts containing keywords related to military-patriotic activity; joins with the `publics` table to enrich with location data

### 4. LLM filtering (`scripts/llm-filtering/`)
- `events_detection_llm.py` — classifies filtered posts using two quantized models (IlyaGusev/saiga_llama3_8b and Qwen/Qwen2.5-7B-Instruct); keeps only posts describing a concrete military-patriotic event related to the war in Ukraine held at an educational institution with student participation; model performance is evaluated on a manually annotated dataset.

### 5. Semantic deduplication (`scripts/semantic-duplication/`)
- `deduplication.py` — removes near-identical posts within the same city and a 2-day time window using TF-IDF cosine similarity

## Requirements

- Python 3.9+
- VK API access token (required for scraping)
- PostgreSQL database credentials
- GPU recommended for LLM filtering step (tested on Google Colab with A100)