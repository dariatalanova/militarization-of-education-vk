import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

def database_connection():
    conn = psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT']
    )
    return conn

KEYWORDS = [
    "(^|[^а-яёА-ЯЁ])сво([^а-яёА-ЯЁ]|$)",
    "(^|[^а-яёА-ЯЁa-zA-Z0-9])гум.{0,30}помощ",
    "специальн.{1,10}военн.{1,10}операц",
    "частичн.{1,30}мобилизац",
    "вагнер",
    "спецоперац",
    "маскировочн.{1,10}сет",
    "окопн.{1,10}свечи",
    "письм.{0,10}солдат",
    "письм.{0,10}военнослужащ",
    "наш.{0,5}бойц",
    "наш.{0,5}солдат",
    "наш.{0,5}военнослужащ",
    "наш.{0,5}воинов",
    "своих.{0,5}не.{0,5}бросаем",
    "посылк.{0,10}солдат",
    "открытк.{0,10}солдат",
    "посылк.{0,10}военнослужащ",
    "открытк.{0,10}военнослужащ",
    "для.{0,10}бойцов",
    "для.{0,10}солдат",
    "для.{0,10}воинов",
    "для.{0,10}военнослужащ",
    "ветеран.{0,5}боевых.{0,5}действий",
    "герои.{0,5}нашего.{0,5}времени",
    "время.{0,5}героев",
    "парт.{0,5}героя",
    "героисво",
    "тепло.{0,5}родного.{0,5}дома",
    "событи.{0,30}украин",
    "ветеран.{0,30}украин",
    "денацификац",
    "демилитаризац",
    "(^|[^а-яёА-ЯЁ])гум.{0,30}груз",
    "талисман.{0,10}добра",
    "добрые.{0,10}письма",
    "военн.{0,10}действ.{0,10}украин",
    "поддерж.{0,30}военнослуж",
    "поддерж.{0,30}солдат",
    "поддерж.{0,30}бойцов",
    "поддерж.{0,30}воинов",
    "знание.{0,3}герои",
    "подар.{0,10}солдат",
    "фронтовая.{0,5}открытка",
    "нашим.{0,5}героям",
    "армейский.{0,5}душ",
    "открыт.{0,50}мемориальн.{0,10}доск",
    "блиндажн.{0,10}свеч",
    "лица героев",
    "zдай.{0,5}бумагу"
]

BATCH_SIZE = 10
OUTPUT_PATH = "../../data/filtered-data/posts_svo.csv"

def build_query(keywords):
    conditions = "\n    OR ".join([f"p.post_text ~* '{kw}'" for kw in keywords])
    return f"""
        SELECT p.*, pub.*
        FROM posts p
        JOIN publics pub ON p.owner_id = pub.owner_id
        WHERE p.post_text IS NOT NULL
          AND p.post_text != ''
          AND ({conditions});
    """

load_dotenv()

for i in range(0, len(KEYWORDS), BATCH_SIZE):
    batch = KEYWORDS[i:i + BATCH_SIZE]
    print(f"\nБатч {i // BATCH_SIZE + 1}: ключи {i + 1}–{min(i + BATCH_SIZE, len(KEYWORDS))}")

    conn = database_connection()
    df = pd.read_sql_query(build_query(batch), conn)
    conn.close()

    print(f"Найдено строк: {len(df)}")

    header = (i == 0)
    df.to_csv(OUTPUT_PATH, mode='a', index=False, encoding='utf-8', header=header)
    print(f"Записано в файл")

print(f"\nГотово. Результат в {OUTPUT_PATH}")