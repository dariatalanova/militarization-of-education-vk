import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import extras


def database_connection():
    conn = psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT']
    )
    print("Connected to the database successfully!")
    cur = conn.cursor()
    return conn, cur


def insert_groups(filepath):
    conn, cur = database_connection()

    df = pd.read_json(filepath, lines=True)
    df = df.replace({np.nan: None})

    insert_query = """
        INSERT INTO publics (owner_id, public_name, city_id, city_name, public_description)
        VALUES %s
        ON CONFLICT (owner_id) DO UPDATE SET
            public_name = EXCLUDED.public_name,
            city_id = EXCLUDED.city_id,
            city_name = EXCLUDED.city_name,
            public_description = EXCLUDED.public_description
    """

    data_to_insert = [
        (
            row['OwnerID'],
            row['PublicName'],
            row['CityID'],
            row['CityName'],
            row['PublicDescription']
        )
        for _, row in df.iterrows()
    ]

    extras.execute_values(cur, insert_query, data_to_insert)
    conn.commit()
    print(f"Вставлено {len(data_to_insert)} записей")

    cur.close()
    conn.close()


if __name__ == '__main__':
    filepath = '../../data/auxiliary/groups_info.jsonl'
    insert_groups(filepath)