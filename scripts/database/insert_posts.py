import os
import json
import zipfile
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


def database_connection():
    conn = psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT']
    )
    print("Connected to the database successfully")
    cur = conn.cursor()
    return conn, cur


def close_database(conn, cur):
    cur.close()
    conn.close()
    print("Database connection closed")


def extract_values(line):
    try:
        dct = json.loads(line.decode('utf-8', errors='ignore'))
        return {
            'OwnerID': dct['OwnerID'],
            'PostID': dct['PostID'],
            'PostText': dct['PostText'],
            'PostDate': dct['PostDate']
        }
    except Exception:
        return None


def batch_insert_values(conn, cur, rows, batch_size):
    insert_query = """
        INSERT INTO posts (id, owner_id, post_id, post_text, post_date)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    values_template = "(%s, %s, %s, %s, TO_DATE(%s, 'DD-MM-YYYY'))"

    try:
        execute_values(cur, insert_query, rows, template=values_template, page_size=batch_size)
        conn.commit()
    except Exception as e:
        print("Ошибка при вставке:", e)
        conn.rollback()


def insert_posts_from_zip(folder_path, batch_size=10000):
    conn, cur = database_connection()

    batch_rows = []
    total_rows = 0

    zip_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.zip')])

    for zip_file_name in zip_files:
        print(f'Начало обработки файла {zip_file_name}')
        zip_file_path = os.path.join(folder_path, zip_file_name)

        with zipfile.ZipFile(zip_file_path, 'r') as z:
            for file_name in z.namelist():
                with z.open(file_name) as f:
                    for line in f:
                        total_rows += 1
                        try:
                            post_data = extract_values(line)
                            if post_data is None:
                                continue

                            row = (
                                f"{post_data['OwnerID']}_{post_data['PostID']}",
                                post_data['OwnerID'],
                                post_data['PostID'],
                                post_data['PostText'],
                                post_data['PostDate']
                            )
                            batch_rows.append(row)

                            if len(batch_rows) >= batch_size:
                                batch_insert_values(conn, cur, batch_rows, batch_size)
                                print(f'Обработано {total_rows} строк (пакет {batch_size})')
                                batch_rows = []

                        except Exception as e:
                            print(e)

    if batch_rows:
        batch_insert_values(conn, cur, batch_rows, batch_size)
        print(f'Финальный пакет: {len(batch_rows)} строк. Всего обработано: {total_rows}')

    close_database(conn, cur)


if __name__ == '__main__':
    folder_path = '../../data/raw-data/'
    insert_posts_from_zip(folder_path)