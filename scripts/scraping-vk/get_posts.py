import requests
import json
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
TOKENS = [v for k, v in os.environ.items() if k.startswith('TOKEN_')]
VK_API_VERSION = "5.103"
OUTPUT_FILE = '../../data/raw-data/posts13.jsonl'
RETRY_LIMIT = 1
SLEEP_BETWEEN_REQUESTS = 30

write_lock = threading.Lock()

def log(msg):
    thread = threading.current_thread().name
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{thread}] {msg}")

def CetGroups():
    df = pd.read_csv('../../data/auxiliary/id_groups.csv')
    print(len(df['owner_id'].to_list()))
    return df['owner_id'].to_list()

def Request(url, params):
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if 'Access denied' not in str(data):
        return data['response']['count'], data['response']['items']
    else:
        return None, None

def GetPosts(owner_id, offset, token):
    url = f'https://api.vk.com/method/wall.get'
    params = {
        "owner_id": owner_id,
        "count": 100,
        "offset": offset,
        "access_token": token,
        "v": VK_API_VERSION,
    }
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return Request(url, params)
        except Exception as e:
            if attempt < RETRY_LIMIT:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            else:
                return None, None

def CheckDates(data):
    if data:
        post_date = datetime.fromtimestamp(data[1]['date'])
        one_month_ago = datetime.now() - timedelta(days=30)
        return post_date > one_month_ago
    return False

def WriteToFile(data, owner_id):
    n = 0
    particular_date = datetime.strptime('01-01-2026', '%d-%m-%Y')
    if data:
        for item in data:
            item_date = datetime.fromtimestamp(item['date'])
            if item_date > particular_date:
                res = {
                    'OwnerID': -abs(owner_id),
                    'PostText': item.get('text', None),
                    'PostID': item['id'],
                    'PostDate': item_date.strftime('%d-%m-%Y')
                }
                with write_lock:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        json.dump(res, f, ensure_ascii=False)
                        f.write('\n')
                n += 1
    return n


def ContinueScraping():
    ids_done = set()
    try:
        with open('../../data/auxiliary/ids_done.txt', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                ids_done.add(int(line))
    except FileNotFoundError:
        return []
    return list(ids_done)


def ProcessWithToken(owners, token, progress):
    for owner in owners:
        try:
            offset = 0
            response = GetPosts(owner, offset, token)
            if response is None:
                continue

            count, data = response
            if count is None:
                continue

            if count > 100:
                if CheckDates(data):
                    WriteToFile(data, owner)
                    for _ in range(0, count // 100):
                        offset += 100
                        _, data = GetPosts(owner, offset, token)
                        if not data:
                            break
                        n = WriteToFile(data, owner)
                        if n < 100:
                            break
        except:
            pass

        finally:
            progress.update(1)
            with write_lock:
                with open('../../data/auxiliary/ids_done.txt', 'a', encoding='utf-8') as f_log:
                    f_log.write(f'{owner}\n')
                    f_log.flush()

def main():
    ids = CetGroups()
    # ids_done = ContinueScraping()
    # ids_left = [i for i in ids if i not in ids_done]
    ids_left = ids

    print(f'Всего пабликов: {len(ids)}')
    # print(f'Обработано ранее: {len(ids_done)}')
    # print(f'Осталось: {len(ids_left)}')

    num_threads = len(TOKENS)
    chunk_size = len(ids_left) // num_threads if ids_left else 0
    id_chunks = [ids_left[i * chunk_size: (i + 1) * chunk_size] for i in range(num_threads)]
    id_chunks[-1].extend(ids_left[num_threads * chunk_size:])

    with tqdm(total=len(ids_left), desc="Обработка пабликов") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(ProcessWithToken, id_chunks[i], TOKENS[i], progress)
                for i in range(num_threads)
            ]
            concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()