import json
import requests
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.environ['TOKEN_1']
VK_API_VERSION = "5.103"
OUTPUT_FILE = "../../data/auxiliary/publics_uni.jsonl"
RETRY_LIMIT = 3
SLEEP_BETWEEN_REQUESTS = 3

def get_publics(keyword, city_id):
    url = "https://api.vk.com/method/groups.search"
    params = {
        "q": keyword,
        "count": 1000,
        "sort": 6,
        "city_id": city_id,
        "access_token": TOKEN,
        "v": VK_API_VERSION,
    }

    for attempt in range(RETRY_LIMIT):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "response" not in data:
                return []

            items = data["response"]["items"]
            return [{"OwnerID": item["id"], "PublicName": item["name"], "CityID": city_id} for item in items]

        except requests.exceptions.RequestException:
            time.sleep(60 * 5 if attempt == RETRY_LIMIT - 1 else 60 * 2)

    return []

def main():
    cities = pd.read_csv("../../data/auxiliary/cities_50k.csv")["CityID"].dropna().astype(int).unique().tolist()
    keywords = ["университет", "факультет", "высшее учебное заведение", "институт",
                "кафедра", "высшая школа"]
    collected_data = []

    with tqdm(total=len(cities) * len(keywords), desc="Поиск пабликов", unit="запрос") as pbar:
        for city in cities:
            for keyword in keywords:
                publics = get_publics(keyword, city)
                if publics:
                    collected_data.extend(publics)

                time.sleep(SLEEP_BETWEEN_REQUESTS)
                pbar.update(1)

    if collected_data:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
            for public in collected_data:
                json.dump(public, file, ensure_ascii=False)
                file.write("\n")

if __name__ == "__main__":
    main()
