import requests
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.environ['TOKEN_1']

def get_cities(city, token=TOKEN, max_retries=3, delay=10):
    url = f"https://api.vk.com/method/database.getCities"
    params = {
        "q": city,
        "access_token": token,
        "need_all": 0,
        "count": 2,
        "v": "5.103",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if "response" in data and "items" in data["response"]:
                return data["response"]["items"]
            else:
                print(f"Ошибка: нет данных для {city} | Ответ: {data}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса {city} (попытка {attempt + 1}): {e}")
            time.sleep(delay)

    return []

settlements = pd.read_csv("../../data/auxiliary/PopulationData.csv")

settlements = settlements[settlements["population"] > 50000].reset_index(drop=True)

data_cities = []

for city in tqdm(settlements["settlement"], desc="Обрабатываем города"):
    cities_info = get_cities(city)
    if cities_info:
        data_cities.extend(cities_info)

df_cities = pd.DataFrame(data_cities)
df_cities.rename(columns={"id": "CityID", "title": "CityName"}, inplace=True)

output_file = "../../data/auxiliary/cities_50k.csv"
df_cities.to_csv(output_file, index=False)
print(f"Данные сохранены в {output_file}")