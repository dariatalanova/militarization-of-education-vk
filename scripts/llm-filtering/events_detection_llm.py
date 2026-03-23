import gc
import os
import re
import random
import time
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import zipfile

# ============================================
# НАСТРОЙКИ
# ============================================

MODEL_NAME_Saiga = "IlyaGusev/saiga_llama3_8b"
MODEL_NAME_Qwen = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 32  # T4: 4-8 / V100: 16-24 / A100: 24-32
DATA_PATH = '../../data/filtered-data/posts_svo.csv'
OUTPUT_DIR = '../../data/filtered-data/'

STOPWORDS = [r'гум.{0,20}помощ.{0,50}бездомн', r'гум.{0,20}помощ.{0,50}животн', r'гум.{0,20}помощ.{0,50}собак',
             'для детей участников', "для семей участников",
             "поддержка детей участников", "поддержка семей участников",
             "семей мобилизованных", "детей мобилизованных"]

# ============================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_Saiga)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_Saiga,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config
    )
    print("Модель загружена")
    return model, tokenizer


# ============================================
# ПРОМПТ
# ============================================

def create_prompt_qwen(text):
    return f"""<|im_start|>system
Определи, описано ли мероприятие для детей про СВО.

ДА - если дети участвовали в событии, связанном с СВО (письма солдатам, концерты для военных, поделки, встречи, уроки про СВО и т.д.)

НЕТ - если хотя бы одно не выполняется:
- Нет конкретного мероприятия
- Мероприятие без детей (школьников, юнармейцев, воспитанников, студентов и т.д.)
- Мероприятие не про СВО

Примеры ДА:
- Дети написали письма солдатам
- Школьный концерт для участников СВО
- Изготовили талисманы для бойцов
- Урок про героев спецоперации
- Акция помощи военным

Примеры НЕТ:
- Губернатор поздравил военных (без мероприятия и без детей)
- Городская выставка приглашает жителей (без детей)
- Праздник в честь Дня Победы (без упоминания СВО)

Если нет уверенности да или нет, то НЕ ЗНАЮ

Ответь: ДА, НЕТ или НЕ ЗНАЮ<|im_end|>
<|im_start|>user
{text[:1200]}<|im_end|>
<|im_start|>assistant
"""

def create_prompt_saiga(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты эксперт по анализу постов образовательных учреждений.

ЗАДАЧА: Определи, содержит ли пост информацию о МЕРОПРИЯТИИ, посвященном СВО (специальной военной операции, войне в Украине, спецоперации),
которое прошло в образовательном учреждении.

КРИТЕРИИ для ответа ДА (все должны выполняться):
- В посте описывается КОНКРЕТНОЕ МЕРОПРИЯТИЕ (сбор гумпомощи, встреча, выставка, урок, акция, экскурсия, классный час, концерт, линейка и т.д.)
- Мероприятие ПРОШЛО или ПЛАНИРУЕТСЯ (есть указание на время или факт проведения)
- Мероприятие связано с СВО/спецоперацией/войной в Украине
- В мероприятии участвуют ученики/студенты (из школы, детского сада, лицея, университета, училища, колледжа, детского оркестра и т.д.)

КРИТЕРИИ для ответа НЕТ:
- Пост содержит только общие новости, призывы, поздравления, объявления без описания конкретного мероприятия
- Мероприятие НЕ связано с образовательным учреждением и без участия учеников/студентов
- Общая информация о СВО без упоминания мероприятия
- Мероприятие для взрослых (родителей, учителей) без участия учеников/студентов
- Пост содержит информацию о множестве мероприятий, которые прошли за большой промежуток времени (например, итоги года)

ПРИМЕРЫ ДА:
- Ребята приняли участие в мастер-классе Добрые браслеты для бойцов СВО
- Хор мальчиков и духовой оркестр из Приморья дали концерт в честь Дня Героев Отечества. Этот концерт посвящается участникам СВО
- Ребята решили принять участие в сборе товаров для солдат
- В школе состоялась благотворительная ярмарка. Сумма в размере 430982 руб передана на нужды СВО

ПРИМЕРЫ НЕТ:
- Творческий мастер-класс по работе с алмазной мозаикой для детей участников СВО (НЕТ, поскольку мероприятие не посвящено СВО)
- В зале Театра юного зрителя собрались дети участников спецоперации (НЕТ, поскольку мероприятие не посвящено СВО)
- Кузбасс вошёл в число лидеров по числу заявок на премию «Служение» (НЕТ, поскольку это просто новость)
- Сегодня мы простились с нашим выпускником, участником СВО (НЕТ, поскольку не указано мероприятие)
- За год мы с ребятами изготовили больше 100 окопных свечей (НЕТ, поскольку это годовой отчет, а не конкретное мероприятие)
- Городские открыли выставки СВОих не бросаем. Приглашаем всех москвичей (НЕТ, поскольку не указано участие учеников/студентов в этих выставках)
- Жители Кислово отправили новогодний привет для бойцов СВО (НЕТ, поскольку не указано участие учеников/студентов)
- В Челябинске ведется комплексная работа по развитию адаптивного спорта для ветеранов спецоперации (НЕТ, поскольку не указано участие учеников/студентов)

Ответь ТОЛЬКО одним словом: ДА или НЕТ

<|eot_id|><|start_header_id|>user<|end_header_id|>
Текст поста: {text[:800]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


# ============================================
# КЛАССИФИКАЦИЯ
# ============================================

def classify_batch_to_df(df, model, tokenizer, device, text_column='post_text',
                          batch_size=32, save_every=320,
                          output_dir='data/'):
    df_result = df.copy()
    df_result['is_event'] = False
    df_result['llm_response'] = ''

    texts = df_result[text_column].fillna('').tolist()
    total = len(texts)
    print(f"\nВсего постов для обработки: {total}")

    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    batch_counter = 0

    for i in tqdm(range(0, total, batch_size), desc="Обработка"):
        batch_texts = texts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, total)))
        prompts = [create_prompt_saiga(text) for text in batch_texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        for j, (output, idx) in enumerate(zip(outputs, batch_indices)):
            response = tokenizer.decode(
                output[inputs['input_ids'][j].shape[0]:],
                skip_special_tokens=True
            ).strip()

            response_lower = response.lower()
            has_da = bool(re.search(r'\bда\b', response_lower[:50]))
            has_net = bool(re.search(r'\bнет\b', response_lower[:50]))

            df_result.loc[idx, 'is_event'] = has_da and not has_net
            df_result.loc[idx, 'llm_response'] = response

        processed_count += len(batch_texts)
        batch_counter += 1

        del inputs, outputs
        torch.cuda.empty_cache()

        if batch_counter % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if processed_count % save_every == 0 or processed_count == total:
            csv_file = os.path.join(output_dir, 'svo_events.csv')
            df_result.head(processed_count).to_csv(csv_file, index=False)
            events_so_far = df_result.head(processed_count)['is_event'].sum()
            print(f"\nСохранено: {processed_count} постов | Мероприятий: {events_so_far} ({events_so_far / processed_count * 100:.1f}%)")

    return df_result


if __name__ == '__main__':
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer = load_model()

    with zipfile.ZipFile(DATA_PATH) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)

    df = df[~df['post_text'].str.contains('|'.join(STOPWORDS), case=False, regex=True)].reset_index(drop=True)
    df = df.drop_duplicates('id').reset_index(drop=True)
    print(f"Загружено постов: {len(df)}")

    start_time = time.time()
    df_classified = classify_batch_to_df(
        df, model, tokenizer, device,
        text_column='post_text',
        batch_size=BATCH_SIZE,
        save_every=320,
        output_dir=OUTPUT_DIR
    )
    print(f"\nВремя выполнения: {(time.time() - start_time) / 60:.1f} мин")