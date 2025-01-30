import openai
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

# API-ключ OpenAI
openai.api_key = "api-key"

# используемая модель
model_name = "gpt-4o"

# Задание файлов
prompt_file_name = "Промпт_with_category.txt"  # Файл с промптом для обработки вопроса
input_file_name = "50 000 вопросов.xlsx"  # Файл с входными данными
processed_file_name = "Combined_QnA.xlsx"  # Итоговый файл

# Функция для подсчёта токенов
def count_tokens(text, model=model_name):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

# Функция для расчета стоимости токенов
def calculate_cost(tokens, input=True):
    tokens_in_millions = tokens / 1_000_000
    return tokens_in_millions * (2.50 if input else 10.00)

# Функция для извлечения ID категории из HTML-ответа
def extract_category_from_html(html_response):
    try:
        soup = BeautifulSoup(html_response, "html.parser")
        category_p = soup.find("h3", string="Категория:").find_next("p")
        if category_p:
            category_text = category_p.get_text(strip=True)
            category_match = re.match(r"^(\d+)$", category_text)
            return category_match.group(1) if category_match else "Не определена"
        return "Не определена"
    except Exception:
        return "Ошибка"

def parse_html_response(html_response):
    try:
        soup = BeautifulSoup(html_response, "html.parser")
        question_h2 = soup.find("h2", string="Вопрос:")
        question = question_h2.find_next("p").get_text(strip=True) if question_h2 else "Вопрос не найден"
        answer_h2 = soup.find("h2", string="Ответ:")
        answer_elements = []
        if answer_h2:
            for sibling in answer_h2.find_next_siblings():
                if sibling.name == "h2":
                    break
                answer_elements.append(str(sibling))
            answer = ''.join(answer_elements)
        else:
            answer = "Заголовок 'Ответ:' не найден"
        answer = re.sub(r"<h3>Категория:</h3>\s*<p>\d+</p>", "", answer, flags=re.DOTALL).strip()
        category = extract_category_from_html(html_response)
        return question, answer, category
    except Exception:
        return "Ошибка при разборе вопроса", "Ошибка при разборе ответа", "Ошибка"

# Функция для обработки вопросов через OpenAI
def process_qna_with_ai(prompt_template, input_data):
    results = []
    with tqdm(total=len(input_data), desc="Обработка вопросов", unit="вопрос") as pbar:
        for idx, row in input_data.iterrows():
            question_id = idx  # Используем номер строки как ID
            question = row[0]
            answers = row[1:].dropna().tolist()
            input_text = f"Спаршенный вопрос с ответами (ID {question_id}):\n{question}#{'#'.join(map(str, answers))}"
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": input_text}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                model_response = response['choices'][0]['message']['content'].strip()
                reformulated_question, answer, category_id = parse_html_response(model_response)
                results.append([question_id, reformulated_question, answer, category_id])
                pbar.update(1)
                #print(f"Обработан вопрос ID {question_id}")
            except Exception as e:
                print(f"Ошибка при обработке вопроса ID {question_id}: {question}\n{str(e)}")
                results.append([question_id, "Ошибка", "Ошибка", "Ошибка"])
    return results

# Загрузка шаблона промпта
with open(prompt_file_name, "r", encoding="windows-1251") as prompt_file:
    prompt_template = prompt_file.read()

while True:
    # Шаг 1: Загрузка данных
    processed_data = pd.read_excel(processed_file_name)
    non_numeric_or_zero_rows = processed_data[
        pd.to_numeric(processed_data['категория'], errors='coerce').isna() |  # Нечисловые значения
        (pd.to_numeric(processed_data['категория'], errors='coerce') == 0)   # Значения равны 0
    ]

    # Вывод количества и процента таких строк
    total_rows = len(processed_data)
    non_numeric_or_zero_count = len(non_numeric_or_zero_rows)
    print(f"Нечисловые значения или значения 0: {non_numeric_or_zero_count} ({(non_numeric_or_zero_count / total_rows) * 100:.2f}%)")

    # Если все значения корректны, выходим из цикла
    if non_numeric_or_zero_count == 0:
        break

    # Шаг 2: Повторная обработка вопросов с нечисловыми категориями
    input_data = pd.read_excel(input_file_name, header=None)  # Без заголовков

    # Получение строк для повторной обработки с учетом смещения на -1
    ids_to_reprocess = non_numeric_or_zero_rows['id'] - 1  # Уменьшаем ID на 1
    rows_to_reprocess = input_data.iloc[ids_to_reprocess]

    # Повторная обработка
    reprocessed_results = process_qna_with_ai(prompt_template, rows_to_reprocess)

    # Замена строк в исходном файле
    for i, id_value in enumerate(non_numeric_or_zero_rows['id']):
        processed_data.loc[processed_data['id'] == id_value, ['переформулированный вопрос', 'ответ', 'категория']] = reprocessed_results[i][1:]

    # Сохранение обновленного файла
    processed_data.to_excel(processed_file_name, index=False)

# Итоговая проверка
print("Все строки обработаны корректно.")

# Итоговая проверка (нечисловые значения или 0)
processed_data = pd.read_excel(processed_file_name)
final_non_numeric_or_zero_count = processed_data[
    pd.to_numeric(processed_data['категория'], errors='coerce').isna() |  
    (pd.to_numeric(processed_data['категория'], errors='coerce') == 0)   
].shape[0]

print(f"Оставшиеся нечисловые значения или значения 0: {final_non_numeric_or_zero_count}")
