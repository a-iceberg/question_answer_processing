import openai
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
import re
from tqdm import tqdm  # Для отображения прогресс-бара

# API-ключ OpenAI
openai.api_key = "api-key"

# используемая модель
model_name = "gpt-4o"

# Задание файлов
prompt_file_name = "Промпт_with_category.txt"  # Файл с промптом для обработки вопроса
input_file_name = "50 000 вопросов.xlsx"  # Файл с входными данными
output_file_name = "Processed_QnA.xlsx"  # Имя итогового файла

# Цены для модели gpt-4o
INPUT_COST_PER_M = 2.50  # для 1M входящих токенов
OUTPUT_COST_PER_M = 10.00  # для 1M выходящих токенов

# Функция для подсчёта токенов
def count_tokens(text, model=model_name):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

# Функция для расчета стоимости токенов
def calculate_cost(tokens, input=True):
    tokens_in_millions = tokens / 1_000_000
    if input:
        return tokens_in_millions * INPUT_COST_PER_M
    else:
        return tokens_in_millions * OUTPUT_COST_PER_M

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


# Функция для обработки вопросов и ответов через OpenAI
def process_qna_with_ai(prompt_template, input_data):
    total_tokens = 0
    total_cost = 0
    results = []

    with tqdm(total=len(input_data), desc="Обработка вопросов", unit="вопрос") as pbar:
        for idx, row in input_data.iterrows():
            question = row[0]
            answers = row[1:].dropna().tolist()
            input_text = f"Спаршенный вопрос с ответами:\n{question}#{'#'.join(map(str, answers))}"
            input_tokens = count_tokens(input_text, model=model_name)

            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "system", "content": prompt_template},
                              {"role": "user", "content": input_text}],
                    max_tokens=1500,
                    temperature=0.7
                )
                model_response = response['choices'][0]['message']['content'].strip()
                output_tokens = count_tokens(model_response, model=model_name)
                total_tokens += input_tokens + output_tokens
                input_cost = calculate_cost(input_tokens, input=True)
                output_cost = calculate_cost(output_tokens, input=False)
                total_cost += input_cost + output_cost

                reformulated_question, answer, category_id = parse_html_response(model_response)
                results.append([idx + 1, reformulated_question, answer, category_id])

                # Обновляем прогресс-бар
                pbar.set_postfix({"Токены": total_tokens, "Стоимость": total_cost})
                pbar.update(1)

                # Сохраняем каждые 100 вопросов
                if (idx + 1) % 100 == 0:
                    save_to_excel(results, output_file_name)
                    results.clear()  # Очищаем список для следующей партии
            except Exception as e:
                print(f"Ошибка при обработке вопроса {idx + 1}: {question}\n{str(e)}")
                results.append([idx + 1, "Ошибка", "Ошибка", "Не определена"])

    return results, total_tokens, total_cost

# Функция для сохранения данных в Excel
def save_to_excel(data, file_name):
    df = pd.DataFrame(data, columns=["id", "переформулированный вопрос", "ответ", "категория"])
    with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        start_row = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
        df.to_excel(writer, index=False, header=start_row == 0, startrow=start_row)

# Загрузка шаблона промпта из файла
with open(prompt_file_name, "r", encoding="windows-1251") as prompt_file:
    prompt_template = prompt_file.read()

# Загрузка входных данных из XLSX
data = pd.read_excel(input_file_name, sheet_name=0, header=None)

# Инициализация файла
pd.DataFrame(columns=["id", "переформулированный вопрос", "ответ", "категория"]).to_excel(output_file_name, index=False)

# Обра батываем данные
processed_results, total_tokens, total_cost = process_qna_with_ai(prompt_template, data)

# Сохраняем оставшиеся данные
if processed_results:
    save_to_excel(processed_results, output_file_name)

print(f"Processed results saved to {output_file_name}")
print(f"Итоговая стоимость: ${total_cost:.4f}")
print(f"Итоговое количество токенов: {total_tokens}")
