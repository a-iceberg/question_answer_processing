# Установить: pip install beautifulsoup4
# pip install openai==0.28
# pip install tiktoken
import openai
import csv
import tiktoken
from bs4 import BeautifulSoup
import re 

# API-ключ OpenAI
openai.api_key = "api_key"

# используемая модель
model_name = "gpt-4o"

# Задание файлов
prompt_file_name = "Промпт3_html2_category.txt"  # Файл с промптом для обработки вопроса
input_file_name = "Формат передачи.txt"  # Файл с входными данными
output_file_name = "Processed_QnA3_category2.csv"  # Имя итогового CSV-файла

# Цены для модели gpt-4o
INPUT_COST_PER_M = 2.50  # для 1M входящих токенов
OUTPUT_COST_PER_M = 10.00  # для 1M выходящих токенов

# Функция для подсчёта токенов
def count_tokens(text, model=model_name):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

# Функция для расчета стоимости токенов
def calculate_cost(tokens, input=True):
    """
    Рассчитывает стоимость в долларах в зависимости от количества токенов.
    Если `input=True`, то считаем для входных токенов.
    Если `input=False`, то считаем для выходных токенов.
    """
    tokens_in_millions = tokens / 1_000_000
    if input:
        return tokens_in_millions * INPUT_COST_PER_M
    else:
        return tokens_in_millions * OUTPUT_COST_PER_M

# Функция для извлечения ID категории из HTML-ответа
def extract_category_from_html(html_response):
    try:
        soup = BeautifulSoup(html_response, "html.parser")

        # Находим блок с категорией
        category_p = soup.find("h3", string="Категория:").find_next("p")
        if category_p:
            category_text = category_p.get_text(strip=True)

            # Пытаемся извлечь только номер категории с помощью регулярного выражения
            category_match = re.match(r"^(\d+)$", category_text)
            if category_match:
                return category_match.group(1)  # Возвращаем только цифру категории
            else:
                print(f"Невозможно определить категорию: {category_text}")
                return "Не определена"
        else:
            print("Блок категории не найден в ответе")
            return "Не определена"
    except Exception as e:
        print(f"Ошибка при разборе HTML для категории: {e}")
        return "Ошибка"

def parse_html_response(html_response):
    try:
        soup = BeautifulSoup(html_response, "html.parser")

        # Извлекаем вопрос
        question_h2 = soup.find("h2", string="Вопрос:")
        if question_h2:
            question_p = question_h2.find_next("p")
            if question_p:
                question = question_p.get_text(strip=True)
            else:
                question = "Вопрос не найден"
        else:
            question = "Заголовок 'Вопрос:' не найден"

        # Извлекаем ответ
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

        # Удаляем блок категории из ответа
        answer = re.sub(r"<h3>Категория:</h3>\s*<p>\d+</p>", "", answer, flags=re.DOTALL).strip()

        # Извлекаем категорию
        category = extract_category_from_html(html_response)

        return question, answer, category

    except Exception as e:
        print(f"Ошибка при разборе HTML: {e}")
        return "Ошибка при разборе вопроса", "Ошибка при разборе ответа", "Ошибка"



# Функция для обработки вопросов и ответов через OpenAI
def process_qna_with_ai(prompt_template, input_data):
    total_tokens = 0  # Итоговая сумма токенов
    total_cost = 0  # Итоговая стоимость
    results = []

    for idx, row in enumerate(input_data, start=1):  # Добавляем индекс для отслеживания номера вопроса
        question = row[0]  # Извлекаем вопрос
        answers = row[1:]  # Извлекаем ответы
        # Формируем текст для отправки в OpenAI
        input_text = f"Спаршенный вопрос с ответами:\n{question}#{'#'.join(answers)}"

        # Подсчитываем входящие токены
        input_tokens = count_tokens(input_text, model=model_name)

        try:
            # Отправляем запрос к OpenAI для получения ответа и переформулированного вопроса
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt_template},
                          {"role": "user", "content": input_text}],
                max_tokens=1500,  # Устанавливаем лимит токенов
                temperature=0.7  # Настроили креативность модели
            )
            # Получаем ответ модели
            model_response = response['choices'][0]['message']['content'].strip()

            # Подсчитываем выходящие токены
            output_tokens = count_tokens(model_response, model=model_name)

            # Обновляем итоговую сумму токенов
            total_tokens += input_tokens + output_tokens

            # Рассчитываем стоимость для входящих и выходящих токенов
            input_cost = calculate_cost(input_tokens, input=True)
            output_cost = calculate_cost(output_tokens, input=False)
            total_cost += input_cost + output_cost

            # Разбираем HTML-ответ
            reformulated_question, answer, category_id = parse_html_response(model_response)

            # Сохраняем результат
            results.append([idx, reformulated_question, answer, category_id])

            # Выводим токены и стоимость после обработки каждого вопроса
            print(f"Вопрос {idx} обработан.")
            print(f"Входящие токены: {input_tokens}, Выходящие токены: {output_tokens}, Всего токенов: {total_tokens}")
            print(f"Стоимость для вопроса {idx}: Вход: ${input_cost:.4f}, Выход: ${output_cost:.4f}, Всего: ${input_cost + output_cost:.4f}")
        except Exception as e:
            # Обработка ошибок
            print(f"Ошибка при обработке вопроса {idx}: {question}\n{str(e)}")
            results.append([idx, "Ошибка", "Ошибка", "Не определена"])

    return results, total_tokens, total_cost

# Загрузка шаблона промпта из файла
with open(prompt_file_name, "r", encoding="windows-1251") as prompt_file:
    prompt_template = prompt_file.read()

# Загрузка входных данных (вопросы и ответы)
input_data = []
with open(input_file_name, "r", encoding="utf-8") as input_file:
    reader = csv.reader(input_file, delimiter="#")  # Указываем # как разделитель
    for row in reader:
        input_data.append(row)

# Обрабатываем данные
processed_results, total_tokens, total_cost = process_qna_with_ai(prompt_template, input_data)

# Сохранение результатов в CSV-файл
with open(output_file_name, "w", encoding="utf-8", newline="") as output_csv:
    writer = csv.writer(output_csv)  # Создаем объект для записи в CSV
    writer.writerow(["id", "переформулированный вопрос", "ответ", "категория"])  # Заголовки колонок
    writer.writerows(processed_results)  # Записываем результаты

print(f"Обработанные результаты сохранены в {output_file_name}")
print(f"Итоговая стоимость: ${total_cost:.4f}")
print(f"Итоговое количество токенов: {total_tokens}")
