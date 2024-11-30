import openai
import csv
import tiktoken  

# API-ключ OpenAI
openai.api_key = "api_key"

# используемая модель
model_name = "gpt-4"  # Можно заменить на "gpt-4"

# Функция для подсчёта токенов
def count_tokens(text, model=model_name):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

# Функция для обработки вопросов и ответов через OpenAI
def process_qna_with_ai(prompt_template, input_data):
    total_tokens = 0  # Итоговая сумма токенов
    results = []
    for idx, row in enumerate(input_data, start=1):  # Добавляем индекс для отслеживания номера вопроса
        question = row[0]  # Извлекаем вопрос
        answers = row[1:]  # Извлекаем ответы
        # Формируем текст для отправки в OpenAI
        input_text = f"Спаршенный вопрос с ответами:\n{question}#{'#'.join(answers)}"

        # Подсчитываем входящие токены
        input_tokens = count_tokens(input_text, model=model_name)

        try:
            # Отправляем запрос к OpenAI
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_template},  
                    {"role": "user", "content": input_text}
                ],
                max_tokens=1500,  # Устанавливаем лимит токенов
                temperature=0.7  # Настраиваем креативность модели
            )
            # Получаем ответ модели
            model_response = response['choices'][0]['message']['content'].strip()
            
            # Подсчитываем выходящие токены
            output_tokens = count_tokens(model_response, model=model_name)
            
            # Обновляем итоговую сумму токенов
            total_tokens += input_tokens + output_tokens
            
            # Разделяем на переформулированный вопрос и ответ
            reformulated_question, answer = model_response.split("\n", 1)
            results.append([idx, reformulated_question.strip(), answer.strip()])
            
            # Выводим токены после обработки каждого вопроса
            print(f"Вопрос {idx} обработан.")
            print(f"Входящие токены: {input_tokens}, Выходящие токены: {output_tokens}, Всего токенов: {total_tokens}")
        except Exception as e:
            # Обработка ошибок
            print(f"Ошибка при обработке вопроса {idx}: {question}\n{str(e)}")
            results.append([idx, "Ошибка", "Ошибка"])
    return results, total_tokens

# Загрузка шаблона промпта из файла
with open("Промпт.txt", "r", encoding="utf-8") as prompt_file:
    prompt_template = prompt_file.read()  # Читаем шаблон промпта

# Загрузка входных данных (вопросы и ответы)
input_data = []
with open("Формат передачи.txt", "r", encoding="utf-8") as input_file:
    reader = csv.reader(input_file, delimiter="#")  # Указываем # как разделитель
    for row in reader:
        input_data.append(row)

# Обрабатываем данные
processed_results, total_tokens = process_qna_with_ai(prompt_template, input_data[:3])

# Сохранение результатов в CSV-файл
output_file = "Processed_QnA.csv"
with open(output_file, "w", encoding="utf-8", newline="") as output_csv:
    writer = csv.writer(output_csv)  # Создаем объект для записи в CSV
    writer.writerow(["id", "переформулированный вопрос", "ответ"])  # Заголовки колонок
    writer.writerows(processed_results)  # Записываем результаты

print(f"Processed results saved to {output_file}")
print(f"Итоговое количество токенов: {total_tokens}")
