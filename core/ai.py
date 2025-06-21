# ======================================================================
# Файл: legal_rag_bot/core/ai.py
# СКОПИРУЙТЕ ЭТОТ КОД В ВАШ ФАЙЛ ai.py
# ======================================================================
import os
from openai import AsyncOpenAI
from typing import List

# Импортируем необходимые компоненты из файла db.py в той же папке
from .db import AsyncSession, vector_search

# Инициализируем асинхронного клиента OpenAI
# Он автоматически подхватит ключ из переменной окружения OPENAI_API_KEY
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Указываем модель для создания векторов
EMBEDDING_MODEL = "text-embedding-3-small"
# Указываем модель для генерации ответов
LLM_MODEL = "gpt-4o"


async def get_embedding(text: str) -> List[float]:
    """
    Создает векторное представление (эмбеддинг) для заданного текста.
    """
    response = await client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


class RAGSystem:
    """
    Класс, инкапсулирующий логику Retrieval-Augmented Generation (RAG).
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.system_prompt = """
Ты — вежливый и профессиональный юридический ассистент в России.
Твоя задача — дать четкий и полезный ответ на вопрос пользователя, основываясь исключительно на предоставленных фрагментах из законов.
Структурируй свой ответ: используй списки, выделение ключевых моментов.
В конце ответа ВСЕГДА указывай ссылку на источник в формате: 'Источник: [URL]'.
Если предоставленные фрагменты не содержат ответа на вопрос, вежливо сообщи об этом и не придумывай информацию.
"""

    async def get_answer(self, user_question: str) -> str:
        """
        Основной метод, реализующий RAG-цепочку.
        """
        # 1. Создаем эмбеддинг для вопроса пользователя
        question_embedding = await get_embedding(user_question)

        # 2. Ищем релевантные фрагменты в базе данных
        relevant_docs = await vector_search(self.db_session, question_embedding)

        if not relevant_docs:
            return "К сожалению, в моей базе знаний не нашлось информации по вашему вопросу. Попробуйте переформулировать его."

        # 3. Формируем контекст и промпт для большой языковой модели (LLM)
        context = "\n\n---\n\n".join([f"Фрагмент из {doc['url']}:\n{doc['text']}" for doc in relevant_docs])

        user_prompt = f"""
Контекст из базы знаний:
{context}

Вопрос пользователя: {user_question}
"""
        # 4. Получаем ответ от LLM
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Делаем ответ более детерминированным
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Произошла ошибка при обращении к OpenAI: {e}")
            return "Прошу прощения, произошла техническая ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
