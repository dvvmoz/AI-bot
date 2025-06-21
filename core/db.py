# ======================================================================
# Файл: legal_rag_bot/core/db.py
# СКОПИРУЙТЕ ЭТОТ КОД В ВАШ ФАЙЛ db.py
# ======================================================================
import os
from dotenv import load_dotenv
from typing import List, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete
from sqlalchemy.sql import text

# Импортируем модели из файла models.py в той же папке
from .models import Base, LegalDocument

# Загружаем переменные окружения из .env файла
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Проверяем, что URL базы данных задан
if not DATABASE_URL:
    raise ValueError("Необходимо установить переменную окружения DATABASE_URL")

# Создаем асинхронный "движок" для подключения к БД
engine = create_async_engine(DATABASE_URL)

# Создаем фабрику асинхронных сессий для взаимодействия с БД
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """
    Инициализирует базу данных. Создает расширение pgvector и все таблицы.
    Эту функцию нужно будет запустить один раз при первом запуске проекта.
    """
    async with engine.begin() as conn:
        # Устанавливаем расширение pgvector, если его еще нет
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        # Создаем все таблицы, описанные в моделях
        await conn.run_sync(Base.metadata.create_all)
    print("База данных успешно инициализирована.")


async def get_db_session() -> AsyncSession:
    """
    Функция-генератор для получения сессии базы данных.
    """
    async with AsyncSessionFactory() as session:
        yield session


# --- Функции для работы с данными ---

async def add_documents(session: AsyncSession, documents: List[LegalDocument]):
    """Добавляет список документов в базу данных."""
    session.add_all(documents)
    await session.commit()


async def delete_documents_by_url(session: AsyncSession, url: str):
    """Удаляет все фрагменты документа по указанному URL."""
    statement = delete(LegalDocument).where(LegalDocument.source_url == url)
    await session.execute(statement)
    await session.commit()


async def get_hash_for_url(session: AsyncSession, url: str) -> Optional[str]:
    """Получает хеш-сумму контента для заданного URL."""
    statement = select(LegalDocument.content_hash).where(LegalDocument.source_url == url).limit(1)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def vector_search(session: AsyncSession, query_embedding: List[float], limit: int = 5) -> List[dict]:
    """
    Выполняет векторный поиск по базе данных.
    Находит `limit` самых похожих фрагментов.
    """
    # l2_distance (<->) находит Евклидово расстояние. Чем оно меньше, тем ближе векторы.
    statement = select(
        LegalDocument.text_chunk,
        LegalDocument.source_url
    ).order_by(
        LegalDocument.embedding.l2_distance(query_embedding)
    ).limit(limit)

    result = await session.execute(statement)
    found_docs = result.fetchall()

    # Форматируем результат в удобный список словарей
    return [
        {"text": doc.text_chunk, "url": doc.source_url} for doc in found_docs
    ]