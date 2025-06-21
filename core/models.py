# ======================================================================
# Файл: legal_rag_bot/core/models.py
# СКОПИРУЙТЕ ЭТОТ КОД В ВАШ ФАЙЛ models.py
# ======================================================================
import sqlalchemy
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TEXT, TIMESTAMP

# Размерность вектора. Зависит от модели OpenAI, которую мы используем.
# У text-embedding-3-small размерность 1536.
VECTOR_DIMENSION = 1536

# Создаем базовый класс для наших моделей
Base = declarative_base()

class LegalDocument(Base):
    """
    Модель таблицы для хранения фрагментов (чанков) юридических документов.
    """
    __tablename__ = 'legal_documents'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    text_chunk = sqlalchemy.Column(TEXT, nullable=False, comment="Сам текст фрагмента (чанк)")
    source_url = sqlalchemy.Column(TEXT, nullable=False, comment="URL страницы, откуда взят текст")
    content_hash = sqlalchemy.Column(TEXT, nullable=False, comment="Хеш-сумма ВСЕЙ статьи (для детекции изменений)")
    # Устанавливаем часовой пояс для корректной работы
    last_updated = sqlalchemy.Column(TIMESTAMP(timezone=True), nullable=False, comment="Дата и время последнего обновления")
    # Определяем поле для хранения векторов
    embedding = sqlalchemy.Column(Vector(VECTOR_DIMENSION), nullable=False, comment="Векторное представление чанка")

    def __repr__(self):
        return f"<LegalDocument(id={self.id}, url='{self.source_url}')>"
