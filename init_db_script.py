import sys
from pathlib import Path

# Динамически добавляем корень проекта
project_root = Path(__file__).parent.parent  # Поднимаемся на уровень выше legal_rag_bot
sys.path.append(str(project_root))

import asyncio
from legal_rag_bot.core.db import init_db

if __name__ == "__main__":
    asyncio.run(init_db())