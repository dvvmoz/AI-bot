# Файл: init_db_script.py (временный)

import sys
sys.path.append('/path/to/legal_rag_bot')
import asyncio
from legal_rag_bot.core.db import init_db

if __name__ == "__main__":
    asyncio.run(init_db())