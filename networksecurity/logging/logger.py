import logging
import os
from datetime import datetime

# 1️⃣ Create logs directory (only the folder)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 2️⃣ Define log file name and full path
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 3️⃣ Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
)


