import logging
from datetime import datetime
import os
import pandas as pd
from forest_cover.constant import get_current_time_stamp 
LOG_DIR="forest_logs"
import logging
from datetime import datetime
import os
os.chdir(r"E:\Ineuron\Project\forest cover classification project\forest-cover-classification-project")

import logging
from datetime import datetime
import os

LOG_DIR="forest_logs"

CURRENT_TIME_STAMP=  f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"


os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
level=logging.INFO
)