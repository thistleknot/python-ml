import pathlib

#import finrl

import pandas as pd
import datetime
import os
import errno
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
#TRAINING_DATA_FILE = r"C:\Users\User\Documents\wiki\wiki\dev\python\python-ml\code\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020\data\ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = r"C:\Users\User\Documents\wiki\wiki\dev\python\python-ml\code\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020\data\dow_30_2009_2020.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = r"C:\Users\User\Documents\wiki\wiki\dev\python\python-ml\code\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020\trained_models\{now}"
try:
    os.makedirs(TRAINED_MODEL_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

TURBULENCE_DATA = r"C:\Users\User\Documents\wiki\wiki\dev\python\python-ml\code\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020\datadow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"


