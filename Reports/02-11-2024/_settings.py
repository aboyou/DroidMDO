import os
import logging.config

"""
    whether clean the workspace after work
        0 : Clean nothing
        1 : Clean res files
        2 : Clean smali files
        3 : Clean everything
"""
CLEAN_WORKSPACE = 2

"""
    Database use db 0 as default
"""

DB_HOST = 'localhost'
DB_PORT = 6379
DB_ID = 2
# if you don't have Password, delete DB_PSWD
DB_PSWD = ''

DB_FEATURE_CNT = 'feature_cnt'
DB_FEATURE_WEIGHT = 'feature_weight'
DB_UN_OB_PN = 'un_ob_pn'
DB_UN_OB_CNT = 'un_ob_cnt'


"""
    running_processes

    Use multi-processing could fully use the cores of cpu.
    Once I set QUEUE_TIME_OUT 5. After about two hours, three processes returns. So it should be little longer.
    I set it 30 yesterday and in two hours' processing, every process runs well.
"""
# RUNNING_PROCESS_NUMBER = 8
RUNNING_PROCESS_NUMBER = 1
QUEUE_TIME_OUT = 30


"""
IGNORE ZERO API FILES

    If there's no API in a class file, just ignore it.
    If there's no API in a package, just ignore it.
"""
IGNORE_ZERO_API_FILES = True

"""
Config Files
"""

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]
if not os.path.exists(SCRIPT_PATH + '\\Data'):
    os.mkdir(SCRIPT_PATH + '\\Data')
FILE_LOGGING = SCRIPT_PATH + '\\Data\\logging.conf'
FILE_RULE = SCRIPT_PATH + '\\Data\\tag_rules.csv'
LITE_DATASET_10 = SCRIPT_PATH + '\\Data\\lite_dataset_10.csv'

"""
    Logs
"""

logger = logging.getLogger('consoleLogger')
