import logging

def setLogger(loggerName, logFilePath):
    logger_AndroGen = logging.getLogger(loggerName)
    logger_AndroGen.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logFilePath)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger_AndroGen.addHandler(console_handler)
    logger_AndroGen.addHandler(file_handler)

    return logger_AndroGen
