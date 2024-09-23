import logging
import os
import errno
import os.path as osp

formatter = logging.Formatter('%(asctime)s - %(filename)s@%(lineno)d - %(levelname)s: %(message)s')

def set_logger(logger, base_level=logging.DEBUG, ch_level=logging.DEBUG, fh_name=None, fh_level=None, formatter=formatter):
    # create logger with 'spam_application'  
    logger.setLevel(base_level)  
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(ch_level)
    # add formatter to the handlers
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch) 
    if fh_name is not None:
        fh_level = base_level if fh_level is None else fh_level
        add_fh(logger, fh_name, fh_level, formatter)
    return logger

def add_fh(logger, fh_name, fh_level=logging.INFO, formatter=formatter):
    fh = logging.FileHandler(fh_name)
    fh.setLevel(fh_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
