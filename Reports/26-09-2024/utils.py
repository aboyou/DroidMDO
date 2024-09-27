import logging
import os
import errno
import os.path as osp
import csv


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


def create_csv(smali_opcode, path):
    f = open(path, 'w+', newline='')
    csv_write = csv.writer(f)
    csv_head = ['id'] + smali_opcode
    csv_write.writerow(csv_head)
    return f

def write_csv(opcode, f, id): 
    csv_write = csv.writer(f)
    data_row = [id]
    for op in opcode.keys():
        data_row.append(opcode[op])
    csv_write.writerow(data_row)

def df_from_G(G):
    import pandas as pd
    df = pd.DataFrame(G.nodes(data=True))
    try:
        attr = df[1].apply(pd.Series)
    except KeyError:
        return False
    node_attr = pd.concat([df[0], attr], axis=1)
    node_attr = node_attr.rename(columns={0: 'label'})
    node_attr.insert(0, 'id', range(0, len(node_attr)))
    return node_attr

def node2function(s):
    first = s.find(' ')
    s = s[first + 1:]
    if s.find('@ ') < 0:
        first = s.rfind('>')
        s = s[:first]
    else:
        first = s.find('[access_flags')
        s = s[:first - 1]
    return s


