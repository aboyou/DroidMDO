import logging
import os
import errno
import os.path as osp
import csv

verbose = 0


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
    node_attr = node_attr.rename(columns={0: 'id'})
    return node_attr

def node2function(s):
    first = s.find(' ')
    s = s[first + 1:]
    #print("[*] node2function: Updated string... :", s)
    if s.find('@ ') < 0:
        first = s.rfind('>')
        s = s[:first]
    else:
        first = s.find('[access_flags')
        s = s[:first - 1]
    return s


def debug(*args):
    if verbose:
        print(args)


def read_permission(path):
    permission = []
    with open(path) as f:
        line = f.readline()
        while line:
            line = line.strip('\n')
            permission.append(line)
            line = f.readline()
    return permission


def n_neighbor(node, graph, hop=1):
    import networkx as nx
    ego_graph = nx.ego_graph(graph, node, radius=hop, undirected=True)
    nodes = ego_graph.nodes
    return nodes


def get_label(node_id, G):
    return G.nodes[node_id]['label']

def get_from_csv_gml(filename):
    per_value = {}
    with open(filename, "r") as csvFile:
        reader = csv.reader(csvFile)
        for item in reader:
            if reader.line_num == 1:
                continue
            name = item[0]
            per = item[1]
            if name not in per_value.keys():
                per_value[name] = [per]
            else:
                per_value[name].append(per)
    return per_value



def getclass(functionname):
    index = functionname.find(';->')
    return functionname[len('<analysis.MethodAnalysis L'):index]


def getfunction(filename): 
    with open(filename) as f:
        line = f.readline().strip('\n')
        line = line.replace('# ', '')
        right = line.find(';->')
        classname = line[:right]
        right = line.find('[access_flags')
        if right < 0:
            function = line
        else:
            function = line[:right - 1]
        return classname, function


def get_nodeid_label(G, function):
    if type(function) == int:
        return function, G.nodes[function]['label']
    nodes = G.nodes
    for node in nodes:
        label = G.nodes[node]['label']
        if label.find(function) >= 0:
            return node, label
    return "", ""


def is_in_funcList(funcList, t):  # 节点是否再函数列表中
    for f in funcList:
        if t.find(f) >= 0:
            return True
    return False


def get_label(node_id, G):
    return G.nodes[node_id]['label']


def get_external(nodeid, G):
    return G.nodes[nodeid]['external']


def get_codesize(nodeid, G):
    return G.nodes[nodeid]['codesize']


def find_all_apk(path, end='.apk', layer=None):
    import glob
    if layer is not None:
        all_apk = glob.glob(f"{path}/{'/'.join(['*' for _ in range(layer)])}/*{end}")
    else:
        all_apk = glob.glob(os.path.join(path, '*%s' % end))

        # Get all dirs
        dirnames = [name for name in os.listdir(path)
                    if os.path.isdir(os.path.join(path, name))]
        for d in dirnames:
            add_apk = find_all_apk(os.path.join(path, d), end=end)
            all_apk += add_apk
    return all_apk
