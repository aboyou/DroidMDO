import torch
import androguard
from androguard.misc import AnalyzeAPK
from androguard.decompiler import decompiler
from androguard.misc import AnalyzeAPK
import networkx as nx
import matplotlib.pyplot as plt

import logging
import os
import time
import re
import sys

from utils import set_logger, makedirs
from utils import *
import utils
from collections import defaultdict

from androguard.core.analysis import auto
from androguard.decompiler.decompiler import DecompilerDAD

from Logger import setLogger
from AndroGenClass import AndroGen



startTime = time.time()
exp_base = './training/Experiment'
graph_base = './training/Graphs'
input_dir = 'C:\\Users\\ThisPC\\Desktop\\GNN_Env\\MsDroid\\APKs\\Test_DB'
apk_base = os.path.abspath(os.path.join(input_dir, '../'))
db_name = input_dir.split(apk_base)[-1].strip('\\')
output_dir = 'C:\\Users\\ThisPC\\Desktop\\GNN_Env\\MsDroid\\Outputs'
makedirs(output_dir)



hop = 2
tpl = False
exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}'
print(exp_dir)


def generate_feature(apk_base, db_name, output_dir, deepth):
    db_path = os.path.join(apk_base, db_name)
    print(db_path)
    cg_path = os.path.join(output_dir, db_name, "decompile")
    feature_path = os.path.join(output_dir, db_name, "result")
    settings = {
        "my": AndroGen(APKpath=db_path, CGPath=cg_path, FeaturePath=feature_path, deepth=deepth),
        "log": auto.DefaultAndroLog,
        "max_fetchers": 4,
    }
    aa = auto.AndroAuto(settings)
    aa.go()
    aa.dump()
    myandro = aa.settings["my"]
    call_graphs = myandro.get_call_graphs()
    return call_graphs


call_gphs = generate_feature(apk_base, db_name, output_dir, 2)
call_gphs.sort()
print("call graph", call_gphs)
endTime= time.time()

print("Process time: ", endTime-startTime, " seconds")

gmlBase = f'{output_dir}\\{db_name}'
print(gmlBase)
