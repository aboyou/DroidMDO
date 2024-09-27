```python
import torch
from androguard.misc import AnalyzeAPK
from androguard.decompiler import decompiler
from androguard.misc import AnalyzeAPK
import networkx as nx
import matplotlib.pyplot as plt
```


```python
import logging
import os
import time
import re
```


```python
from utils import set_logger, makedirs
import utils
```


```python
from androguard.core.analysis import auto
from androguard.decompiler.decompiler import DecompilerDAD
```

# Define requirements


```python
logger_ = logging.getLogger("MyLogger")
logger = set_logger(logger_)

exp_base = './training/Experiment'
graph_base = './training/Graphs'
input_dir = 'C:\\Users\\ThisPC\\Desktop\\GNN_Env\\MsDroid\\APKs\\Dataset1_Benign'
apk_base = os.path.abspath(os.path.join(input_dir, '../'))
db_name = input_dir.split(apk_base)[-1].strip('\\')
output_dir = 'C:\\Users\\ThisPC\\Desktop\\GNN_Env\\MsDroid\\Outputs'
makedirs(output_dir)

hop = 2
tpl = False
exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}'
print(exp_dir)
```

    ./training/Graphs/Dataset1_Benign/HOP_2/TPL_False
    

# Loading APKs


```python
apk_path = "C:\\Users\\ThisPC\\Desktop\\MsDroid\\andapp.apk"
a, d, dx = AnalyzeAPK(apk_path)
```

# Generating graph and subgraphs
### Generated Behaviour Subgraphs


```python
cg = dx.get_call_graph()
```


```python
for node in cg.nodes():
    print(node)
    break
```

    Landroid/support/v4/BuildConfig;-><init>()V [access_flags=public constructor] @ 0x7cc98
    


```python
if not os.path.exists(f'{exp_dir}/dataset.pt'):
    print("It's not!")
    makedirs('Mappings')
    T1 = time.process_time()
    num_apk = 2
    #num_apk = generate_behaviour_subgraph()
    #time.sleep(10)
    T2 = time.process_time()
    print(f'Generate Behavior Subgraphs for {num_apk} APKs: {T2-T1}')
    testonly = True if num_apk==1 else False
```

    It's not!
    Generate Behavior Subgraphs for 2 APKs: 0.0
    


```python
# Create a new logger
logger_AndroGen = logging.getLogger("AndroGen")

# Set the logging level (DEBUG, INFO, WARNING, ERROR)
logger_AndroGen.setLevel(logging.DEBUG)

# Create handlers (console and file handlers as examples)
# Example: Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Example: File handler to log errors to a file
file_handler = logging.FileHandler('AndroGen.log')
file_handler.setLevel(logging.ERROR)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger_AndroGen.addHandler(console_handler)
logger_AndroGen.addHandler(file_handler)

# Assign the logger to _settings
class logger_Settings:
    def __init__(self):
        self.logger = logger_AndroGen

# Create settings instance
_settings_log = logger_Settings()
```


```python
# Example of using _settings.logger
_settings_log.logger.error("This is an error message3")
```

    2024-09-27 03:48:06,554 - AndroGen - ERROR - This is an error message3
    


```python
class AndroGen(auto.DirectoryAndroAnalysis):
    def __init__(self, APKpath, CGPath, FeaturePath, deepth):
        self.replacemap = {'Landroid/os/AsyncTask;':['onPreExecute', 'doInBackground'],
                          'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']}
        super(AndroGen, self).__init__(APKpath)
        self.APKpath = APKpath
        self.has_crashed = False          # It shows any failure in analysis process
        self.CGPath = CGPath
        self.FeaturePath = FeaturePath
        self.smali_opcode = self.get_smaliOpcode("smaliOpcode.txt")   # A list of Smali opcodes that the analysis will track.
        self.permission = []
        with open("head.txt") as f:
            self.permission = eval(f.read())
        self.cppermission = self.get_permission()   # CP is abstract of Content Provider
        self.call_graphs = []
        self.count = 0
        self.deepth = deepth

    def get_smaliOpcode(self, FileName):
        opcode = list()
        with open(FileName, 'r') as fileObject:
            lines = fileObject.readlines()
        for line in lines:
            opcode.append(line.rstrip('\n'))
        return opcode

    def get_permission(self):
        filename = "all_cp.txt"
        permission = {}
        with open(filename) as f:
            content = f.readline().strip('\n')
            while content:
                cons = content.split(' ')
                if cons[0] not in permission:
                    permission[cons[0]] = set()
                permission[cons[0]].add((cons[1], 'Permission:' + cons[2]))
                content = f.readline().strip('\n')
        return permission

    def analysis_app(self, log, apkobj, dexobj, analysisobj):
        dexobj.set_decompiler(DecompilerDAD(dexobj, analysisobj))
        apk_filename = log.filename
        CGPath = apk_filename.replace(self.APKpath, self.CGPath)[:-4]
        CGfilename = os.path.join(CGPath, "call.gml")
        if not os.path.exists(CGPath):
            try:
                os.makedirs(CGPath)
            except Exception:
                pass
        opcodeFilename = apk_filename.replace(self.APKpath, self.FeaturePath + "/opcode").replace(".apk", ".csv")
        opcodePath = opcodeFilename.replace(".apk", ".csv")
        if not os.path.exists(opcodePath):
            try:
                makedirs(opcodePath)
            except Exception:
                pass
        permissionFilename = apk_filename.replace(self.APKpath, self.FeaturePath + "/permission").replace(".apk", ".csv")
        permissionPath = permissionFilename[:permissionFilename.rfind('/')]
        if not os.path.exists(permissionPath):
            try:
                os.makedirs(opcodePath)
            except Exception:
                pass
        tplFilename = apk_filename.replace(self.APKpath, self.FeaturePath + "/tpl").replace(".apk", ".csv")
        tplPath = tplFilename[:tplFilename.rfine('/')]
        if not os.path.exists(tplPath):
            try:
                os.makedirs(tplPath)
            except Exception:
                pass
        if not os.path.exists(CGfilename):
            G = analysisobj.get_call_graph()
            nx.write_gml(G, CGfilename, stringizer=str)
        self.call_graphs.append(CGfilename)
        G = nx.read_gml(CGfilename, label='id')
        if os.path.exists(tplFilename):
            return
        opcodeFilename = utils.create_csv(self.smali_opcode, opcodeFilename)
        method2nodeMap = self.getMethod2NodeMap(G)
        if method2nodeMap == {}:
            _settings_log.logger.error("%s has call graph error"%log.filename)
            print("%s has call graph error"%log.filename)
            return 
        class_functions = defaultdict(list)
        super_dic = {}
        implement_dic = {}

        for classes in analysis.get_classes():
            class_name = str(classes.get_class().get_name())
            if (classes.extends != "Ljava/lang/Object;"):
                super_dic[class_name] = str(classes.extends)                # Store extends of each class
                if str(classes.extends) in self.replacemap:
                    implement_dic[class_name] = str(classes.extends)
            if classes.implements:                                          # Store interfaces of classes
                for imp in classes.implements:
                    if str(imp) in self.replacemap:
                        implement_dic[class_name] = str(imp)
            for method in classes.get_methods():
                if method.is_external():
                    continue
                m = method.get_method()
                class_functions[class_name].append(str(m.full_name))        # Store methods of a class as functions
                c = defaultdict(int)
                flag = False
                for ins in m.get_instructions():  # count
                    flag = True  # exist instructions
                    c[ins.get_name()] += 1
                opcode = {}
                for p in self.smali_opcode:
                    opcode[p] = 0
                for op in c:
                    if op in self.smali_opcode:
                        opcode[op] += c[op]
                if flag:
                    try:
                        utils.write_csv(opcode, opcodeFile, method2nodeMap[str(m.full_name)][0])
                    except Exception:
                        print("apk: %s, method: %s not exists"%(log.filename, str(m.full_name)))
        opcodeFile.close()
        
        pass


    
    def getMethod2NodeMap(self, G):
        method2nodemap = {}
        try:
            node_attr = utils.df_from_G(G)
            labels = node_attr.label
            ids = node_attr.id
        except Exception:
            return method2nodemap
        i = 0
        pattern = re.compile(r'&#(.+?);')
        while i < len(ids):
            nodeid = ids.get(i)
            label = labels.get(i)
            function = utils.node2function(label)
            rt = pattern.findall(function)
            for r in rt:
                function.replace("&#%s; "%r, chr(int(r)))
            method = function.replace(";->", "; ").replace("(", " (")
            method2nodeMap.update({method: (nodeid, function)})
            i = i + 1
        return method2nodeMap
        
            
```


```python
pd_cg = utils.df_from_G(cg)
ids = pd_cg.id
nodeid = ids.get(5)
pointer = pd_cg.label
print(pointer.loc[0])
print(utils.node2function(str(pointer.loc[0])))
```

    Landroid/support/v4/BuildConfig;-><init>()V [access_flags=public constructor] @ 0x7cc98
    [access_flags=public constructor] @ 0x7cc9
    


```python
t = AndroGen("APKpath", "CGPath", "FeaturePath", 1)
```


```python
def generate_feature(apk_base, db_name, output_dir, deepth):
    db_path = os.path.join(apk_base, db_name)
    print(db_path)
    cg_path = os.path.join(output_dir, db_name, "decompile")
    feature_path = os.path.join(output_dir, db_name, "result")
    settings = {
        "my": AndroGen(APKpath=db_path, CGPath=cg_path, FeaturePath=feature_path, deepth=deepth),
        "log": auto.DefaultAndroLog,
        "max_fetchers": 2,
    }
    aa = auto.AndroAuto(settings)
    aa.go()
    aa.dump()
    myandro = aa.settings["my"]
    call_graphs = myandro.get_call_graphs()
    return call_graphs
```


```python
def generate_behavior_subgraph(apk_base, db_name, output_dir, deepth, label, hop=2, tpl=True, training=False, api_map=False):
    call_graphs = generate_feature(apk_base, db_name, output_dir, deepth)
    
```


```python

```
