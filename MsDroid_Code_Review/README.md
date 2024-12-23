
# **MsDroid Code Review**
برای مشاهده گیت‌هاب MsDroid به آدرس زیر می‌‌توان رجوع کرد:
[MsDroid (TDSC'22)](https://github.com/E0HYL/MsDroid)

# Preprocess step
در ابتدا به بررسی مرحلۀ preprocess می‌پردازیم. در گیت‌هاب MsDroid آمده است که برای آموزش مدل بر اساس دیتاست خود باید برنامۀ پایتونی `train.py` را اجرا نمود. همچنین اشاره شده است که به صورت زیر باید این فایل را به اجرا گذاشت:
```Bash
python3 train.py -i <input directory (APK dataset)>
```
البته که این اسکریپت پایتونی، آرگومان‌های دیگری به عنوان ورودی اخذ می‌کند که در ادامه به آن‌ها پرداخته می‌شود. پس در ابتدا به بررسی مسیر اجرای یک این اسکریپت برای یک دایرکتوری از APKها می‌پردازیم. در گام اول یک دایرکتوری متشکل از 3 فایل APK از دیتاست Androzoo را در نظر می‌گیریم. آدرس این دایرکتوری در سیستم خود برابر است با:
```Bash
/home/user/MsDroid2/APKs/Test_DB
```
## فعلا کمی از train.py
در این اسکریپت در ابتدا دو آدرس تنظیم می‌شوند که در ادامه بسیار کاربردی هستند:
```Bash
exp_base = './training/Experiments'
graph_base = f'./training/Graphs'
```
در ادامه یکی از آگومان‌های ورودی این اسکریپت output است که به صورت پیش‌فرض دایرکتوری Output از همان دایرکتوری است که اسکریپت در آن قرار دارد. برای مثال در سیستم ما به این صورت است:
```Bash
/home/user/MsDroid2/MsDroid-main/src/Output
```
نکته بعدی آرگومان label است که به صورت پیش‌فرض برابر با 1 در نظر گرفته شده است و منظور آن است که به صورت پیش‌فرض malware در نظر گرفته می‌شود. (این label برای دیتاست APK در نظر گرفته می‌شود)
یکی از دایرکتوری‌هایی که بسیار مهم است، `apk_base` است و این دایرکتوری فولدری را که فولدرهای (دیتاست‌های) شامل APKها را شامل می‌شود، معرفی می‌کند. یعنی در مثال سیستم ما این دایرکتوری برابر است با:
```Bash
/home/user/MsDroid2/APKs
```
همچنین نام دیتاست هم در داخل `db_name` ذخیره می‌گردد. در مثال ما `Test_DB` نام دیتاست ما می‌باشد.
### Experiment
برای هر آزمایش یا همان experiment، یک فایل و دایرکتوری ایجاد می‌شود. آدرس این دایرکتوری به صورت زیر است:
```python
exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}'
```
همانطور که مشاهده می‌گردد، دایرکتوری `training` که مسئول ذخیره اطلاعات آموزش شبکه عصبی است، اطلاعات گرا‌های بوجود آمده را ذخیره می‌نماید. 
در `train.py` مشاهده می‌گردد که یک شرط وجود دارد و به صورت زیر است:
```python
if not os.path.exists(f'{exp_dir}/dataset.pt'):
        makedirs('Mappings')
        import time
        T1 = time.process_time()    
        '''
        ./training/Graphs/<db_name>/processed/data_<apk_id>_<subgraph_id>.pt
        '''
        num_apk = generate_behavior_subgraph(apk_base, db_name, output_dir, args.deepth, label, hop=hop, tpl=tpl, training=True, api_map=True)
        T2 = time.process_time()
        print(f'Generate Behavior Subgraphs for {num_apk} APKs: {T2-T1}')
        testonly = True if num_apk==1 else False
```
در کد بالا، ابتدا بررسی می‌کند که آیا `dataset.pt` وجود دارد یا خیر! اگر وجود نداشت، دایرکتوری `Mappings` را درون دایرکتوری اصلی می‌سازد. در ادامه با استفاده از تابع `generate_behavior_subgraph()`، زیرگراف‌های رفتاری را می‌سازد و در دایرکتوری زیر ذخیره می‌سازد:
```bash
./training/Graphs/<db_name>/processed/
```
زیرگراف‌ها در فرمت `pt.*` ذخیره می‌شوند که برای آن است که در ساختارهای PyTorch ای استفاده شوند. نام این فایل‌ها به صورت زیر است:
```bash
data_<apk_id>_<subgraph_id>.pt
```
این نام‌گذاری نشان می‌دهد که هر فایل APK یک apk_id دریافت می‌کند و هر زیرگراف از هر فایل APK نیز یک ID مخصوص به خود دریافت می‌کند.
> نکته‌ای که وجود دارد آن است که وقتی هم‌اکنون اسکریپت را اجرا می‌کنیم، تابع `generate_behavior_subgraph()` را به طور کامل درست اجرا نمی‌کند و در میانۀ آن خطا دارد!

در `train.py` توابع دیگری نیز وجود دارد و همۀ توابع باید بررسی شوند. در موقع آن‌ها نام این توابع به میان خواهد آمد و چگونگی عملکرد آن‌ها بررسی خواهد شد! به همین دلیل هم‌اکنون به بررسی `generate_behavior_subgraph()` پرداخته می‌شود.
#زیرگراف #تولید #subgraph
## درود بر `generate_behavior_subgraph`
این تابع در فایل `main.py` تعریف شده است. کد این تابع به صورت زیر است:
```python
def generate_behavior_subgraph(apk_base, db_name, output_dir, deepth, label, hop=2, tpl=True, training=False, api_map=False):
    '''
    <output_dir>/<db_name>/decompile/<apk_name>/call.gml
    <output_dir>/<db_name>/result/<permission | opcode | tpl>/<apk_name>.csv
    '''
    call_graphs = generate_feature(apk_base, db_name, output_dir, deepth)   # `.gml`
    call_graphs.sort()
    print("call graph", call_graphs)
    '''
    <output_dir>/<db_name>/processed/data_<apk_id>_<subgraph_id>.pt
    '''
    gml_base = f'{output_dir}/{db_name}'
    generate_graph(call_graphs, output_dir, gml_base, db_name, label, hop, tpl, training, api_map)
    return call_graphs
```
این تابع وظیفه آن را دارد که فایل‌ها call graph را ایجاد کند. همچنین نتیجه استخراج ویژگی‌های یک APK را در دایرکتوری‌های زیر ذخیره می‌کند:
```bash
<output_dir>/<db_name>/result/tpl/<apk_name>.csv
<output_dir>/<db_name>/result/opcode/<apk_name>.csv
<output_dir>/<db_name>/result/permission/<apk_name>.csv
```
تابعی که وظیفه ایجاد فایل‌های بالا را بر عهده دارد، `generate_features()` است. 
ما قبل‌تر اشاره کردیم که زیرگراف‌ها در تابع `generate_behavior_subgraph()` تولید می‌گردند. در واقع تابعی که وظیفه اصلی ایجاد این زیرگراف‌ها را برعهده دارد، `generate_graph()` است. این تابع آرگومان‌هایی مانند call graphها را به عنوان ورودی می‌گیرد و خروجی آن زیرگراف‌های ذخیره شده در دایرکتوری مخصوص به خود است. 
سوال اصلی اینجاست که چگونه featureها بدست می‌آیند؟ به همین منظور ابتدا به قسمت نظری در مقاله می‌پردازیم. برای دریافت مقاله می‌توان لینک روبرو را دنبال کرد: [لینک مقاله](https://ieeexplore.ieee.org/document/9762803)
## بدست آوردن ویژگی‌ها (!!features!!)
### نظری

معانی API یا همان API semantics از دو راه جمع می‌شوند:
1. با متمایز کردن APIهای مرتبط با permissionها (مجوزهای حساس) که با $V_{per}$ نمایش داده می‌شوند و منظور مجموعه‌ای است که شامل گره‌های حساس است. همچنین داریم که:
$$
V_{per} \subset V
$$
در MsDroid برای بدست آوردن این مجموعه از دو نگاشت API-permission به نام‌های PSCout و Axplorer استفاده می‌گردد.
هر چند اپلیکیشن‌های اندرویدی از عملیات‌های CRUD برای content providerها استفاده می‌کنند که معمولا به مجوزهای حساس دسترسی پیدا می‌کنند. برای مثال `content://mms` که نیاز به مجوز `READ_SMS` دارد. عملیات‌های CRUD توسط متدهایی از اینترفیس `ContentRsolver` هندل می‌شوند.
عملا MsDroid این متدها را هندل می‌کند و نگاشت API-permission را به صورت $M_{per}$ تکمیل می‌کند.

2. برای شناسایی APIهایی که به ماژول‌های کاربردی ایزوله مرتبط هستند از LibRadar استفاده می‌گردد تا TPL یا همان Third-Party Libraries درون اپلیکیشن‌ها شناسایی شود.


### عملی
این تابع در فایل `main.py_` در دایرکتوری زیر وجود دارد:
```bash
/home/user/MsDroid2/MsDroid-main/src/feature/
```
در ابتدا به کد این تابع نگاهی می‌اندازیم:
```python
import os
from androguard.core.analysis import auto
from .Andro.Andro import AndroGen
def generate_feature(apk_base, db_name, output_dir, deepth):
    '''
    save files:
    <output_dir>/<db_name>/decompile/<apk_name>/call.gml
    <output_dir>/<db_name>/result/<permission | opcode | tpl>/<apk_name>.csv
    '''
    # return all complete paths for `call.gml`
    db_path = os.path.join(apk_base, db_name)
    print(db_path)
    cg_path = os.path.join(output_dir, db_name, "decompile")
    feature_path = os.path.join(output_dir, db_name, "result")
    settings = {
        # The directory `some/directory` should contain some APK files
        "my": AndroGen(APKpath=db_path, CGPath=cg_path, FeaturePath=feature_path, deepth=deepth),  # apkfile
        # Use the default Logger
        "log": auto.DefaultAndroLog,
        # Use maximum of 2 threads
        "max_fetcher": 2,
    }
    aa = auto.AndroAuto(settings)
    aa.go()
    aa.dump()
    myandro = aa.settings["my"]
    call_graphs = myandro.get_call_graphs()
    print("generate feature finished")
    return call_graphs
```
در این تابع، دایرکتوری `db_path` بوجود می‌آید که عبارت است از جمع `apk_base` و `db_name`!  در این مثال ما، `db_path` برابر است با:
```bash
/home/user/MsDroid2/APKs/Test_DB
```
در ادامه آدرس دایرکتوری `cg_path` تولید می‌گردد که برابر است با:
```bash
<output_dir>/<db_name>/decompile/
```
در این دایرکتوری، فایل call graph ذخیره می‌گردد.
پس از آن، آدرس دایرکتوری `feature_path` تولید می‌گردد که برابر است با:
```bash
<output_dir>/<db_name>/result/
```
برای آنالیز فایل‌های APK از کتابخانه پایتونی Androguard استفاده می‌گردد. کلاس `AndroAuto` اینچنین است که بر اساس تنظیمات آن (settings در اینجا) تحلیلی انجام می‌دهد. در واقع کلاس `AndroGen` وظیفه اصلی این تحلیل را برعهده دارد. در انتها نیز call graphها به عنوان خروجی، return می‌گردند.
خروجی این تابع که همان call graphها هستند را یک بار نمایش می‌دهیم:
```bash
 ['/home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/decompile/1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720/call.gml', '/home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/decompile/1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3/call.gml', '/home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/decompile/2b28128271d07a1e31f3a4eb8318886fba9becd9f1125833aaea5eb89d85ee47/call.gml']
```
حال به بررسی کلاس `AndroGen` می‌پردازیم.

### **`AndroGen`**
این کلاس در فایل پایتونی `Andro.py` در دایرکتوری زیر قرار دارد:
```bash
/home/user/MsDroid2/MsDroid-main/src/feature/Andro/
```
کد این کلاس عبارت است از:
```python
import csv
import os
import sys
from collections import defaultdict
import networkx as nx
from androguard.core.analysis import auto
from androguard.decompiler.decompiler import DecompilerDAD
from . import _settings
from .permission import Permission
from .tpl import Tpl
from feature.Utils import utils
import re

# Functionality: opcode generation, call graph generation, mapping between generation and parenting

class AndroGen(auto.DirectoryAndroAnalysis):
    def __init__(self, APKpath, CGPath, FeaturePath, deepth):
        self.replacemap = {'Landroid/os/AsyncTask;': ['onPreExecute', 'doInBackground'],
                           'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']}
        super(AndroGen, self).__init__(APKpath)
        self.APKPath = APKpath
        self.has_crashed = False
        self.CGPath = CGPath
        self.FeaturePath = FeaturePath
        self.smali_opcode = self.get_smaliOpcode(_settings.smaliOpcodeFilename)
        self.permission = []
        with open(_settings.headerfile) as f:
            self.permission = eval(f.read())
        self.cppermission = self.get_permission()
        self.call_graphs = []
        self.count = 0
        self.deepth = deepth

    def get_smaliOpcode(self, filename):  # read all smali opcode list
        opcode = list()
        with open(filename, 'r') as fileObject:
            lines = fileObject.readlines()
        for line in lines:
            opcode.append(line.rstrip('\n'))
        return opcode

    def get_permission(self):
        filename = _settings.cppermissiontxt
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
        CGpath = apk_filename.replace(self.APKPath, self.CGPath)[:-4]
        CGfilename = os.path.join(CGpath, "call.gml")
        if not os.path.exists(CGpath):
            try:
                os.makedirs(CGpath)
            except Exception:
                pass
        opcodeFilename = apk_filename.replace(self.APKPath, self.FeaturePath + "/opcode").replace(".apk", ".csv")
        opcodePath = opcodeFilename[:opcodeFilename.rfind('/')]
        if not os.path.exists(opcodePath):
            try:
                os.makedirs(opcodePath)
            except Exception:
                pass
        permissionFilename = apk_filename.replace(self.APKPath, self.FeaturePath + "/permission").replace(".apk",".csv")
        permissionPath = permissionFilename[:permissionFilename.rfind('/')]
        if not os.path.exists(permissionPath):
            try:
                os.makedirs(permissionPath)
            except Exception:
                pass
        tplFilename = apk_filename.replace(self.APKPath, self.FeaturePath + "/tpl").replace(".apk", ".csv")
        tplPath = tplFilename[:tplFilename.rfind('/')]
        if not os.path.exists(tplPath):
            try:
                os.makedirs(tplPath)
            except Exception:
                pass
        if not os.path.exists(CGfilename):
            G = analysisobj.get_call_graph()  # call graph
            nx.write_gml(G, CGfilename, stringizer=str)  # save the call graph
        self.call_graphs.append(CGfilename)
        G = nx.read_gml(CGfilename, label='id')
        if os.path.exists(tplFilename):
            return
        opcodeFile = utils.create_csv(self.smali_opcode, opcodeFilename)
        method2nodeMap = self.getMethod2NodeMap(G)
        if method2nodeMap == {}:
            _settings.logger.error("%s has call graph error"%log.filename)
            print("%s has call graph error"%log.filename)
            return
        class_functions = defaultdict(list)  # mappings of class and its functions
        super_dic = {}  # mappings of class and its superclass(for class replacement)
        implement_dic = {}

        for classes in analysisobj.get_classes():  # all class
            class_name = str(classes.get_class().get_name())
            if classes.extends != "Ljava/lang/Object;":
                super_dic[class_name] = str(classes.extends)
                if str(classes.extends) in self.replacemap:
                    implement_dic[class_name] = str(classes.extends)
            if classes.implements:
                for imp in classes.implements:
                    if str(imp) in self.replacemap:
                        implement_dic[class_name] = str(imp)
            for method in classes.get_methods():
                if method.is_external():
                    continue
                m = method.get_method()
                class_functions[class_name].append(str(m.full_name))
                c = defaultdict(int)
                flag = False
                for ins in m.get_instructions():  # count
                    flag = True  # exist instructions
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
        opcodeFile.close(
        cpermission = Permission(G=G, path=permissionFilename, class_functions=class_functions, super_dic=super_dic,
                                 implement_dic=implement_dic, dexobj=dexobj, permission=self.permission, cppermission=self.cppermission, method2nodeMap=method2nodeMap)
        cpermission.generate()
        class2init = cpermission.getClass2init()
        sensitiveapimap = cpermission.getsensitive_api()
        ctpl = Tpl(log.filename, G, tplFilename, sensitiveapimap, self.permission, class2init, self.deepth)
        ctpl.generate()


    def getMethod2NodeMap(self, G):
        method2nodeMap = {}
        try:
            node_attr = utils.df_from_G(G)
            labels = node_attr.label
            ids = node_attr.id
        except Exception:
            return method2nodeMap
        i = 0
        pattern = re.compile(r'&#(.+?);')
        while i < len(ids):
            nodeid = ids.get(i)
            label = labels.get(i)
            function = utils.node2function(label)
            rt = pattern.findall(function)
            for r in rt:
                function.replace("&#%s;"%r, chr(int(r)))
            method = function.replace(";->", "; ").replace("(", " (")
            method2nodeMap.update({method: (nodeid, function)})
            i = i + 1
        return method2nodeMap

  

    def get_call_graphs(self):
        return self.call_graphs
  

    def finish(self, log):
        # This method can be used to save information in `log`
        # finish is called regardless of a crash, so maybe store the
        # information somewhere
        if self.has_crashed:
            _settings.logger.debug("Analysis of {} has finished with Errors".format(log))
            print("Analysis of %s has finished with Errors, %d"%(log.filename, self.count))
        else:
            _settings.logger.info("Analysis of {} has finished!".format(log))
            print("Analysis of %s has finished!, %d"%(log.filename, self.count))
        self.count = self.count + 1

    def crash(self, log, why):
        # If some error happens during the analysis, this method will be
        # called
        self.has_crashed = True
        _settings.logger.debug("Error during analysis of {}: {}".format(log, why), file=sys.stderr)
```