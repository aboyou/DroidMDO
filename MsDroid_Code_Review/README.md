
# **MsDroid Code Review - Subgraph generation**
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
همانطور که مشاهده می‌گردد، دایرکتوری `training` که مسئول ذخیره اطلاعات آموزش شبکه عصبی است، اطلاعات گره‌های بوجود آمده را ذخیره می‌نماید. 
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
با بررسی استخراج ویژگی‌ها (features) می‌توانیم execution flow را به صورت تصویر زیر در نظر بگیریم:
![image_execution_flow](images/Execution_Flow_Feature_Generation.png)


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
هدف آن است که بفهمیم چگونه این ویژگی‌ها (features) تولید می‌شوند. حال یا درون کلاس `AndroGen` تولید خواهند شد یا این که از طریق تابع `generate_graph()` بدست خواهند آمد.

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

قبل از پرداختن به جزئیات قسمت‌های مختلف این کلاس، مرور کلی بر عملکرد این کلاس خواهیم داشت.

- **اصلی‌ترین بخش تجزیه و تحلیل: `Analysis_app (log, apkobj, dexobj, analysisobj)

	 دایرکتوری‌هایی را برای ذخیره گراف‌های فراخوانی، کدهای عملیاتی(opcodes)، مجوزها و الگوها ایجاد می‌کند.
	 گراف فراخوانی را با استفاده از شی تجزیه و تحلیل (analysisobj) استخراج می‌کند و آن را با فرمت gml. با استفاده از networkx ذخیره می‌کند.
	 کدهای عملیاتی را برای هر روش در APK استخراج می‌کند و آن‌ها را به گره‌های مربوطه در گراف فراخوانی نگاشت می‌کند.
	 کلاس Permission را برای ایجاد نقشه‌های (نگاشت‌های) مجوز و کلاس Tpl را برای استخراج نگاشت‌های API حساس فراخوانی می‌کند.

- **تابع getMethod2NodeMap(G)** 
	یک نگاشت بین متدها و گره‌های مربوط به آن‌ها در گراف فراخوانی ایجاد می‌کند.

### در ابتدا init
باید بر این بخش تفصیلی ارائه گردد. در ابتدا باید بر کد این بخش مروری صورت گیرد که معادل است با:
```python
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
```
در ابتدا یک جایگزینی صورت می‌پذیرد که برابر است با:
```python
self.replacemap = {'Landroid/os/AsyncTask;': ['onPreExecute', 'doInBackground'],
                           'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']}
```
اما خب این به چه معنا است؟! 

در واقع این نگاشت جایگزینی (replacement) برای هندل کردن کامپوننت‌های اندرویدی استفاده می‌شود که از الگوی خاصی در طراحی پیروی می‌کنند. در واقع کامپوننت‌های `AsyncTask`، `Handler` و `Runnable` متدهایی دارند که قابلیت اجرای asynchronously را دارا هستند و یا این که override می‌شوند و تسک خاصی را انجام می‌دهند. این متدها برای فهم عملکرد اپلیکیشن‌های اندرویدی حیاتی هستند، به خصوص در آنالیز عملیات‌های حیاتی API callها! 
یک مثال می‌تواند راحت‌تر موضوع را بیان کند:
```java
public class MyAsyncTask extends AsyncTask<Void, Void, Void> {
    @Override
    protected void onPreExecute() {
        // Pre-task setup
    }

    @Override
    protected Void doInBackground(Void... params) {
        // Sensitive API call
        ContentResolver.query(Uri.parse("content://sms"), null, null, null, null);
        return null;
    }

    @Override
    protected void onPostExecute(Void result) {
        // Post-task actions
    }
}
```
در این مثال، اپلیکیشن اندرویدی از `AsyncTask` استفاده می‌کند تا یک فعالیت حساس (خواندن `content://sms`) را به صورت غیرهمزمان در background انجام دهد. 
بررسی می‌کنیم اگر بدون replacement آنالیز صورت بگیرد چه می‌شود!
- شاید call graph نتواند `doInBackground` را به عنوان یک گره حساس تشخیص دهد.
- عملا API call های حساس را از دست بدهیم. زیرا به دلیل وراثت دچار عمق شده‌اند!
اگر این نگاشت لحاظ گردد خواهیم داشت:
```plaintext
node0 [label="MyAsyncTask->doInBackground"]
edge [source=0 target=ContentResolver.query]
```

#### **و برمی‌گردیم به تابع init** 
در یک خط از این قسمت از `init()` داریم که:
```python
self.smali_opcode = self.get_smaliOpcode(_settings.smaliOpcodeFilename)
```
عملا یک فایل config را تنظیم می‌کند. خروجی این خط از کد را بررسی می‌کنیم.
```bash
>>> ['nop', 'move', 'move/from16', 'move/16', 'move-wide', 'move-wide/from16', 'move-wide/16', 'move-object', 'move-object/from16', 'move-object/16', 'move-result', 'move-result-wide', 'move-result-object', 'move-exception', 'return-void', 'return', 'return-wide', 'return-object', 'const/4', 'const/16', 'const', 'const/high16', 'const-wide/16', 'const-wide/32', 'const-wide', 'const-wide/high16', 'const-string', 'const-string/jumbo', 'const-class', 'monitor-enter', 'monitor-exit', 'check-cast', 'instance-of', 'array-length', 'new-instance', 'new-array', 'filled-new-array', 'filled-new-array/range', 'filled-array-data', 'throw', 'goto', 'goto/16', 'goto/32', 'packed-switch', 'sparse-switch', 'cmpl-float', 'cmpg-float', 'cmpl-double', 'cmpg-double', 'cmp-long', 'if-eq', 'if-ne', 'if-lt', 'if-ge', 'if-gt', 'if-le', 'if-eqz', 'if-nez', 'if-ltz', 'if-gez', 'if-gtz', 'if-lez', 'aget', 'aget-wide', 'aget-object', 'aget-boolean', 'aget-byte', 'aget-char', 'aget-short', 'aput', 'aput-wide', 'aput-object', 'aput-boolean', 'aput-byte', 'aput-char', 'aput-short', 'iget', 'iget-wide', 'iget-object', 'iget-boolean', 'iget-byte', 'iget-char', 'iget-short', 'iput', 'iput-wide', 'iput-object', 'iput-boolean', 'iput-byte', 'iput-char', 'iput-short', 'sget', 'sget-wide', 'sget-object', 'sget-boolean', 'sget-byte', 'sget-char', 'sget-short', 'sput', 'sput-wide', 'sput-object', 'sput-boolean', 'sput-byte', 'sput-char', 'sput-short', 'invoke-virtual', 'invoke-super', 'invoke-direct', 'invoke-static', 'invoke-interface', 'invoke-virtual/range', 'invoke-super/range', 'invoke-direct/range', 'invoke-static/range', 'invoke-interface/range', 'neg-int', 'not-int', 'neg-long', 'not-long', 'neg-float', 'neg-double', 'int-to-long', 'int-to-float', 'int-to-double', 'long-to-int', 'long-to-float', 'long-to-double', 'float-to-int', 'float-to-long', 'float-to-double', 'double-to-int', 'double-to-long', 'double-to-float', 'int-to-byte', 'int-to-char', 'int-to-short', 'add-int', 'sub-int', 'mul-int', 'div-int', 'rem-int', 'and-int', 'or-int', 'xor-int', 'shl-int', 'shr-int', 'ushr-int', 'add-long', 'sub-long', 'mul-long', 'div-long', 'rem-long', 'and-long', 'or-long', 'xor-long', 'shl-long', 'shr-long', 'ushr-long', 'add-float', 'sub-float', 'mul-float', 'div-float', 'rem-float', 'add-double', 'sub-double', 'mul-double', 'div-double', 'rem-double', 'add-int/2addr', 'sub-int/2addr', 'mul-int/2addr', 'div-int/2addr', 'rem-int/2addr', 'and-int/2addr', 'or-int/2addr', 'xor-int/2addr', 'shl-int/2addr', 'shr-int/2addr', 'ushr-int/2addr', 'add-long/2addr', 'sub-long/2addr', 'mul-long/2addr', 'div-long/2addr', 'rem-long/2addr', 'and-long/2addr', 'or-long/2addr', 'xor-long/2addr', 'shl-long/2addr', 'shr-long/2addr', 'ushr-long/2addr', 'add-float/2addr', 'sub-float/2addr', 'mul-float/2addr', 'div-float/2addr', 'rem-float/2addr', 'add-double/2addr', 'sub-double/2addr', 'mul-double/2addr', 'div-double/2addr', 'rem-double/2addr', 'add-int/lit16', 'rsub-int', 'mul-int/lit16', 'div-int/lit16', 'rem-int/lit16', 'and-int/lit16', 'or-int/lit16', 'xor-int/lit16', 'add-int/lit8', 'rsub-int/lit8', 'mul-int/lit8', 'div-int/lit8', 'rem-int/lit8', 'and-int/lit8', 'or-int/lit8', 'xor-int/lit8', 'shl-int/lit8', 'shr-int/lit8', 'ushr-int/lit8', 'invoke-polymorphic', 'invoke-polymorphic/range', 'invoke-custom', 'invoke-custom/range', 'const-method-handle', 'const-method-type']
```
این opcode ها تنها بخشی از opcode های Android Dalvik نیستند بلکه تمام opcode ها هستند. پس فقط بخشی از opcode ها به عنوان حساس در نظر گرفته نشده‌اند و استخراج اطلاعات مربوط به تمامی opcode ها مدنظر است! 
تعداد این opcode ها برابر 224 است! -> کل opcode های Android Dalvik

در ادامه تابع `init()` به قسمتی از کد می‌رسیم که برابر است با:
```python
with open(_settings.headerfile) as f:
	self.permission = eval(f.read())
```
نتیجه اجرای این قسمت از کد برابر است با:
```bash
['Permission:android.car.permission.CAR_CAMERA', 'Permission:android.car.permission.CAR_HVAC', 'Permission:android.car.permission.CAR_MOCK_VEHICLE_HAL', 'Permission:android.car.permission.CAR_NAVIGATION_MANAGER', 'Permission:android.car.permission.CAR_PROJECTION', 'Permission:android.car.permission.CAR_RADIO', 'Permission:android.car.permission.CONTROL_APP_BLOCKING', 'Permission:android.permission.ACCESS_ALL_DOWNLOADS', 'Permission:android.permission.ACCESS_ALL_EXTERNAL_STORAGE', 'Permission:android.permission.ACCESS_BLUETOOTH_SHARE', 'Permission:android.permission.ACCESS_CACHE_FILESYSTEM', 'Permission:android.permission.ACCESS_COARSE_LOCATION', 'Permission:android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY', 'Permission:android.permission.ACCESS_DOWNLOAD_MANAGER', 'Permission:android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED', 'Permission:android.permission.ACCESS_DRM', 'Permission:android.permission.ACCESS_FINE_LOCATION', 'Permission:android.permission.ACCESS_KEYGUARD_SECURE_STORAGE', 'Permission:android.permission.ACCESS_LOCATION_EXTRA_COMMANDS', 'Permission:android.permission.ACCESS_MOCK_LOCATION', 'Permission:android.permission.ACCESS_NETWORK_STATE', 'Permission:android.permission.ACCESS_NOTIFICATIONS', 'Permission:android.permission.ACCESS_VOICE_INTERACTION_SERVICE', 'Permission:android.permission.ACCESS_WIFI_STATE', 'Permission:android.permission.ACCOUNT_MANAGER', 'Permission:android.permission.ASEC_ACCESS', 'Permission:android.permission.ASEC_CREATE', 'Permission:android.permission.ASEC_DESTROY', 'Permission:android.permission.ASEC_MOUNT_UNMOUNT', 'Permission:android.permission.ASEC_RENAME', 'Permission:android.permission.AUTHENTICATE_ACCOUNTS', 'Permission:android.permission.BACKUP', 'Permission:android.permission.BATTERY_STATS', 'Permission:android.permission.BIND_APPWIDGET', 'Permission:android.permission.BIND_CARRIER_SERVICES', 'Permission:android.permission.BIND_DEVICE_ADMIN', 'Permission:android.permission.BIND_VOICE_INTERACTION', 'Permission:android.permission.BLUETOOTH', 'Permission:android.permission.BLUETOOTH_ADMIN', 'Permission:android.permission.BLUETOOTH_PRIVILEGED', 'Permission:android.permission.BROADCAST_NETWORK_PRIVILEGED', 'Permission:android.permission.BROADCAST_SCORE_NETWORKS', 'Permission:android.permission.BROADCAST_STICKY', 'Permission:android.permission.CACHE_CONTENT', 'Permission:android.permission.CALL_PHONE', 'Permission:android.permission.CALL_PRIVILEGED', 'Permission:android.permission.CAMERA', 'Permission:android.permission.CAPTURE_AUDIO_OUTPUT', 'Permission:android.permission.CAPTURE_SECURE_VIDEO_OUTPUT', 'Permission:android.permission.CAPTURE_TV_INPUT', 'Permission:android.permission.CAPTURE_VIDEO_OUTPUT', 'Permission:android.permission.CHANGE_APP_IDLE_STATE', 'Permission:android.permission.CHANGE_COMPONENT_ENABLED_STATE', 'Permission:android.permission.CHANGE_CONFIGURATION', 'Permission:android.permission.CHANGE_DEVICE_IDLE_TEMP_WHITELIST', 'Permission:android.permission.CHANGE_NETWORK_STATE', 'Permission:android.permission.CHANGE_WIFI_MULTICAST_STATE', 'Permission:android.permission.CHANGE_WIFI_STATE', 'Permission:android.permission.CLEAR_APP_CACHE', 'Permission:android.permission.CLEAR_APP_GRANTED_URI_PERMISSIONS', 'Permission:android.permission.CLEAR_APP_USER_DATA', 'Permission:android.permission.CONFIGURE_DISPLAY_COLOR_MODE', 'Permission:android.permission.CONFIGURE_DISPLAY_COLOR_TRANSFORM', 'Permission:android.permission.CONFIGURE_WIFI_DISPLAY', 'Permission:android.permission.CONFIRM_FULL_BACKUP', 'Permission:android.permission.CONNECTIVITY_INTERNAL', 'Permission:android.permission.CONNECTIVITY_USE_RESTRICTED_NETWORKS', 'Permission:android.permission.CONTROL_LOCATION_UPDATES', 'Permission:android.permission.CONTROL_VPN', 'Permission:android.permission.CRYPT_KEEPER', 'Permission:android.permission.DELETE_CACHE_FILES', 'Permission:android.permission.DELETE_PACKAGES', 'Permission:android.permission.DEVICE_POWER', 'Permission:android.permission.DISABLE_KEYGUARD', 'Permission:android.permission.DOWNLOAD_CACHE_NON_PURGEABLE', 'Permission:android.permission.DOWNLOAD_WITHOUT_NOTIFICATION', 'Permission:android.permission.DUMP', 'Permission:android.permission.DVB_DEVICE', 'Permission:android.permission.EXPAND_STATUS_BAR', 'Permission:android.permission.FILTER_EVENTS', 'Permission:android.permission.FLASHLIGHT', 'Permission:android.permission.FORCE_BACK', 'Permission:android.permission.FORCE_STOP_PACKAGES', 'Permission:android.permission.FRAME_STATS', 'Permission:android.permission.FREEZE_SCREEN', 'Permission:android.permission.GET_ACCOUNTS', 'Permission:android.permission.GET_APP_GRANTED_URI_PERMISSIONS', 'Permission:android.permission.GET_APP_OPS_STATS', 'Permission:android.permission.GET_DETAILED_TASKS', 'Permission:android.permission.GET_INTENT_SENDER_INTENT', 'Permission:android.permission.GET_PACKAGE_SIZE', 'Permission:android.permission.GET_PROCESS_STATE_AND_OOM_SCORE', 'Permission:android.permission.GET_TASKS', 'Permission:android.permission.GET_TOP_ACTIVITY_INFO', 'Permission:android.permission.GLOBAL_SEARCH', 'Permission:android.permission.GRANT_REVOKE_PERMISSIONS', 'Permission:android.permission.GRANT_RUNTIME_PERMISSIONS', 'Permission:android.permission.HDMI_CEC', 'Permission:android.permission.INSTALL_DRM', 'Permission:android.permission.INSTALL_GRANT_RUNTIME_PERMISSIONS', 'Permission:android.permission.INSTALL_LOCATION_PROVIDER', 'Permission:android.permission.INSTALL_PACKAGES', 'Permission:android.permission.INTENT_FILTER_VERIFICATION_AGENT', 'Permission:android.permission.INTERACT_ACROSS_USERS', 'Permission:android.permission.INTERACT_ACROSS_USERS_FULL', 'Permission:android.permission.INTERNAL_SYSTEM_WINDOW', 'Permission:android.permission.INTERNET', 'Permission:android.permission.KILL_BACKGROUND_PROCESSES', 'Permission:android.permission.KILL_UID', 'Permission:android.permission.LOCAL_MAC_ADDRESS', 'Permission:android.permission.LOCATION_HARDWARE', 'Permission:android.permission.MAGNIFY_DISPLAY', 'Permission:android.permission.MANAGE_ACCOUNTS', 'Permission:android.permission.MANAGE_ACTIVITY_STACKS', 'Permission:android.permission.MANAGE_APP_OPS_RESTRICTIONS', 'Permission:android.permission.MANAGE_APP_TOKENS', 'Permission:android.permission.MANAGE_CA_CERTIFICATES', 'Permission:android.permission.MANAGE_DEVICE_ADMINS', 'Permission:android.permission.MANAGE_DOCUMENTS', 'Permission:android.permission.MANAGE_FINGERPRINT', 'Permission:android.permission.MANAGE_MEDIA_PROJECTION', 'Permission:android.permission.MANAGE_NETWORK_POLICY', 'Permission:android.permission.MANAGE_PROFILE_AND_DEVICE_OWNERS', 'Permission:android.permission.MANAGE_SOUND_TRIGGER', 'Permission:android.permission.MANAGE_USB', 'Permission:android.permission.MANAGE_USERS', 'Permission:android.permission.MANAGE_VOICE_KEYPHRASES', 'Permission:android.permission.MARK_NETWORK_SOCKET', 'Permission:android.permission.MEDIA_CONTENT_CONTROL', 'Permission:android.permission.MODIFY_APPWIDGET_BIND_PERMISSIONS', 'Permission:android.permission.MODIFY_AUDIO_ROUTING', 'Permission:android.permission.MODIFY_AUDIO_SETTINGS', 'Permission:android.permission.MODIFY_NETWORK_ACCOUNTING', 'Permission:android.permission.MODIFY_PARENTAL_CONTROLS', 'Permission:android.permission.MODIFY_PHONE_STATE', 'Permission:android.permission.MOUNT_FORMAT_FILESYSTEMS', 'Permission:android.permission.MOUNT_UNMOUNT_FILESYSTEMS', 'Permission:android.permission.MOVE_PACKAGE', 'Permission:android.permission.NFC', 'Permission:android.permission.NOTIFY_PENDING_SYSTEM_UPDATE', 'Permission:android.permission.OBSERVE_GRANT_REVOKE_PERMISSIONS', 'Permission:android.permission.PACKAGE_USAGE_STATS', 'Permission:android.permission.PACKAGE_VERIFICATION_AGENT', 'Permission:android.permission.PACKET_KEEPALIVE_OFFLOAD', 'Permission:android.permission.PEERS_MAC_ADDRESS', 'Permission:android.permission.PERSISTENT_ACTIVITY', 'Permission:android.permission.PROCESS_OUTGOING_CALLS', 'Permission:android.permission.QUERY_DO_NOT_ASK_CREDENTIALS_ON_BOOT', 'Permission:android.permission.READ_CALENDAR', 'Permission:android.permission.READ_CALL_LOG', 'Permission:android.permission.READ_CELL_BROADCASTS', 'Permission:android.permission.READ_CONTACTS', 'Permission:android.permission.READ_DREAM_STATE', 'Permission:android.permission.READ_EXTERNAL_STORAGE', 'Permission:android.permission.READ_FRAME_BUFFER', 'Permission:android.permission.READ_LOGS', 'Permission:android.permission.READ_NETWORK_USAGE_HISTORY', 'Permission:android.permission.READ_PHONE_STATE', 'Permission:android.permission.READ_PRECISE_PHONE_STATE', 'Permission:android.permission.READ_PRIVILEGED_PHONE_STATE', 'Permission:android.permission.READ_PROFILE', 'Permission:android.permission.READ_SEARCH_INDEXABLES', 'Permission:android.permission.READ_SMS', 'Permission:android.permission.READ_SOCIAL_STREAM', 'Permission:android.permission.READ_SYNC_SETTINGS', 'Permission:android.permission.READ_SYNC_STATS', 'Permission:android.permission.READ_USER_DICTIONARY', 'Permission:android.permission.READ_WIFI_CREDENTIAL', 'Permission:android.permission.REAL_GET_TASKS', 'Permission:android.permission.REBOOT', 'Permission:android.permission.RECEIVE_BLUETOOTH_MAP', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED', 'Permission:android.permission.RECEIVE_MMS', 'Permission:android.permission.RECEIVE_SMS', 'Permission:android.permission.RECORD_AUDIO', 'Permission:android.permission.RECOVERY', 'Permission:android.permission.REGISTER_CONNECTION_MANAGER', 'Permission:android.permission.REGISTER_WINDOW_MANAGER_LISTENERS', 'Permission:android.permission.REMOTE_AUDIO_PLAYBACK', 'Permission:android.permission.REMOVE_TASKS', 'Permission:android.permission.REORDER_TASKS', 'Permission:android.permission.RESET_FINGERPRINT_LOCKOUT', 'Permission:android.permission.RESET_SHORTCUT_MANAGER_THROTTLING', 'Permission:android.permission.RESTART_PACKAGES', 'Permission:android.permission.RETRIEVE_WINDOW_INFO', 'Permission:android.permission.REVOKE_RUNTIME_PERMISSIONS', 'Permission:android.permission.SCORE_NETWORKS', 'Permission:android.permission.SEND_RESPOND_VIA_MESSAGE', 'Permission:android.permission.SEND_SMS', 'Permission:android.permission.SEND_SMS_NO_CONFIRMATION', 'Permission:android.permission.SERIAL_PORT', 'Permission:android.permission.SET_ACTIVITY_WATCHER', 'Permission:android.permission.SET_ALWAYS_FINISH', 'Permission:android.permission.SET_ANIMATION_SCALE', 'Permission:android.permission.SET_DEBUG_APP', 'Permission:android.permission.SET_INPUT_CALIBRATION', 'Permission:android.permission.SET_KEYBOARD_LAYOUT', 'Permission:android.permission.SET_ORIENTATION', 'Permission:android.permission.SET_POINTER_SPEED', 'Permission:android.permission.SET_PREFERRED_APPLICATIONS', 'Permission:android.permission.SET_PROCESS_LIMIT', 'Permission:android.permission.SET_SCREEN_COMPATIBILITY', 'Permission:android.permission.SET_TIME', 'Permission:android.permission.SET_TIME_ZONE', 'Permission:android.permission.SET_WALLPAPER', 'Permission:android.permission.SET_WALLPAPER_COMPONENT', 'Permission:android.permission.SET_WALLPAPER_HINTS', 'Permission:android.permission.SHUTDOWN', 'Permission:android.permission.SIGNAL_PERSISTENT_PROCESSES', 'Permission:android.permission.START_ANY_ACTIVITY', 'Permission:android.permission.START_TASKS_FROM_RECENTS', 'Permission:android.permission.STATUS_BAR', 'Permission:android.permission.STATUS_BAR_SERVICE', 'Permission:android.permission.STOP_APP_SWITCHES', 'Permission:android.permission.STORAGE_INTERNAL', 'Permission:android.permission.SYSTEM_ALERT_WINDOW', 'Permission:android.permission.TABLET_MODE', 'Permission:android.permission.TABLET_MODE_LISTENER', 'Permission:android.permission.TETHER_PRIVILEGED', 'Permission:android.permission.TRANSMIT_IR', 'Permission:android.permission.TV_INPUT_HARDWARE', 'Permission:android.permission.UPDATE_APP_OPS_STATS', 'Permission:android.permission.UPDATE_DEVICE_STATS', 'Permission:android.permission.UPDATE_LOCK', 'Permission:android.permission.USE_CREDENTIALS', 'Permission:android.permission.USE_FINGERPRINT', 'Permission:android.permission.USE_SIP', 'Permission:android.permission.VIBRATE', 'Permission:android.permission.WAKE_LOCK', 'Permission:android.permission.WRITE_APN_SETTINGS', 'Permission:android.permission.WRITE_CALENDAR', 'Permission:android.permission.WRITE_CALL_LOG', 'Permission:android.permission.WRITE_CONTACTS', 'Permission:android.permission.WRITE_DREAM_STATE', 'Permission:android.permission.WRITE_EXTERNAL_STORAGE', 'Permission:android.permission.WRITE_PROFILE', 'Permission:android.permission.WRITE_SECURE_SETTINGS', 'Permission:android.permission.WRITE_SETTINGS', 'Permission:android.permission.WRITE_SMS', 'Permission:android.permission.WRITE_SOCIAL_STREAM', 'Permission:android.permission.WRITE_SYNC_SETTINGS', 'Permission:android.permission.WRITE_USER_DICTIONARY', 'Permission:com.android.browser.permission.READ_HISTORY_BOOKMARKS', 'Permission:com.android.browser.permission.WRITE_HISTORY_BOOKMARKS', 'Permission:com.android.cts.permissionNormal', 'Permission:com.android.cts.permissionNotUsedWithSignature', 'Permission:com.android.cts.permissionWithSignature', 'Permission:com.android.email.permission.ACCESS_PROVIDER', 'Permission:com.android.email.permission.READ_ATTACHMENT', 'Permission:com.android.gallery3d.filtershow.permission.READ', 'Permission:com.android.gallery3d.filtershow.permission.WRITE', 'Permission:com.android.gallery3d.permission.GALLERY_PROVIDER', 'Permission:com.android.launcher.permission.READ_SETTINGS', 'Permission:com.android.launcher.permission.WRITE_SETTINGS', 'Permission:com.android.launcher3.permission.READ_SETTINGS', 'Permission:com.android.launcher3.permission.WRITE_SETTINGS', 'Permission:com.android.printspooler.permission.ACCESS_ALL_PRINT_JOBS', 'Permission:com.android.providers.imps.permission.READ_ONLY', 'Permission:com.android.providers.imps.permission.WRITE_ONLY', 'Permission:com.android.providers.tv.permission.READ_EPG_DATA', 'Permission:com.android.providers.tv.permission.WRITE_EPG_DATA', 'Permission:com.android.rcs.eab.permission.READ_WRITE_EAB', 'Permission:com.android.server.telecom.permission.REGISTER_PROVIDER_OR_SUBSCRIPTION', 'Permission:com.android.voicemail.permission.ADD_VOICEMAIL', 'Permission:getWindowToken', 'Permission:temporaryEnableAccessibilityStateUntilKeyguardRemoved', 'Permission:ti.permission.FMRX', 'Permission:ti.permission.FMRX_ADMIN']
```
خروجی این قسمت از کد نشان می‌دهد که این مجوزها که درون فایل `head.txt` قرار دارند، عملا زیرمجموعه‌ای از permission های اندرویدی هستند که **==حساس==** تلقی می‌گردند و برای تشخیص بدافزار و عمل‌های مخرب به کار می‌روند.

به ادامه تابع `init()` خواهیم پرداخت که به تکه کد زیر می‌رسیم:
```python
self.cppermission = self.get_permission()
```
خروجی این قسمت از کد عبارت است از:
```bash
{'content://browser': {('R', 'Permission:com.android.browser.permission.READ_HISTORY_BOOKMARKS'), ('W', 'Permission:com.android.browser.permission.WRITE_HISTORY_BOOKMARKS')}, 'content://com.android.browser': {('R', 'Permission:com.android.browser.permission.READ_HISTORY_BOOKMARKS'), ('W', 'Permission:com.android.browser.permission.WRITE_HISTORY_BOOKMARKS')}, 'content://cellbroadcasts': {('R', 'Permission:android.permission.READ_CELL_BROADCASTS')}, 'content://com.android.email.attachmentprovider': {('R', 'Permission:com.android.email.permission.READ_ATTACHMENT')}, 'content://com.android.email.notifier': {('W', 'Permission:com.android.email.permission.ACCESS_PROVIDER'), ('R', 'Permission:com.android.email.permission.ACCESS_PROVIDER')}, 'content://com.android.email.provider': {('W', 'Permission:com.android.email.permission.ACCESS_PROVIDER'), ('R', 'Permission:com.android.email.permission.ACCESS_PROVIDER')}, 'content://com.android.exchange.directory.provider': {('R', 'Permission:android.permission.READ_CONTACTS')}, 'content://com.android.launcher2.settings': {('W', 'Permission:com.android.launcher.permission.WRITE_SETTINGS'), ('R', 'Permission:com.android.launcher.permission.READ_SETTINGS')}, 'content://com.android.mms.SuggestionsProvider': {('R', 'Permission:android.permission.READ_SMS')}, 'content://icc': {('R', 'Permission:android.permission.READ_CONTACTS'), ('W', 'Permission:android.permission.WRITE_CONTACTS')}, 'content://com.android.calendar': {('W', 'Permission:android.permission.WRITE_CALENDAR'), ('R', 'Permission:android.permission.READ_CALENDAR')}, 'content://call_log': {('R', 'Permission:android.permission.READ_CONTACTS'), ('W', 'Permission:android.permission.WRITE_CALL_LOG'), ('W', 'Permission:android.permission.WRITE_CONTACTS'), ('R', 'Permission:android.permission.READ_CALL_LOG')}, 'content://com.android.contacts': {('R', 'Permission:android.permission.READ_CONTACTS'), ('W', 'Permission:android.permission.WRITE_CONTACTS')}, 'content://contacts': {('R', 'Permission:android.permission.READ_CONTACTS'), ('W', 'Permission:android.permission.WRITE_CONTACTS')}, 'content://com.android.voicemail': {('W', 'Permission:com.android.voicemail.permission.ADD_VOICEMAIL'), ('R', 'Permission:com.android.voicemail.permission.ADD_VOICEMAIL')}, 'content://settings': {('W', 'Permission:android.permission.WRITE_SETTINGS')}, 'content://mms': {('W', 'Permission:android.permission.WRITE_SMS'), ('R', 'Permission:android.permission.READ_SMS')}, 'content://mms-sms': {('W', 'Permission:android.permission.WRITE_SMS'), ('R', 'Permission:android.permission.READ_SMS')}, 'content://sms': {('W', 'Permission:android.permission.WRITE_SMS'), ('R', 'Permission:android.permission.READ_SMS')}, 'content://user_dictionary': {('R', 'Permission:android.permission.READ_USER_DICTIONARY'), ('W', 'Permission:android.permission.WRITE_USER_DICTIONARY')}, 'content://com.android.gallery3d.filtershow.provider.SharedImageProvider': {('W', 'Permission:com.android.gallery3d.filtershow.permission.WRITE'), ('R', 'Permission:com.android.gallery3d.filtershow.permission.READ')}, 'content://com.android.gallery3d.provider': {('W', 'Permission:com.android.gallery3d.permission.GALLERY_PROVIDER'), ('R', 'Permission:com.android.gallery3d.permission.GALLERY_PROVIDER')}, 'content://com.android.externalstorage.documents': {('W', 'Permission:android.permission.MANAGE_DOCUMENTS'), ('R', 'Permission:android.permission.MANAGE_DOCUMENTS')}, 'content://com.android.launcher3.settings': {('R', 'Permission:com.android.launcher3.permission.READ_SETTINGS'), ('W', 'Permission:com.android.launcher3.permission.WRITE_SETTINGS')}, 'content://com.android.providers.downloads.documents': {('W', 'Permission:android.permission.MANAGE_DOCUMENTS'), ('R', 'Permission:android.permission.MANAGE_DOCUMENTS')}, 'content://com.android.providers.media.documents': {('W', 'Permission:android.permission.MANAGE_DOCUMENTS'), ('R', 'Permission:android.permission.MANAGE_DOCUMENTS')}, 'content://com.android.cellbroadcastreceiver': {('W', 'Permission:android.permission.READ_SEARCH_INDEXABLES'), ('R', 'Permission:android.permission.READ_SEARCH_INDEXABLES')}, 'content://com.android.phone': {('W', 'Permission:android.permission.READ_SEARCH_INDEXABLES'), ('R', 'Permission:android.permission.READ_SEARCH_INDEXABLES')}, 'content://hbpcd_lookup': {('W', 'Permission:android.permission.MODIFY_PHONE_STATE')}, 'content://android.media.tv': {('R', 'Permission:com.android.providers.tv.permission.READ_EPG_DATA'), ('W', 'Permission:com.android.providers.tv.permission.WRITE_EPG_DATA')}, 'content://com.android.settings': {('W', 'Permission:android.permission.READ_SEARCH_INDEXABLES'), ('R', 'Permission:android.permission.READ_SEARCH_INDEXABLES')}, 'content://com.android.mtp.documents': {('W', 'Permission:android.permission.MANAGE_DOCUMENTS'), ('R', 'Permission:android.permission.MANAGE_DOCUMENTS')}, 'content://com.android.crashreportprovider': {('W', 'Permission:android.permission.READ_LOGS'), ('R', 'Permission:android.permission.READ_LOGS')}, 'content://call_log_shadow': {('W', 'Permission:android.permission.MANAGE_USERS'), ('R', 'Permission:android.permission.MANAGE_USERS')}, 'content://com.android.rcs.eab': {('W', 'Permission:com.android.rcs.eab.permission.READ_WRITE_EAB'), ('R', 'Permission:com.android.rcs.eab.permission.READ_WRITE_EAB')}, 'content://browser/bookmarks/search_suggest_query': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.bluetooth.opp/btopp': {('W', 'Permission:android.permission.ACCESS_BLUETOOTH_SHARE'), ('R', 'Permission:android.permission.ACCESS_BLUETOOTH_SHARE')}, 'content://com.android.contacts/contacts/.*/photo': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.contacts/search_suggest_query': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.contacts/search_suggest_shortcut': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.exchange.provider': {('W', 'Permission:com.android.email.permission.ACCESS_PROVIDER'), ('R', 'Permission:com.android.email.permission.ACCESS_PROVIDER')}, 'content://com.android.mms.SuggestionsProvider/search_suggest_query': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.mms.SuggestionsProvider/search_suggest_shortcut': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.social': {('R', 'Permission:android.permission.READ_CONTACTS'), ('W', 'Permission:android.permission.WRITE_CONTACTS')}, 'content://contacts/contacts/.*/photo': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://contacts/search_suggest_query': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://contacts/search_suggest_shortcut': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://ctspermissionwithsignature': {('R', 'Permission:com.android.cts.permissionWithSignature'), ('W', 'Permission:com.android.cts.permissionWithSignature')}, 'content://downloads': {('W', 'Permission:android.permission.ACCESS_ALL_DOWNLOADS'), ('R', 'Permission:android.permission.ACCESS_ALL_DOWNLOADS'), ('W', 'Permission:android.permission.ACCESS_DOWNLOAD_MANAGER'), ('R', 'Permission:android.permission.ACCESS_DOWNLOAD_MANAGER')}, 'content://imps': {('R', 'Permission:com.android.providers.imps.permission.READ_ONLY'), ('W', 'Permission:com.android.providers.imps.permission.WRITE_ONLY')}, 'content://ctspermissionwithsignaturegranting': {('R', 'Permission:com.android.cts.permissionWithSignature'), ('W', 'Permission:com.android.cts.permissionWithSignature')}, 'content://ctspermissionwithsignaturepath/foo': {('R', 'Permission:com.android.cts.permissionWithSignature')}, 'content://ctspermissionwithsignaturepath': {('R', 'Permission:com.android.cts.permissionNotUsedWithSignature'), ('W', 'Permission:com.android.cts.permissionNotUsedWithSignature')}, 'content://ctspermissionwithsignaturepath/yes': {('R', 'Permission:com.android.cts.permissionWithSignature')}, 'content://downloads/download': {('W', 'Permission:android.permission.INTERNET'), ('R', 'Permission:android.permission.INTERNET')}, 'content://downloads/my_downloads': {('W', 'Permission:android.permission.INTERNET'), ('R', 'Permission:android.permission.INTERNET')}, 'content://com.android.browser/bookmarks/search_suggest_query': {('R', 'Permission:android.permission.GLOBAL_SEARCH')}, 'content://com.android.browser.home': {('R', 'Permission:com.android.browser.permission.READ_HISTORY_BOOKMARKS')}, 'content://downloads/all_downloads': {('W', 'Permission:android.permission.ACCESS_ALL_DOWNLOADS'), ('R', 'Permission:android.permission.ACCESS_ALL_DOWNLOADS')}, 'content://ctspermissionwithsignaturepathrestricting/foo/bar': {('R', 'Permission:com.android.cts.permissionNormal')}, 'content://ctspermissionwithsignaturepathrestricting/foo': {('R', 'Permission:com.android.cts.permissionWithSignature')}, 'content://media/external/': {('R', 'Permission:android.permission.WRITE_EXTERNAL_STORAGE'), ('R', 'Permission:android.permission.READ_EXTERNAL_STORAGE')}, 'content://telephony/carriers': {('W', 'Permission:android.permission.WRITE_APN_SETTINGS'), ('R', 'Permission:android.permission.WRITE_APN_SETTINGS')}}
```
در واقع در این قسمت داریم که uri های مربوط به content providerها به permissionهای مربوط به خود نگاشت می‌گردند.

### تابع `analysis_app`
در ابتدا کد این تابع در اینجا مطرح می‌گردد و قدم به قدم تحلیل می‌شود:
```python
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
        opcodeFile.close()
        cpermission = Permission(G=G, path=permissionFilename, class_functions=class_functions, super_dic=super_dic,
                                 implement_dic=implement_dic, dexobj=dexobj, permission=self.permission,
                                 cppermission=self.cppermission, method2nodeMap=method2nodeMap)
        cpermission.generate()
        class2init = cpermission.getClass2init()
        sensitiveapimap = cpermission.getsensitive_api()
        ctpl = Tpl(log.filename, G, tplFilename, sensitiveapimap, self.permission, class2init, self.deepth)
        ctpl.generate()
```
در ابتدا این تابع یک خط کد هست به صورت زیر:
```python
CGpath = apk_filename.replace(self.APKPath, self.CGPath)[:-4]
```
خروجی این تابع بر اساس مثال ما برابر است با:
```bash
/home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/decompile/1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720
```
در واقع در این خروجی، `1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720` نام فایل APK است.

در ادامه فایل `call.gml` در دایرکتوری زیر ذخیره می‌گردد:
```bash
/home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/decompile/1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720
```

پس از آن برای ذخیره opcodeها اقدام می‌شود که متغیرهای مربوط به آن (خروجی‌شان) به صورت زیر هستند:
```bash
Opcode Filename:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/opcode/1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720.csv
opcodePath:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/opcode
```

و بعد از آن فایل permissionها:
```bash
PermissionFileName:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/permission/1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720.csv
permissionPath:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/permission
```

در ادامه برای TPLها هم داریم:
```bash
tplFileName:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/tpl/1aa440d4f99f709345b44484edd0d31aad29f5c5e00201be6d222fc16a896720.csv
tplPath:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/tpl
```

پس از ایجاد آدرس‌های مدنظر یا چک کردن آن‌ها به جهت اطمینان از وجود آن‌ها، به استخراج call graphها پرداخته می‌شود:
```python
if not os.path.exists(CGfilename):
	G = analysisobj.get_call_graph()  # call graph
```
خروجی این قسمت، `G` عبارت است از:
```bash
>>> MultiDiGraph with 7959 nodes and 24686 edges
```
در ادامه بررسی می‌کند که آیا فایل `tplFileName` موجود است یا نه! در صورتی که موجود باشد، خارج می‌گردد.
اما قسمت بعدی کد عبارت است از:
```python
opcodeFile = utils.create_csv(self.smali_opcode, opcodeFilename)
```
با استفاده از خط کد بالا یک فایل `.csv` با نامی که درون متغیر `opcodeFileName` ذخیره شده است، ایجاد می‌شود. هِدِرهای این فایل csv، اعضای لیست `self.smali_opcode` هستند. برای مثال اگر لیست `self.smali_opcode` برابر زیر باشد:
```bash
['nop', 'move', 'return-void', 'invoke-virtual']
```
آنگاه فایل csv دارای ساختار زیر خواهد بود:
```bash
nop,move,return-void,invoke-virtual
```

یکی از اصلی‌ترین قسمت‌های این کد، قسمت زیر است:
```python
method2nodeMap = self.getMethod2NodeMap(G)
```
در واقع این خط از کد برای ایجاد نگاشت بین متدها در APK و گره‌های مربوط به آن‌ها در گراف فراخوانی `G` استفاده می‌گردد. از این نگاشت بعدا در شمردن تعداد opcodeها، نگاشت مجوزها و ... استفاده می‌شود.
نگاشت درون `method2nodeMap` به صورت زیر است:
```bash
{..., 'Ljava/text/DecimalFormat; format (D)Ljava/lang/String;': (7925, 'Ljava/text/DecimalFormat;->format(D)Ljava/lang/String;'), 'Ljava/lang/Math; abs (D)D': (7926, 'Ljava/lang/Math;->abs(D)D'), 'Landroid/graphics/Canvas; drawRect (Landroid/graphics/Rect; Landroid/graphics/Paint;)V': (7927, 'Landroid/graphics/Canvas;->drawRect(Landroid/graphics/Rect; Landroid/graphics/Paint;)V'), 'Lcom/widget/view/BrokenLineGraphView; drawSumPoint ()V': (7928, 'Lcom/widget/view/BrokenLineGraphView;->drawSumPoint()V'), 'Lcom/widget/view/BrokenLineGraphView; setForceHardRender (Z)V': (7929, 'Lcom/widget/view/BrokenLineGraphView;->setForceHardRender(Z)V'), 'Landroid/graphics/Canvas; drawLine (F F F F Landroid/graphics/Paint;)V': (7930, 'Landroid/graphics/Canvas;->drawLine(F F F F Landroid/graphics/Paint;)V'), 'Landroid/graphics/Paint; setPathEffect (Landroid/graphics/PathEffect;)Landroid/graphics/PathEffect;': (7931, 'Landroid/graphics/Paint;->setPathEffect(Landroid/graphics/PathEffect;)Landroid/graphics/PathEffect;'), 'Landroid/graphics/Paint; setStrokeWidth (F)V': (7932, 'Landroid/graphics/Paint;->setStrokeWidth(F)V'), 'Landroid/graphics/DashPathEffect; <init> ([F F)V': (7933, 'Landroid/graphics/DashPathEffect;-><init>([F F)V'), 'Landroid/graphics/Color; argb (I I I I)I': (7934, 'Landroid/graphics/Color;->argb(I I I I)I'), 'Lcom/widget/view/BrokenLineGraphView; getTallyWeekItem ()Lcom/model/TallyWeekItem;': (7935, 'Lcom/widget/view/BrokenLineGraphView;->getTallyWeekItem()Lcom/model/TallyWeekItem;'), 'Landroid/graphics/Paint; getTextWidths (Ljava/lang/String; [F)I': (7936, 'Landroid/graphics/Paint;->getTextWidths(Ljava/lang/String; [F)I'), 'Lcom/widget/view/BrokenLineGraphView; setBackgroundResource (I)V': (7937, 'Lcom/widget/view/BrokenLineGraphView;->setBackgroundResource(I)V'), 'Lcom/widget/view/BrokenLineGraphView; setLayerType (I Landroid/graphics/Paint;)V': (7938, 'Lcom/widget/view/BrokenLineGraphView;->setLayerType(I Landroid/graphics/Paint;)V'), 'Landroid/graphics/Canvas; drawCircle (F F F Landroid/graphics/Paint;)V': (7939, 'Landroid/graphics/Canvas;->drawCircle(F F F Landroid/graphics/Paint;)V'), 'Landroid/view/View; onDraw (Landroid/graphics/Canvas;)V': (7940, 'Landroid/view/View;->onDraw(Landroid/graphics/Canvas;)V'), 'Landroid/view/View; onMeasure (I I)V': (7941, 'Landroid/view/View;->onMeasure(I I)V'), 'Lcom/widget/view/MyRelativeLayout; <init> (Landroid/content/Context;)V': (7942, 'Lcom/widget/view/MyRelativeLayout;-><init>(Landroid/content/Context;)V'), 'Lcom/widget/view/MyRelativeLayout; <init> (Landroid/content/Context; Landroid/util/AttributeSet;)V': (7943, 'Lcom/widget/view/MyRelativeLayout;-><init>(Landroid/content/Context; Landroid/util/AttributeSet;)V'), 'Lcom/widget/view/MyRelativeLayout; <init> (Landroid/content/Context; Landroid/util/AttributeSet; I)V': (7944, 'Lcom/widget/view/MyRelativeLayout;-><init>(Landroid/content/Context; Landroid/util/AttributeSet; I)V'), 'Lcom/widget/view/MyRelativeLayout; onDraw (Landroid/graphics/Canvas;)V': (7945, 'Lcom/widget/view/MyRelativeLayout;->onDraw(Landroid/graphics/Canvas;)V'), 'Landroid/widget/RelativeLayout; onDraw (Landroid/graphics/Canvas;)V': (7946, 'Landroid/widget/RelativeLayout;->onDraw(Landroid/graphics/Canvas;)V'), 'Lcom/widget/view/TallyBarChartView; <init> (Landroid/content/Context;)V': (7947, 'Lcom/widget/view/TallyBarChartView;-><init>(Landroid/content/Context;)V'), 'Lcom/widget/view/TallyBarChartView; init ()V': (7948, 'Lcom/widget/view/TallyBarChartView;->init()V'), 'Lcom/widget/view/TallyBarChartView; <init> (Landroid/content/Context; Landroid/util/AttributeSet;)V': (7949, 'Lcom/widget/view/TallyBarChartView;-><init>(Landroid/content/Context; Landroid/util/AttributeSet;)V'), 'Lcom/widget/view/TallyBarChartView; <init> (Landroid/content/Context; Landroid/util/AttributeSet; I)V': (7950, 'Lcom/widget/view/TallyBarChartView;-><init>(Landroid/content/Context; Landroid/util/AttributeSet; I)V'), 'Lcom/widget/view/TallyBarChartView; getFontHeight (F)I': (7951, 'Lcom/widget/view/TallyBarChartView;->getFontHeight(F)I'), 'Landroid/graphics/Paint; getFontMetrics ()Landroid/graphics/Paint$FontMetrics;': (7952, 'Landroid/graphics/Paint;->getFontMetrics()Landroid/graphics/Paint$FontMetrics;'), 'Lcom/widget/view/TallyBarChartView; getMonthItem ()Lcom/model/TallMonthItem;': (7953, 'Lcom/widget/view/TallyBarChartView;->getMonthItem()Lcom/model/TallMonthItem;'), 'Lcom/widget/view/TallyBarChartView; getTextWidth (Ljava/lang/String;)F': (7954, 'Lcom/widget/view/TallyBarChartView;->getTextWidth(Ljava/lang/String;)F'), 'Lcom/widget/view/TallyBarChartView; setBackgroundResource (I)V': (7955, 'Lcom/widget/view/TallyBarChartView;->setBackgroundResource(I)V'), 'Landroid/graphics/Paint; setFakeBoldText (Z)V': (7956, 'Landroid/graphics/Paint;->setFakeBoldText(Z)V'), 'Landroid/graphics/BitmapFactory; decodeResource (Landroid/content/res/Resources; I)Landroid/graphics/Bitmap;': (7957, 'Landroid/graphics/BitmapFactory;->decodeResource(Landroid/content/res/Resources; I)Landroid/graphics/Bitmap;'), 'Lcom/widget/view/TallyBarChartView; getResources ()Landroid/content/res/Resources;': (7958, 'Lcom/widget/view/TallyBarChartView;->getResources()Landroid/content/res/Resources;')}
```
یعنی در واقع داریم که به قالب زیر است:
```bash
{
    'Lcom/example/MyClass;->methodName()V': (node_id, label)
}
```
در این قالب، `node_id` برابر شماره گره است. `label` هم اصلاح‌شدۀ نام متد است.

بعد از این قسمت یک حلقه بر روی تمامی کلاس‌ها گذرانده می‌شود. در ابتدای این حلقه نام هر کلاس بدست می‌آید که مثلا نام یکی از کلاس‌ها می‌تواند چنین باشد:
```bash
Landroid/support/v4/content/ContextCompat;
```
بررسی superclass هر class یکی از کارهای مهم در این حلقه است. کد این قسمت عبارت است از:
```python
if classes.extends != "Ljava/lang/Object;":
	super_dic[class_name] = str(classes.extends)
	if str(classes.extends) in self.replacemap:
		implement_dic[class_name] = str(classes.extends)
```
در واقع چک می‌شود که آیا این class از کلاس `Ljava/lang/object;` ارث‌بری کرده است یا نه!
> دلیل این موضوع به ساختار زبان جاوا و بالتبع اندروید بستگی دارد. کلاس `Ljava/lang/object` یک کلاس root برای تمامی کلاس‌ها محسوب می‌شود. به همین دلیل وقتی کلاس ما از کلاسی غیر از root ارث‌بری نکرده باشد، superclass آن همین `Ljava/lang/object` خواهد بود.
> می‌توان در پردازش‌ها کلاس `Ljava/lang/object` را نیز به عنوان superclass آورد اما بهتر است صرفا نکات با معنای بالاتر را آورد تا **نویز در پردازش** کمتر شود.

در صورتی که superclass چیزی غیر از root بود، نگاشت بین کلاس و سوپرکلاس در دیکشنری `super_dic` ذخیره می‌گردد.

اگر superclass این کلاس در `self.replacemap` باشد، مشخص است که کلاس از برخی از رفتارهای از پیش‌تعریف‌شده ارث می‌برد. که در بالاتر نیز به متدهای `doInBackground` و ... اشاره شده بود. علاوه بر نگاشت قبلی، `super_dic`، نگاشتی نیز برای این کلاس‌ها صورت می‌پذیرد که در `implement_dic` ذخیره می‌گردد.
در ادامه بررسی می‌گردد که آیا کلاس interface ای از interface های مدنظر را implement کرده است یا نه:
```python
if classes.implements:
	for imp in classes.implements:
		if str(imp) in self.replacemap:
			implement_dic[class_name] = str(imp)
```
در واقع هر چک می‌گردد که آیا از متدهایی مانند `Runnable` که در بالاتر گفته شده بود، implement صورت پذیرفته است یا نه! فرقی که بین ارث‌بری و implement هست را در زیر معین می‌کنیم:
```java
public class MyClass extends AsyncTask implements Runnable {
    @Override
    protected void doInBackground(Void... params) {
        // Async task logic
    }

    @Override 
    // run() in Runnable
    public void run() {
        // Thread logic
    }
}
```
پس از پردازش این بلاک از کد در ابتدا خواهیم داشت:
```bash
super_dic['Lcom/example/MyClass;'] = 'Landroid/os/AsyncTask;'
implement_dic['Lcom/example/MyClass;'] = 'Landroid/os/AsyncTask;'
```
و سپس بعد از شرط دوم:
```bash
implement_dic['Lcom/example/MyClass;'] = 'Ljava/lang/Runnable;'
```

از این دو نگاشت برای بهینه‌سازی گراف فراخوانی، بدست آوردن permissionها و ... استفاده می‌گردد که بعدا به آن‌ها در جای خود پرداخته خواهد شد.

در ادامه به بررسی methodها پرداخته می‌شود:
```python
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
```
این حلقه methodهای یک کلاس را بررسی می‌کند. در ابتدا بررسی می‌شود که آیا یک متد external است یا نه؟ یک متد external، در بحث permission ها اهمیت دارد وگرنه چون از کتابخانه‌های استاندارد و ... استفاده  می‌شود، نمی‌توان ویژگی‌های دیگر مانند opcodeها و ... را بررسی کرد.
در این کد، نگاشتی بین یک کلاس و متدهای آن (توابع آن) نوشته می‌شود که در `class_function` ذخیره می‌گردد. نمونه‌ای از این مقدار برابر است با:
```bash
defaultdict(<class 'list'>, {'Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl;': ['Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl; getCanRetrieveWindowContent (Landroid/accessibilityservice/AccessibilityServiceInfo;)Z', 'Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl; getDescription (Landroid/accessibilityservice/AccessibilityServiceInfo;)Ljava/lang/String;', 'Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl; getId (Landroid/accessibilityservice/AccessibilityServiceInfo;)Ljava/lang/String;', 'Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl; getResolveInfo (Landroid/accessibilityservice/AccessibilityServiceInfo;)Landroid/content/pm/ResolveInfo;', 'Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl; getSettingsActivityName (Landroid/accessibilityservice/AccessibilityServiceInfo;)Ljava/lang/String;']}
```
در این مثال داریم که کلاس `Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl;` دارای یک متد به نام `Landroid/support/v4/accessibilityservice/AccessibilityServiceInfoCompat$AccessibilityServiceInfoVersionImpl; getId (Landroid/accessibilityservice/AccessibilityServiceInfo;)Ljava/lang/String;` است.

در ادامه یک ماتریس تشکیل می‌گردد که برای هر method تعداد هر کدام از instruction های یک method ذخیره می‌گردد.
```plaintext
Method's instruction vector:
defaultdict(<class 'int'>, {'invoke-direct': 1, 'invoke-virtual': 2, 'move-result-object': 1, 'iput-object': 1, 'if-eqz': 2, 'iget-object': 2, 'return-void': 1})
```
در انتها opcodeها در یک فایل نوشته می‌شود. برای مثال:

| method_node_id | move | invoke-virtual | return-void | const/4 | const-string |
| -------------- | ---- | -------------- | ----------- | ------- | ------------ |
| 12             | 5    | 3              | 1           | 2       | 0            |
| 15             | 0    | 6              | 3           | 1       | 1            |

### کلاس `Permission`
کلاس `Permission` نقش مهمی در تجزیه و تحلیل مجوزهای استفاده شده توسط APK ایفا می‌کند. این برنامه بر روی نگاشت مجوزها به APIها، متذها و گره‌های گراف فراخوانی تمرکز می‌کند و امکان تجزیه و تحلیل دقیق پیامدهای امنیت و حریم خصوصی برنامه را فراهم می‌کند.

کد این کلاس عبارت است از:
```python
# coding=utf-8

from collections import defaultdict
from . import _settings
from feature.Utils.utils import *

# Permission includes the api and contentprovider parts
# Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder)
query_method = "Landroid/content/ContentResolver;->query(Landroid/net/Uri; [Ljava/lang/String; Ljava/lang/String; [Ljava/lang/String; Ljava/lang/String;)Landroid/database/Cursor;"
# Uri insert(Uri uri, ContentValues values)
insert_method = "Landroid/content/ContentResolver;->insert(Landroid/net/Uri; Landroid/content/ContentValues;)Landroid/net/Uri;"
# int delete(Uri uri, String selection, String[] selectionArgs)
delete_method = "Landroid/content/ContentResolver;->delete(Landroid/net/Uri; Ljava/lang/String; [Ljava/lang/String;)I"
# int update(Uri uri, ContentValues values, String selection, String[] selectionArgs)
update_method = "Landroid/content/ContentResolver;->update(Landroid/net/Uri; Landroid/content/ContentValues; Ljava/lang/String; [Ljava/lang/String;)I"

method_dic = {query_method: ['R'], insert_method: ['W'], delete_method: ['W'], update_method: ['W', 'R']}

class Permission():
    def __init__(self, G, path, class_functions, super_dic, implement_dic, dexobj, permission, cppermission, method2nodeMap):
        self.G = G  # call graph
        self.path = path  # csv path to save
        self.class_functions = class_functions
        self.super_dic = super_dic
        self.implement_dic = implement_dic
        self.dexobj = dexobj
        self.sensitiveapimap = {}  # All sensitive nodes in this apk and the permissions they involve
        self.class2runinit = defaultdict(dict)
        self.replacemap = {'Landroid/os/AsyncTask;': ['onPreExecute', 'doInBackground'],
                           'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']}
        self.permission = permission
        self.cp_permission = cppermission
        self.method2nodeMap = method2nodeMap
  
    def deal_node(self, nodeid):
        permission_node = {}
        targets = self.G.successors(nodeid)
        for t in targets:
            label = get_label(t, self.G)
            for k in method_dic.keys():
                if label.find(k) >= 0:
                    for ki in method_dic[k]:
                        if ki not in permission_node:
                            permission_node[ki] = set()
                        permission_node[ki].add(nodeid)
        return permission_node

    def count_permission(self, name, per_map):
        result = {}
        if name in per_map.keys():
            for p in self.permission:
                result[p] = 0
            pers = per_map[name]
            for per in pers:
                if per not in result.keys():
                    _settings.logger.debug(per + " not in permission list")
                    continue
                result[per] = 1
        return result

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

    def get_mappings(self, function, classname):
        mappings = {}
        nodeid, nodelabel = get_nodeid_label(self.G, function)
        if nodeid == "":
            _settings.logger.debug(function)
            _settings.logger.debug(nodelabel)
            return mappings
        # targets = self.G.successors(nodeid)
        classname = 'L' + classname.replace('<analysis.MethodAnalysis ', '') + ';'
        # debug(classname)
        funcList = []
        try:
            tmp = self.class_functions[classname]
            for t in tmp:
                funcList.append(self.method2nodeMap[t][1])
        except KeyError:
            funcList = []
        t = nodelabel
        t_id = nodeid
        external = get_external(nodeid, self.G)
        if external == 1:
            t_class = classname
            if not is_in_funcList(funcList, t):  # t is in super class or system function
                if t_class in self.super_dic.keys():
                    super_class = self.super_dic[t_class]
                    while True:
                        new_label = t.replace(t_class, super_class)
                        try:
                            super_funcList = self.class_functions[super_class]
                            if is_in_funcList(super_funcList, new_label):
                                mappings[t] = (new_label, t_id)
                                break
                            else:
                                t_class = super_class
                                super_class = self.super_dic[t_class]
                        except KeyError:
                            mappings[t] = (new_label, t_id)
                            break
        return mappings

    def substitude(self):  # Replace with subclass
        functions = self.node_attr.label
        ids = self.node_attr.id
        for c in self.class_functions: # start method connect run method
            if c in self.implement_dic:
                super_c = self.implement_dic[c]
            else:
                super_c = ""
            # print(super_c)
            if super_c in self.replacemap:
                # print(c, super_c)
                index = 0
                while index < len(ids):
                    func = functions.get(index)
                    if func.find(c + "-><init>(L") >= 0:
                        left = func.find(";-><init>(L") + len(";-><init>(L")
                        right = func.find(";", left)
                        baseclass = func[left: right]
                        # print("baseclass--->", baseclass) # MainActivity
                        # baseclass = getclass(func)
                        index2 = 0
                        func_list = self.replacemap[super_c]
                        while index2 < len(ids):
                            func_tmp = functions.get(index2)
                            for ftmp in func_list:
                                if func_tmp.find(c + "->" + ftmp) >= 0:
                                    self.class2runinit[baseclass].update({ids.get(index2): super_c[1:-1]})
                            index2 = index2 + 1
                        break
                    index = index + 1
        debug("class2init", self.class2runinit)
        mappings = {}
        index = 0
        while index < len(ids):
            label = functions.get(index)
            classname = getclass(label)
            mappings.update(self.get_mappings(label, classname))
            index = index + 1
        # for classname in self.class_functions:
        #     for function in self.class_functions[classname]:
        #         label = self.method2nodeMap[function][1]
        #         mappings.update(self.get_mappings(label, classname))
        # debug("mappings--->", mappings)
        per_map = get_from_csv_gml(_settings.api_file)
        res = {}
        for function in mappings:
            super_function = mappings[function][0]
            for func in per_map:
                if super_function.find(func) >= 0:
                    res[mappings[function][1]] = []
                    for p in per_map[func]:
                        res[mappings[function][1]].append(p)
                    # res[mappings[function][1]] = mappings[function][0]
        debug(res)
        return res  # All sensitive APIs replaced by subclasses
  
    def generate(self):
        per_map = get_from_csv_gml(_settings.api_file)
        result_f = create_csv(self.permission, self.path)
        self.node_attr = df_from_G(self.G)
        if type(self.node_attr) == bool and not self.node_attr:
            result_f.close()
            return 2
        getresolver = ";->getContentResolver()Landroid/content/ContentResolver;"
        functions = self.node_attr.label
        ids = self.node_attr.id
  
        substitude_permission = self.substitude()  # Subclasses involving sensitive APIs
        # Get contentprovider related permissions
        node_cp_permission = defaultdict(list)
        java_class = {}  # need to generate java file
        for i in range(len(ids)):
            function = functions.get(i)
            # debug(function)
            if function.find(getresolver) >= 0:
                node_id = ids.get(i)
                nodes = n_neighbor(node_id, self.G)
                debug(function, nodes)
                for node in nodes:
                    node_permission = self.deal_node(node)
                    if node_permission:
                        label = get_label(node, self.G)
                        left = label.find(' ')
                        right = label.find('->')
                        function_class = label[left + 1: right]
                        debug(function_class, node_permission)
                        java_class.update({function_class: node_permission})
        debug("java_class", java_class)
        for method in self.dexobj.get_methods():
            if str(method.get_class_name()) in java_class:
                current_class = self.dexobj.get_class(method.get_class_name())
                content = str(current_class.get_source())
                try:
                    node_permission = java_class.pop(method.get_class_name())
                except Exception:
                    _settings.logger.error("%s has error method name %s"%(self.path, method.get_class_name()))
                    continue
                if content.find('content://') >= 0:
                    for per in self.cp_permission.keys():
                        if content.find(per) >= 0:
                            pers = self.cp_permission[per]
                            for p in pers:
                                if p[0] in node_permission:
                                    for n_id in node_permission[p[0]]:
                                        node_cp_permission[n_id].append(p[1])
        debug("node_cp_permission", node_cp_permission)
        i = 0
        while i < len(ids):
            s = functions.get(i)
            s = node2function(s)
            p = self.count_permission(s, per_map)
            node_id = ids.get(i)
            if node_id in node_cp_permission:  # Permissions related to content providers
                for per in self.permission:
                    p[per] = 0
                for per in node_cp_permission[node_id]:
                    p[per] = 1
            if node_id in substitude_permission:  # Subclasses are sensitive APIs
                for per in self.permission:
                    p[per] = 0
                for per in substitude_permission[node_id]:
                    p[per] = 1
            if p != {}:
                write_csv(p, result_f, node_id)
                node_permission = []
                for k in p:
                    if p[k] == 1:
                        node_permission.append(k)
                self.sensitiveapimap.update({node_id: node_permission})
            i += 1
        result_f.close()
        return 0

    def getsensitive_api(self):  # Get the API list of all sensitive nodes of this apk
        return self.sensitiveapimap

    def getPermissionList(self):
        return self.permission

    def getClass2init(self):
        return self.class2runinit
```
عملا روند اجرای فرایند کلی به صورت زیر است:
```plaintext
cpermission.generate()
├── Create Permission CSV
├── Analyze Sensitive APIs
├── Analyze Content Providers
└── Update Sensitive API Map
       │
       ↓
Tpl.generate()
├── Analyze Template Usage
└── Generate Template CSV
       │
       ↓
Feature Extraction
├── Combine Permissions, Opcodes, Templates
└── Save Features for Analysis
       │
       ↓
Finish Analysis
├── Save Call Graph
└── Log Results
```

روند اجرای فرایند کلاس `Permission` و پس از آن به صورت زیر است:
```plaintext
AndroGen.analysis_app()
│
├── Initialize Permission Analysis:
│   ├── cpermission = Permission(...)
│   └── cpermission.generate()  <-- Start Permission Analysis
│
├── Permission.generate()
│   ├── Create CSV for permissions.
│   ├── Identify sensitive APIs.
│   ├── Map permissions to call graph nodes.
│   ├── Substitute sensitive APIs in subclasses or interfaces.
│   ├── Analyze Content Provider permissions.
│   └── Write results to CSV and update sensitive API map.
│
├── Permission.getClass2init()
│   └── Maps classes to their initializers or entry points (e.g., `run()` for `Runnable`).
│
├── Permission.getsensitive_api()
│   └── Retrieves a dictionary mapping call graph nodes to sensitive APIs.
│
│   ↓ Results passed to next stage
│
├── Template Analysis (`Tpl.generate()`)
│   ├── Initialize Template Analysis:
│   │   ├── Sensitive API map from `cpermission.getsensitive_api()`
│   │   ├── Class-to-initializer map from `cpermission.getClass2init()`
│   │   ├── Analyze sensitive template usage (e.g., overridden methods like `doInBackground`).
│   │   └── Generate CSV for template analysis.
│
│   ↓ Results passed to next stage
│
├── Feature Extraction
│   ├── Combine:
│   │   ├── Permissions (from `cpermission.generate()` output).
│   │   ├── Opcode counts.
│   │   ├── Sensitive APIs and call graph mappings.
│   │   └── Templates.
│   └── Save combined feature data for further use (e.g., CSVs or machine learning).
│
│   ↓
│
└── Finish Analysis
    ├── Log completion or errors.
    └── Prepare outputs (e.g., call graph, feature CSVs).
```


در ابتدای بررسی فایل `permission.py` چهار رشته به صورت زیر تعریف می‌گردند:
```python
# Permission includes the api and contentprovider parts
# Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder)
query_method = "Landroid/content/ContentResolver;->query(Landroid/net/Uri; [Ljava/lang/String; Ljava/lang/String; [Ljava/lang/String; Ljava/lang/String;)Landroid/database/Cursor;"
# Uri insert(Uri uri, ContentValues values)
insert_method = "Landroid/content/ContentResolver;->insert(Landroid/net/Uri; Landroid/content/ContentValues;)Landroid/net/Uri;"
# int delete(Uri uri, String selection, String[] selectionArgs)
delete_method = "Landroid/content/ContentResolver;->delete(Landroid/net/Uri; Ljava/lang/String; [Ljava/lang/String;)I"
# int update(Uri uri, ContentValues values, String selection, String[] selectionArgs)
update_method = "Landroid/content/ContentResolver;->update(Landroid/net/Uri; Landroid/content/ContentValues; Ljava/lang/String; [Ljava/lang/String;)I"
method_dic = {query_method: ['R'], insert_method: ['W'], delete_method: ['W'], update_method: ['W', 'R']}
```

در این خطوط از کد، ContentResolver API ها مورد بررسی قرار می‌گیرند و نوع دسترسی مدنظر هر کدام هم تعریف می‌گردد که در اینجا 'W' یا 'R' عنوان شده است. 4 API مختلف در اینجا وجود دارد که به هر کدام نوع دسترسی مدنظر را اطلاق کردیم. 
این کلاس از این قسمت به بعد شروع خواهد شد و تابع __init__ تعریف می‌گردد:
```python
class Permission():
    def __init__(self, G, path, class_functions, super_dic, implement_dic, dexobj, permission, cppermission, method2nodeMap):
        self.G = G  # call graph
        self.path = path  # csv path to save
        self.class_functions = class_functions
        self.super_dic = super_dic
        self.implement_dic = implement_dic
        self.dexobj = dexobj
        self.sensitiveapimap = {}  # All sensitive nodes in this apk and the permissions they involve
        self.class2runinit = defaultdict(dict)
        self.replacemap = {'Landroid/os/AsyncTask;': ['onPreExecute', 'doInBackground'],
                           'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']}
        self.permission = permission
        self.cp_permission = cppermission
        self.method2nodeMap = method2nodeMap
```

در کد بالا، یکی از موجودیت‌هایی که فلسفه نویی دارد، class2runinit است. این موجودیت یک نگاشت بین کلاس‌ها و نقاط ورودی (entry points) آن‌هاست. 

در کلاس `Permission` تابع اصلی، تابع `generate()` است. به همین دلیل در ابتدا از این تابع شروع می‌کنیم:
```python
    def generate(self):
        per_map = get_from_csv_gml(_settings.api_file)
        result_f = create_csv(self.permission, self.path)
        self.node_attr = df_from_G(self.G)
        if type(self.node_attr) == bool and not self.node_attr:
            result_f.close()
            return 2
        getresolver = ";->getContentResolver()Landroid/content/ContentResolver;"
        functions = self.node_attr.label
        ids = self.node_attr.id
  
        substitude_permission = self.substitude()  # Subclasses involving sensitive APIs
        # Get contentprovider related permissions
        node_cp_permission = defaultdict(list)
        java_class = {}  # need to generate java file
        for i in range(len(ids)):
            function = functions.get(i)
            # debug(function)
            if function.find(getresolver) >= 0:
                node_id = ids.get(i)
                nodes = n_neighbor(node_id, self.G)
                debug(function, nodes)
                for node in nodes:
                    node_permission = self.deal_node(node)
                    if node_permission:
                        label = get_label(node, self.G)
                        left = label.find(' ')
                        right = label.find('->')
                        function_class = label[left + 1: right]
                        debug(function_class, node_permission)
                        java_class.update({function_class: node_permission})
        debug("java_class", java_class)
        for method in self.dexobj.get_methods():
            if str(method.get_class_name()) in java_class:
                current_class = self.dexobj.get_class(method.get_class_name())
                content = str(current_class.get_source())
                try:
                    node_permission = java_class.pop(method.get_class_name())
                except Exception:
                    _settings.logger.error("%s has error method name %s"%(self.path, method.get_class_name()))
                    continue
                if content.find('content://') >= 0:
                    for per in self.cp_permission.keys():
                        if content.find(per) >= 0:
                            pers = self.cp_permission[per]
                            for p in pers:
                                if p[0] in node_permission:
                                    for n_id in node_permission[p[0]]:
                                        node_cp_permission[n_id].append(p[1])
        debug("node_cp_permission", node_cp_permission)
        i = 0
        while i < len(ids):
            s = functions.get(i)
            s = node2function(s)
            p = self.count_permission(s, per_map)
            node_id = ids.get(i)
            if node_id in node_cp_permission:  # Permissions related to content providers
                for per in self.permission:
                    p[per] = 0
                for per in node_cp_permission[node_id]:
                    p[per] = 1
            if node_id in substitude_permission:  # Subclasses are sensitive APIs
                for per in self.permission:
                    p[per] = 0
                for per in substitude_permission[node_id]:
                    p[per] = 1
            if p != {}:
                write_csv(p, result_f, node_id)
                node_permission = []
                for k in p:
                    if p[k] == 1:
                        node_permission.append(k)
                self.sensitiveapimap.update({node_id: node_permission})
            i += 1
        result_f.close()
        return 0
```

این تابع وظیفۀ اصلی تطابق گره‌های گراف با مجوزهای درخواستی را دارد.
در ابتدای کد داریم که:
```python
per_map = get_from_csv_gml(_settings.api_file)
```
در واقع این قسمت از کد در حال اتخاذ یک نگاشت بین APIهای حساس و مجوزهای مورد نیاز آن‌هاست. این نگاشت درون فایل `Data/APIs/API_all.csv` قرار دارد. قسمتی از خروجی این قسمت عبارت است از:
```bash
 {...'Lcom/android/internal/telephony/cdma/CdmaSMSDispatcher;->sendSubmitPdu(Lcom/android/internal/telephony/cdma/SmsMessage$SubmitPdu; Landroid/app/PendingIntent; Landroid/app/PendingIntent; Ljava/lang/String;)V': ['Permission:android.permission.WAKE_LOCK'], 'Landroid/media/AudioService$4;->onReceive(Landroid/content/Context; Landroid/content/Intent;)V': ['Permission:android.permission.WAKE_LOCK'], 'Landroid/media/AudioService;->onSendFinished(Landroid/app/PendingIntent; Landroid/content/Intent; I Ljava/lang/String; Landroid/os/Bundle;)V': ['Permission:android.permission.WAKE_LOCK'], 'Landroid/net/wifi/WifiStateMachine$DefaultState;->processMessage(Landroid/os/Message;)Z': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/mms/transaction/NotificationPlayer;->releaseWakeLock()V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/phone/CallerInfoCache$CacheAsyncTask;->releaseWakeLock()V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/systemui/media/NotificationPlayer;->releaseWakeLock()V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/contacts/voicemail/VoicemailPlaybackPresenter;->access$2200(Lcom/android/contacts/voicemail/VoicemailPlaybackPresenter; I I)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/internal/policy/impl/KeyguardViewMediator;->access$1100(Lcom/android/internal/policy/impl/KeyguardViewMediator; I)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/mms/transaction/NotificationPlayer;->access$700(Lcom/android/mms/transaction/NotificationPlayer;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/phone/BluetoothHandsfree;->access$3600(Lcom/android/phone/BluetoothHandsfree;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/phone/CallerInfoCache$CacheAsyncTask;->onCancelled(Ljava/lang/Void;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/phone/CallerInfoCache$CacheAsyncTask;->onPostExecute(Ljava/lang/Void;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/server/am/ActivityManagerService;->comeOutOfSleepIfNeededLocked()V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/systemui/media/NotificationPlayer;->access$700(Lcom/android/systemui/media/NotificationPlayer;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/phone/CallerInfoCache$CacheAsyncTask;->onCancelled(Ljava/lang/Object;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/phone/CallerInfoCache$CacheAsyncTask;->onPostExecute(Ljava/lang/Object;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/server/sip/SipService;->updateWakeLocks()V': ['Permission:android.permission.WAKE_LOCK'], 'Landroid/app/ActivityManagerNative;->setLockScreenShown(Z)V': ['Permission:android.permission.WAKE_LOCK'], 'Landroid/app/IActivityManager;->setLockScreenShown(Z)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/exchange/ExchangeService$AccountObserver;->access$700(Lcom/android/exchange/ExchangeService$AccountObserver;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/internal/policy/impl/KeyguardViewMediator;->updateActivityLockScreenState()V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/server/sip/SipService;->access$1300(Lcom/android/server/sip/SipService; Landroid/net/sip/SipProfile; I)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/server/sip/SipService;->notifyProfileRemoved(Landroid/net/sip/SipProfile;)V': ['Permission:android.permission.WAKE_LOCK'], 'Lcom/android/settings/accounts/ManageAccountsSettings;->requestOrCancelSyncForAccounts(Z)V': ['Permission:android.permission.READ_SYNC_SETTINGS'], 'Lcom/android/settings/accounts/ManageAccountsSettings;->onOptionsItemSelected(Landroid/view/MenuItem;)Z': ['Permission:android.permission.READ_SYNC_SETTINGS'], 'Lcom/android/settings/accounts/AccountPreferenceBase;->onOptionsItemSelected(Landroid/view/MenuItem;)Z': ['Permission:android.permission.READ_SYNC_SETTINGS'], 'Lcom/android/settings/accounts/AccountSyncSettings;->onOptionsItemSelected(Landroid/view/MenuItem;)Z': ['Permission:android.permission.READ_SYNC_SETTINGS'], 'Lcom/android/launcher2/Workspace$1;->run()V': ['Permission:android.permission.SET_WALLPAPER_HINTS'], 'Lcom/android/server/SerialService;->getSerialPorts()[Ljava/lang/String;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager$Stub$Proxy;->getSerialPorts()[Ljava/lang/String;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager$Stub;->getSerialPorts()[Ljava/lang/String;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager;->getSerialPorts()[Ljava/lang/String;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager$Stub;->onTransact(I Landroid/os/Parcel; Landroid/os/Parcel; I)Z': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/SerialManager;->getSerialPorts()[Ljava/lang/String;': ['Permission:android.permission.SERIAL_PORT'], 'Lcom/android/server/SerialService;->onTransact(I Landroid/os/Parcel; Landroid/os/Parcel; I)Z': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager$Stub$Proxy;->openSerialPort(Ljava/lang/String;)Landroid/os/ParcelFileDescriptor;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager$Stub;->openSerialPort(Ljava/lang/String;)Landroid/os/ParcelFileDescriptor;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/ISerialManager;->openSerialPort(Ljava/lang/String;)Landroid/os/ParcelFileDescriptor;': ['Permission:android.permission.SERIAL_PORT'], 'Landroid/hardware/SerialManager;->openSerialPort(Ljava/lang/String; I)Landroid/hardware/SerialPort;': ['Permission:android.permission.SERIAL_PORT'], 'Lcom/android/nfc/handover/HandoverManager;->createBluetoothOobDataRecord()Landroid/nfc/NdefRecord;': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/server/input/InputManagerService;->getDeviceAlias(Ljava/lang/String;)Ljava/lang/String;': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/nfc/handover/HandoverManager;->createHandoverRequestMessage()Landroid/nfc/NdefMessage;': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/nfc/handover/HandoverManager;->createHandoverSelectMessage(Z)Landroid/nfc/NdefMessage;': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/nfc/handover/ConfirmConnectActivity;->onCreate(Landroid/os/Bundle;)V': ['Permission:android.permission.BLUETOOTH'], 'Landroid/media/AudioService;->onSetA2dpConnectionState(Landroid/bluetooth/BluetoothDevice; I)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/settings/Settings$HeaderAdapter;-><init>(Landroid/content/Context; Ljava/util/List; Lcom/android/settings/accounts/AuthenticatorHelper;)V': ['Permission:android.permission.BLUETOOTH'], 'Landroid/media/AudioService;->access$6200(Landroid/media/AudioService; Landroid/bluetooth/BluetoothDevice; I)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/settings/Settings$KeyboardLayoutPickerActivity;->setListAdapter(Landroid/widget/ListAdapter;)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/nfc/handover/HandoverManager$HandoverPowerManager;->isBluetoothEnabled()Z': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/systemui/statusbar/policy/BluetoothController;-><init>(Landroid/content/Context;)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/BluetoothHandsfree$1;->onServiceConnected(I Landroid/bluetooth/BluetoothProfile;)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/BluetoothHandsfree;->access$800(Lcom/android/phone/BluetoothHandsfree; I Landroid/bluetooth/BluetoothDevice;)V': ['Permission:android.permission.BLUETOOTH'], 'Landroid/media/AudioService;->access$2200(Landroid/media/AudioService;)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/BluetoothHandsfree;->access$4800(Lcom/android/phone/BluetoothHandsfree; Landroid/bluetooth/BluetoothDevice;)I': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/settings/bluetooth/RequestPermissionActivity;->createDialog()V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/BluetoothHandsfree$ScoSocketConnectThread;->failedScoConnect()V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/InCallTouchUi;->access$300(Lcom/android/phone/InCallTouchUi; Lcom/android/internal/telephony/CallManager;)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/InCallTouchUi$2;->onAnimationStart(Landroid/animation/Animator;)V': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/phone/InCallScreen;->onDialerOpen(Z)V': ['Permission:android.permission.BLUETOOTH'], 'Landroid/server/BluetoothService;->notifyIncomingConnection(Ljava/lang/String; Z)Z': ['Permission:android.permission.BLUETOOTH'], 'Landroid/bluetooth/IBluetooth$Stub$Proxy;->notifyIncomingConnection(Ljava/lang/String; Z)Z': ['Permission:android.permission.BLUETOOTH'], 'Landroid/bluetooth/IBluetooth$Stub;->notifyIncomingConnection(Ljava/lang/String; Z)Z': ['Permission:android.permission.BLUETOOTH'], 'Landroid/bluetooth/IBluetooth;->notifyIncomingConnection(Ljava/lang/String; Z)Z': ['Permission:android.permission.BLUETOOTH'], 'Lcom/android/server/input/InputManagerService;->systemReady(Landroid/server/BluetoothService;)V': ['Permission:android.permission.BLUETOOTH']}
```

در خط بعدی از کد، یک csv برای ذخیرۀ permissionهای مربوط به هر گره ایجاد می‌گردد.
```python
result_f = create_csv(self.permission, self.path)
```
خط بعدی کد ویژگی‌های گره‌های گراف را استخراج و ذخیره می‌کند:
```python
self.node_attr = df_from_G(self.G)
```
آن‌ چیزی که درون `self.node_attr` ذخیره می‌گردد، عبارت است از:
```bash
        id                                              label  external  entrypoint  native  public  static             vm  codesize
0        0  <analysis.MethodAnalysis Landroid/support/v4/a...         0           0       0       1       0  8767136463272         0
1        1  <analysis.MethodAnalysis Landroid/support/v4/a...         0           0       0       1       1  8767136463272         4
2        2  <analysis.MethodAnalysis Landroid/support/v4/a...         0           0       0       1       0  8767136463272         0
3        3  <analysis.MethodAnalysis Landroid/support/v4/a...         0           0       0       1       1  8767136463272         4
4        4  <analysis.MethodAnalysis Landroid/support/v4/a...         0           0       0       1       0  8767136463272         0
...    ...                                                ...       ...         ...     ...     ...     ...            ...       ...
7954  7954  <analysis.MethodAnalysis Lcom/widget/view/Tall...         0           0       0       1       0  8767136463272        14
7955  7955  <analysis.MethodAnalysis Lcom/widget/view/Tall...         1           0       0       0       0              0         0
7956  7956  <analysis.MethodAnalysis Landroid/graphics/Pai...         1           0       0       0       0              0         0
7957  7957  <analysis.MethodAnalysis Landroid/graphics/Bit...         1           0       0       0       0              0         0
7958  7958  <analysis.MethodAnalysis Lcom/widget/view/Tall...         1           0       0       0       0              0         0
```
نوع این متغیر، `self.node_attr`، برابر است با:
```python
<class 'pandas.core.frame.DataFrame'>
```
 در ادامه متغیر زیر تعریف می‌گردد:
 ```python
 getresolver = ";->getContentResolver()Landroid/content/ContentResolver;"
```
دلیل تعریف این متغیر شناسایی فراخوانی‌های تابع `getContentResolver()` است. خروجی این تابع آبجکتی است که برای چهار عمل اصلی ContentProvider ها از آن‌ها استفاده می‌گردد.
در ادامه به کد زیر می‌رسیم:
```python
substitude_permission = self.substitude()
```
متدهای overridden یا جایگزین شده را در زیرکلاس‌ها یا کلاس‌هایی که رابط‌های پیاده‌سازی می‌کنند، شناسایی می‌کند. این متدها را ردیابی می‌کند تا آن‌ها را به همتایان حساس API خود در کلاس پایه یا رابط پیوند دهد. API های حساس را در این زیرکلاس ها به مجوزهای مربوطه فراخوانی می‌کند.

در ادامه خواهیم داشت:
```python
java_class = {}  # need to generate java file
        for i in range(len(ids)):
            function = functions.get(i)
            # debug(function)
            if function.find(getresolver) >= 0:
                node_id = ids.get(i)
                nodes = n_neighbor(node_id, self.G)
                debug(function, nodes)
                for node in nodes:
                    node_permission = self.deal_node(node)
                    if node_permission:
                        label = get_label(node, self.G)
                        left = label.find(' ')
                        right = label.find('->')
                        function_class = label[left + 1: right]
                        debug(function_class, node_permission)
                        java_class.update({function_class: node_permission})
```
برای این قسمت از کد، هدف این است که:

 گره‌هایی را در گراف فراخوانی که `getContentResolver()` را فراخوانی می‌کنند، شناسایی کنید.
 سپس عملیات بعدی (به عنوان مثال، پرس و جو، درج) انجام شده با استفاده از شی ContentResolver را شناسایی کنید.
 این عملیات را به مجوزهای مورد نیاز برای دسترسی ارائه دهنده محتوا ترسیم کنید.
 نتایج را در `java_class` ذخیره کنید، که نگاشت نام کلاس‌ها را به مجوزهای مورد نیاز آن‌ها ردیابی می‌کند.
 خروجی این قسمت چیزی شبیه به این خواهد بود:
 ```bash
 java_class = {
    "Lcom/example/MyClass;": {
        "query": "Permission:READ_CONTACTS",
        "insert": "Permission:WRITE_CONTACTS"
    },
    "Lcom/example/AnotherClass;": {
        "query": "Permission:READ_SMS"
    }
}
```

در ادامه به بررسی این قسمت از کد می‌پردازیم:
```python
for method in self.dexobj.get_methods():
            if str(method.get_class_name()) in java_class:
                current_class = self.dexobj.get_class(method.get_class_name())
                content = str(current_class.get_source())
                try:
                    node_permission = java_class.pop(method.get_class_name())
                except Exception:
                    _settings.logger.error("%s has error method name %s"%(self.path, method.get_class_name()))
                    continue
                if content.find('content://') >= 0:
                    for per in self.cp_permission.keys():
                        if content.find(per) >= 0:
                            pers = self.cp_permission[per]
                            for p in pers:
                                if p[0] in node_permission:
                                    for n_id in node_permission[p[0]]:
                                        node_cp_permission[n_id].append(p[1])
```

این قسمت از کد مسئول تجزیه و تحلیل متدهای کلاس جاوا در APK برای شناسایی تعاملات با ContentProviderها (به عنوان مثال، از طریق URIهایی مانند content://) و نگاشت آن‌ها با مجوزهای مورد نیاز است.
روی متدها در APK تکرار می‌شود تا آن‌هایی را که متعلق به کلاس‌هایی هستند که با ContentProviderها تعامل دارند، شناسایی کند.
کد منبع این کلاس‌ها را بررسی می کند تا URI های محتوا را شناسایی کند (به عنوان مثال، content://contacts).
سپس `Maps`، می‌آید و `content URI` ها را شناسایی می‌کند و به مجوزهای مربوطه تطابق داده و آن‌ها را با گره‌های گراف فراخوانی مرتبط می‌کند.
و بعد `node_cp_permission` را با مجوزهای شناسایی شده برای تعاملات ContentProvider به روز می‌کند.
خروجی این بخش که `node_cp_permission` است، شکلی شبیه به زیر است:
```bash
node_cp_permission = {
    10: ["Permission:READ_CONTACTS"],
    15: ["Permission:WRITE_SMS", "Permission:READ_SMS"]
}
```

بعد از آن داریم:
```python
i = 0
        while i < len(ids):
            s = functions.get(i)
            s = node2function(s)
            p = self.count_permission(s, per_map)
            node_id = ids.get(i)
            if node_id in node_cp_permission:  # Permissions related to content providers
                for per in self.permission:
                    p[per] = 0
                for per in node_cp_permission[node_id]:
                    p[per] = 1
            if node_id in substitude_permission:  # Subclasses are sensitive APIs
                for per in self.permission:
                    p[per] = 0
                for per in substitude_permission[node_id]:
                    p[per] = 1
            if p != {}:
                write_csv(p, result_f, node_id)
                node_permission = []
                for k in p:
                    if p[k] == 1:
                        node_permission.append(k)
                self.sensitiveapimap.update({node_id: node_permission})
            i += 1
```

این کد هر گره را در گراف فراخوانی (`self.G`) پردازش می‌کند تا مجوزها را به عملیات نشان‌داده‌شده توسط آن گره ترسیم کند. نتایج حاصل از تجزیه و تحلیل مجوز ContentProvider (`node_cp_permission`) و APIهای حساس جایگزین‌‎شده (`substitude_permission`) را برای ایجاد یک نقشه مجوز جامع ترکیب می‌کند.
در قسمت ابتدایی این حلقه داریم که:
```python
s = functions.get(i)
s = node2function(s)
```
 در ابتدا نام متد مربوط به گره `i` را می‌گیرد و سپس فرم استاندارد آن را خروجی می‌دهد. نتیجه این بخش می‌تواند مانند زیر باشد:
```bash
 Function get:  <analysis.MethodAnalysis Landroid/support/v4/app/BackStackState;-><init>(Landroid/os/Parcel;)V [access_flags=public constructor] @ 0x68274>
Node to Function:  Landroid/support/v4/app/BackStackState;-><init>(Landroid/os/Parcel;)V
```

در قسمت بعدی کد داریم که:
```python
p = self.count_permission(s, per_map)
```
این بخش از کد وجود مجوزهای یک گره را می‌سنجد و در `p` ذخیره می‌نماید. اگر باشد، 1 و اگر نباشد، 0 برای هر مجوز ذخیره می‌گردد:
```bash
p = {"Permission:READ_CONTACTS": 0, "Permission:SEND_SMS": 0}
```

```bash
{'Permission:android.car.permission.CAR_CAMERA': 0, 'Permission:android.car.permission.CAR_HVAC': 0, 'Permission:android.car.permission.CAR_MOCK_VEHICLE_HAL': 0, 'Permission:android.car.permission.CAR_NAVIGATION_MANAGER': 0, 'Permission:android.car.permission.CAR_PROJECTION': 0, 'Permission:android.car.permission.CAR_RADIO': 0, 'Permission:android.car.permission.CONTROL_APP_BLOCKING': 0, 'Permission:android.permission.ACCESS_ALL_DOWNLOADS': 0, 'Permission:android.permission.ACCESS_ALL_EXTERNAL_STORAGE': 0, 'Permission:android.permission.ACCESS_BLUETOOTH_SHARE': 0, 'Permission:android.permission.ACCESS_CACHE_FILESYSTEM': 0, 'Permission:android.permission.ACCESS_COARSE_LOCATION': 0, 'Permission:android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY': 0, 'Permission:android.permission.ACCESS_DOWNLOAD_MANAGER': 0, 'Permission:android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED': 0, 'Permission:android.permission.ACCESS_DRM': 0, 'Permission:android.permission.ACCESS_FINE_LOCATION': 0, 'Permission:android.permission.ACCESS_KEYGUARD_SECURE_STORAGE': 0, 'Permission:android.permission.ACCESS_LOCATION_EXTRA_COMMANDS': 0, 'Permission:android.permission.ACCESS_MOCK_LOCATION': 0, 'Permission:android.permission.ACCESS_NETWORK_STATE': 0, 'Permission:android.permission.ACCESS_NOTIFICATIONS': 0, 'Permission:android.permission.ACCESS_VOICE_INTERACTION_SERVICE': 0, 'Permission:android.permission.ACCESS_WIFI_STATE': 0, 'Permission:android.permission.ACCOUNT_MANAGER': 0, 'Permission:android.permission.ASEC_ACCESS': 0, 'Permission:android.permission.ASEC_CREATE': 0, 'Permission:android.permission.ASEC_DESTROY': 0, 'Permission:android.permission.ASEC_MOUNT_UNMOUNT': 0, 'Permission:android.permission.ASEC_RENAME': 0, 'Permission:android.permission.AUTHENTICATE_ACCOUNTS': 0, 'Permission:android.permission.BACKUP': 0, 'Permission:android.permission.BATTERY_STATS': 0, 'Permission:android.permission.BIND_APPWIDGET': 0, 'Permission:android.permission.BIND_CARRIER_SERVICES': 0, 'Permission:android.permission.BIND_DEVICE_ADMIN': 0, 'Permission:android.permission.BIND_VOICE_INTERACTION': 0, 'Permission:android.permission.BLUETOOTH': 0, 'Permission:android.permission.BLUETOOTH_ADMIN': 0, 'Permission:android.permission.BLUETOOTH_PRIVILEGED': 0, 'Permission:android.permission.BROADCAST_NETWORK_PRIVILEGED': 0, 'Permission:android.permission.BROADCAST_SCORE_NETWORKS': 0, 'Permission:android.permission.BROADCAST_STICKY': 1, 'Permission:android.permission.CACHE_CONTENT': 0, 'Permission:android.permission.CALL_PHONE': 0, 'Permission:android.permission.CALL_PRIVILEGED': 0, 'Permission:android.permission.CAMERA': 0, 'Permission:android.permission.CAPTURE_AUDIO_OUTPUT': 0, 'Permission:android.permission.CAPTURE_SECURE_VIDEO_OUTPUT': 0, 'Permission:android.permission.CAPTURE_TV_INPUT': 0, 'Permission:android.permission.CAPTURE_VIDEO_OUTPUT': 0, 'Permission:android.permission.CHANGE_APP_IDLE_STATE': 0, 'Permission:android.permission.CHANGE_COMPONENT_ENABLED_STATE': 0, 'Permission:android.permission.CHANGE_CONFIGURATION': 0, 'Permission:android.permission.CHANGE_DEVICE_IDLE_TEMP_WHITELIST': 0, 'Permission:android.permission.CHANGE_NETWORK_STATE': 0, 'Permission:android.permission.CHANGE_WIFI_MULTICAST_STATE': 0, 'Permission:android.permission.CHANGE_WIFI_STATE': 0, 'Permission:android.permission.CLEAR_APP_CACHE': 0, 'Permission:android.permission.CLEAR_APP_GRANTED_URI_PERMISSIONS': 0, 'Permission:android.permission.CLEAR_APP_USER_DATA': 0, 'Permission:android.permission.CONFIGURE_DISPLAY_COLOR_MODE': 0, 'Permission:android.permission.CONFIGURE_DISPLAY_COLOR_TRANSFORM': 0, 'Permission:android.permission.CONFIGURE_WIFI_DISPLAY': 0, 'Permission:android.permission.CONFIRM_FULL_BACKUP': 0, 'Permission:android.permission.CONNECTIVITY_INTERNAL': 0, 'Permission:android.permission.CONNECTIVITY_USE_RESTRICTED_NETWORKS': 0, 'Permission:android.permission.CONTROL_LOCATION_UPDATES': 0, 'Permission:android.permission.CONTROL_VPN': 0, 'Permission:android.permission.CRYPT_KEEPER': 0, 'Permission:android.permission.DELETE_CACHE_FILES': 0, 'Permission:android.permission.DELETE_PACKAGES': 0, 'Permission:android.permission.DEVICE_POWER': 0, 'Permission:android.permission.DISABLE_KEYGUARD': 0, 'Permission:android.permission.DOWNLOAD_CACHE_NON_PURGEABLE': 0, 'Permission:android.permission.DOWNLOAD_WITHOUT_NOTIFICATION': 0, 'Permission:android.permission.DUMP': 0, 'Permission:android.permission.DVB_DEVICE': 0, 'Permission:android.permission.EXPAND_STATUS_BAR': 0, 'Permission:android.permission.FILTER_EVENTS': 0, 'Permission:android.permission.FLASHLIGHT': 0, 'Permission:android.permission.FORCE_BACK': 0, 'Permission:android.permission.FORCE_STOP_PACKAGES': 0, 'Permission:android.permission.FRAME_STATS': 0, 'Permission:android.permission.FREEZE_SCREEN': 0, 'Permission:android.permission.GET_ACCOUNTS': 0, 'Permission:android.permission.GET_APP_GRANTED_URI_PERMISSIONS': 0, 'Permission:android.permission.GET_APP_OPS_STATS': 0, 'Permission:android.permission.GET_DETAILED_TASKS': 0, 'Permission:android.permission.GET_INTENT_SENDER_INTENT': 0, 'Permission:android.permission.GET_PACKAGE_SIZE': 0, 'Permission:android.permission.GET_PROCESS_STATE_AND_OOM_SCORE': 0, 'Permission:android.permission.GET_TASKS': 0, 'Permission:android.permission.GET_TOP_ACTIVITY_INFO': 0, 'Permission:android.permission.GLOBAL_SEARCH': 0, 'Permission:android.permission.GRANT_REVOKE_PERMISSIONS': 0, 'Permission:android.permission.GRANT_RUNTIME_PERMISSIONS': 0, 'Permission:android.permission.HDMI_CEC': 0, 'Permission:android.permission.INSTALL_DRM': 0, 'Permission:android.permission.INSTALL_GRANT_RUNTIME_PERMISSIONS': 0, 'Permission:android.permission.INSTALL_LOCATION_PROVIDER': 0, 'Permission:android.permission.INSTALL_PACKAGES': 0, 'Permission:android.permission.INTENT_FILTER_VERIFICATION_AGENT': 0, 'Permission:android.permission.INTERACT_ACROSS_USERS': 0, 'Permission:android.permission.INTERACT_ACROSS_USERS_FULL': 0, 'Permission:android.permission.INTERNAL_SYSTEM_WINDOW': 0, 'Permission:android.permission.INTERNET': 0, 'Permission:android.permission.KILL_BACKGROUND_PROCESSES': 0, 'Permission:android.permission.KILL_UID': 0, 'Permission:android.permission.LOCAL_MAC_ADDRESS': 0, 'Permission:android.permission.LOCATION_HARDWARE': 0, 'Permission:android.permission.MAGNIFY_DISPLAY': 0, 'Permission:android.permission.MANAGE_ACCOUNTS': 0, 'Permission:android.permission.MANAGE_ACTIVITY_STACKS': 0, 'Permission:android.permission.MANAGE_APP_OPS_RESTRICTIONS': 0, 'Permission:android.permission.MANAGE_APP_TOKENS': 0, 'Permission:android.permission.MANAGE_CA_CERTIFICATES': 0, 'Permission:android.permission.MANAGE_DEVICE_ADMINS': 0, 'Permission:android.permission.MANAGE_DOCUMENTS': 0, 'Permission:android.permission.MANAGE_FINGERPRINT': 0, 'Permission:android.permission.MANAGE_MEDIA_PROJECTION': 0, 'Permission:android.permission.MANAGE_NETWORK_POLICY': 0, 'Permission:android.permission.MANAGE_PROFILE_AND_DEVICE_OWNERS': 0, 'Permission:android.permission.MANAGE_SOUND_TRIGGER': 0, 'Permission:android.permission.MANAGE_USB': 0, 'Permission:android.permission.MANAGE_USERS': 0, 'Permission:android.permission.MANAGE_VOICE_KEYPHRASES': 0, 'Permission:android.permission.MARK_NETWORK_SOCKET': 0, 'Permission:android.permission.MEDIA_CONTENT_CONTROL': 0, 'Permission:android.permission.MODIFY_APPWIDGET_BIND_PERMISSIONS': 0, 'Permission:android.permission.MODIFY_AUDIO_ROUTING': 0, 'Permission:android.permission.MODIFY_AUDIO_SETTINGS': 0, 'Permission:android.permission.MODIFY_NETWORK_ACCOUNTING': 0, 'Permission:android.permission.MODIFY_PARENTAL_CONTROLS': 0, 'Permission:android.permission.MODIFY_PHONE_STATE': 0, 'Permission:android.permission.MOUNT_FORMAT_FILESYSTEMS': 0, 'Permission:android.permission.MOUNT_UNMOUNT_FILESYSTEMS': 0, 'Permission:android.permission.MOVE_PACKAGE': 0, 'Permission:android.permission.NFC': 0, 'Permission:android.permission.NOTIFY_PENDING_SYSTEM_UPDATE': 0, 'Permission:android.permission.OBSERVE_GRANT_REVOKE_PERMISSIONS': 0, 'Permission:android.permission.PACKAGE_USAGE_STATS': 0, 'Permission:android.permission.PACKAGE_VERIFICATION_AGENT': 0, 'Permission:android.permission.PACKET_KEEPALIVE_OFFLOAD': 0, 'Permission:android.permission.PEERS_MAC_ADDRESS': 0, 'Permission:android.permission.PERSISTENT_ACTIVITY': 0, 'Permission:android.permission.PROCESS_OUTGOING_CALLS': 0, 'Permission:android.permission.QUERY_DO_NOT_ASK_CREDENTIALS_ON_BOOT': 0, 'Permission:android.permission.READ_CALENDAR': 0, 'Permission:android.permission.READ_CALL_LOG': 0, 'Permission:android.permission.READ_CELL_BROADCASTS': 0, 'Permission:android.permission.READ_CONTACTS': 0, 'Permission:android.permission.READ_DREAM_STATE': 0, 'Permission:android.permission.READ_EXTERNAL_STORAGE': 0, 'Permission:android.permission.READ_FRAME_BUFFER': 0, 'Permission:android.permission.READ_LOGS': 0, 'Permission:android.permission.READ_NETWORK_USAGE_HISTORY': 0, 'Permission:android.permission.READ_PHONE_STATE': 0, 'Permission:android.permission.READ_PRECISE_PHONE_STATE': 0, 'Permission:android.permission.READ_PRIVILEGED_PHONE_STATE': 0, 'Permission:android.permission.READ_PROFILE': 0, 'Permission:android.permission.READ_SEARCH_INDEXABLES': 0, 'Permission:android.permission.READ_SMS': 0, 'Permission:android.permission.READ_SOCIAL_STREAM': 0, 'Permission:android.permission.READ_SYNC_SETTINGS': 0, 'Permission:android.permission.READ_SYNC_STATS': 0, 'Permission:android.permission.READ_USER_DICTIONARY': 0, 'Permission:android.permission.READ_WIFI_CREDENTIAL': 0, 'Permission:android.permission.REAL_GET_TASKS': 0, 'Permission:android.permission.REBOOT': 0, 'Permission:android.permission.RECEIVE_BLUETOOTH_MAP': 0, 'Permission:android.permission.RECEIVE_BOOT_COMPLETED': 0, 'Permission:android.permission.RECEIVE_MMS': 0, 'Permission:android.permission.RECEIVE_SMS': 0, 'Permission:android.permission.RECORD_AUDIO': 0, 'Permission:android.permission.RECOVERY': 0, 'Permission:android.permission.REGISTER_CONNECTION_MANAGER': 0, 'Permission:android.permission.REGISTER_WINDOW_MANAGER_LISTENERS': 0, 'Permission:android.permission.REMOTE_AUDIO_PLAYBACK': 0, 'Permission:android.permission.REMOVE_TASKS': 0, 'Permission:android.permission.REORDER_TASKS': 0, 'Permission:android.permission.RESET_FINGERPRINT_LOCKOUT': 0, 'Permission:android.permission.RESET_SHORTCUT_MANAGER_THROTTLING': 0, 'Permission:android.permission.RESTART_PACKAGES': 0, 'Permission:android.permission.RETRIEVE_WINDOW_INFO': 0, 'Permission:android.permission.REVOKE_RUNTIME_PERMISSIONS': 0, 'Permission:android.permission.SCORE_NETWORKS': 0, 'Permission:android.permission.SEND_RESPOND_VIA_MESSAGE': 0, 'Permission:android.permission.SEND_SMS': 0, 'Permission:android.permission.SEND_SMS_NO_CONFIRMATION': 0, 'Permission:android.permission.SERIAL_PORT': 0, 'Permission:android.permission.SET_ACTIVITY_WATCHER': 0, 'Permission:android.permission.SET_ALWAYS_FINISH': 0, 'Permission:android.permission.SET_ANIMATION_SCALE': 0, 'Permission:android.permission.SET_DEBUG_APP': 0, 'Permission:android.permission.SET_INPUT_CALIBRATION': 0, 'Permission:android.permission.SET_KEYBOARD_LAYOUT': 0, 'Permission:android.permission.SET_ORIENTATION': 0, 'Permission:android.permission.SET_POINTER_SPEED': 0, 'Permission:android.permission.SET_PREFERRED_APPLICATIONS': 0, 'Permission:android.permission.SET_PROCESS_LIMIT': 0, 'Permission:android.permission.SET_SCREEN_COMPATIBILITY': 0, 'Permission:android.permission.SET_TIME': 0, 'Permission:android.permission.SET_TIME_ZONE': 0, 'Permission:android.permission.SET_WALLPAPER': 0, 'Permission:android.permission.SET_WALLPAPER_COMPONENT': 0, 'Permission:android.permission.SET_WALLPAPER_HINTS': 0, 'Permission:android.permission.SHUTDOWN': 0, 'Permission:android.permission.SIGNAL_PERSISTENT_PROCESSES': 0, 'Permission:android.permission.START_ANY_ACTIVITY': 0, 'Permission:android.permission.START_TASKS_FROM_RECENTS': 0, 'Permission:android.permission.STATUS_BAR': 0, 'Permission:android.permission.STATUS_BAR_SERVICE': 0, 'Permission:android.permission.STOP_APP_SWITCHES': 0, 'Permission:android.permission.STORAGE_INTERNAL': 0, 'Permission:android.permission.SYSTEM_ALERT_WINDOW': 0, 'Permission:android.permission.TABLET_MODE': 0, 'Permission:android.permission.TABLET_MODE_LISTENER': 0, 'Permission:android.permission.TETHER_PRIVILEGED': 0, 'Permission:android.permission.TRANSMIT_IR': 0, 'Permission:android.permission.TV_INPUT_HARDWARE': 0, 'Permission:android.permission.UPDATE_APP_OPS_STATS': 0, 'Permission:android.permission.UPDATE_DEVICE_STATS': 0, 'Permission:android.permission.UPDATE_LOCK': 0, 'Permission:android.permission.USE_CREDENTIALS': 0, 'Permission:android.permission.USE_FINGERPRINT': 0, 'Permission:android.permission.USE_SIP': 0, 'Permission:android.permission.VIBRATE': 0, 'Permission:android.permission.WAKE_LOCK': 0, 'Permission:android.permission.WRITE_APN_SETTINGS': 0, 'Permission:android.permission.WRITE_CALENDAR': 0, 'Permission:android.permission.WRITE_CALL_LOG': 0, 'Permission:android.permission.WRITE_CONTACTS': 0, 'Permission:android.permission.WRITE_DREAM_STATE': 0, 'Permission:android.permission.WRITE_EXTERNAL_STORAGE': 0, 'Permission:android.permission.WRITE_PROFILE': 0, 'Permission:android.permission.WRITE_SECURE_SETTINGS': 0, 'Permission:android.permission.WRITE_SETTINGS': 0, 'Permission:android.permission.WRITE_SMS': 0, 'Permission:android.permission.WRITE_SOCIAL_STREAM': 0, 'Permission:android.permission.WRITE_SYNC_SETTINGS': 0, 'Permission:android.permission.WRITE_USER_DICTIONARY': 0, 'Permission:com.android.browser.permission.READ_HISTORY_BOOKMARKS': 0, 'Permission:com.android.browser.permission.WRITE_HISTORY_BOOKMARKS': 0, 'Permission:com.android.cts.permissionNormal': 0, 'Permission:com.android.cts.permissionNotUsedWithSignature': 0, 'Permission:com.android.cts.permissionWithSignature': 0, 'Permission:com.android.email.permission.ACCESS_PROVIDER': 0, 'Permission:com.android.email.permission.READ_ATTACHMENT': 0, 'Permission:com.android.gallery3d.filtershow.permission.READ': 0, 'Permission:com.android.gallery3d.filtershow.permission.WRITE': 0, 'Permission:com.android.gallery3d.permission.GALLERY_PROVIDER': 0, 'Permission:com.android.launcher.permission.READ_SETTINGS': 0, 'Permission:com.android.launcher.permission.WRITE_SETTINGS': 0, 'Permission:com.android.launcher3.permission.READ_SETTINGS': 0, 'Permission:com.android.launcher3.permission.WRITE_SETTINGS': 0, 'Permission:com.android.printspooler.permission.ACCESS_ALL_PRINT_JOBS': 0, 'Permission:com.android.providers.imps.permission.READ_ONLY': 0, 'Permission:com.android.providers.imps.permission.WRITE_ONLY': 0, 'Permission:com.android.providers.tv.permission.READ_EPG_DATA': 0, 'Permission:com.android.providers.tv.permission.WRITE_EPG_DATA': 0, 'Permission:com.android.rcs.eab.permission.READ_WRITE_EAB': 0, 'Permission:com.android.server.telecom.permission.REGISTER_PROVIDER_OR_SUBSCRIPTION': 0, 'Permission:com.android.voicemail.permission.ADD_VOICEMAIL': 0, 'Permission:getWindowToken': 0, 'Permission:temporaryEnableAccessibilityStateUntilKeyguardRemoved': 0, 'Permission:ti.permission.FMRX': 0, 'Permission:ti.permission.FMRX_ADMIN': 0}
```

در ادامه هم یک دیکشنری تنظیم می‌گردد که بسیار مهم است و آن `self.sensitiveapimap` است. عملا یک دیکشنری جهانی همه گره‌ها را در گراف فراخوانی به مجوزهای API حساس مرتبط با آنها نگاشت می‌کند.
مقدار این متغیر برای یک اپلیکیشن تست برابر است با:
```bash
Sensitive API Map:  {558: ['Permission:android.permission.BROADCAST_STICKY'], 999: ['Permission:android.permission.BROADCAST_STICKY'], 1495: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1496: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1499: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1500: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1503: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1504: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1506: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1507: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1509: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1510: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1513: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 1517: ['Permission:android.permission.ACCESS_NETWORK_STATE'], 2988: ['Permission:android.permission.READ_CONTACTS', 'Permission:android.permission.READ_SOCIAL_STREAM', 'Permission:android.permission.READ_USER_DICTIONARY', 'Permission:android.permission.VIBRATE', 'Permission:android.permission.WRITE_CONTACTS'], 2989: ['Permission:android.permission.READ_CONTACTS', 'Permission:android.permission.READ_SOCIAL_STREAM', 'Permission:android.permission.READ_USER_DICTIONARY', 'Permission:android.permission.VIBRATE', 'Permission:android.permission.WRITE_CONTACTS'], 2994: ['Permission:android.permission.READ_CONTACTS', 'Permission:android.permission.READ_SOCIAL_STREAM', 'Permission:android.permission.READ_USER_DICTIONARY', 'Permission:android.permission.VIBRATE', 'Permission:android.permission.WRITE_CONTACTS'], 2995: ['Permission:android.permission.READ_CONTACTS', 'Permission:android.permission.READ_SOCIAL_STREAM', 'Permission:android.permission.READ_USER_DICTIONARY', 'Permission:android.permission.VIBRATE', 'Permission:android.permission.WRITE_CONTACTS'], 3704: ['Permission:android.permission.ACCESS_NETWORK_STATE', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED'], 3757: ['Permission:android.permission.ACCESS_NETWORK_STATE', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED'], 3772: ['Permission:android.permission.ACCESS_NETWORK_STATE', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED'], 3775: ['Permission:android.permission.INTERNET', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED'], 3777: ['Permission:android.permission.ACCESS_NETWORK_STATE', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED'], 3780: ['Permission:android.permission.ACCESS_NETWORK_STATE', 'Permission:android.permission.RECEIVE_BOOT_COMPLETED'], 3824: ['Permission:android.permission.READ_PHONE_STATE'], 3825: ['Permission:android.permission.ACCESS_COARSE_LOCATION', 'Permission:android.permission.ACCESS_FINE_LOCATION'], 3875: ['Permission:android.permission.INTERNET'], 3882: ['Permission:android.permission.INTERNET'], 3935: ['Permission:android.permission.VIBRATE'], 3945: ['Permission:android.permission.READ_PHONE_STATE'], 3947: ['Permission:android.permission.ACCESS_WIFI_STATE'], 5343: ['Permission:android.permission.READ_PHONE_STATE'], 5345: ['Permission:android.permission.READ_PHONE_STATE'], 5350: ['Permission:android.permission.GET_TASKS'], 5624: ['Permission:android.permission.BROADCAST_STICKY'], 5626: ['Permission:android.permission.BROADCAST_STICKY'], 5849: ['Permission:android.permission.INTERNET'], 6708: ['Permission:android.permission.INTERNET'], 6709: ['Permission:android.permission.INTERNET']}
```

## بازگشتی به کلاس `AndroGen`
در کلاس `AndroGen`، برای استفاده از کلاس `Permission` داریم:
```python
cpermission = Permission(G=G, path=permissionFilename, class_functions=class_functions, super_dic=super_dic,implement_dic=implement_dic, dexobj=dexobj, permission=self.permission, cppermission=self.cppermission, method2nodeMap=method2nodeMap)
cpermission.generate()
class2init = cpermission.getClass2init()
sensitiveapimap = cpermission.getsensitive_api()
```
کارکرد تابع `getClass2init`، بازیابی نگاشت کلاس‌ها به متدهای اولیه یا نقطه ورودی (به عنوان مثال، constructureها یا متدهای کلیدی مانند run() در Runnable) است. 
خروجی getClass2init()
متد یک دیکشنری با ساختار زیر برمی گرداند:
 کلید: نام کلاس (به عنوان مثال، Lcom/example/MyClass؛)
 مقدار مقابل کلید: نگاشت دیکشنری دیگر:
 شناسه گره: شناسه گره گراف فراخوانی که متد را نشان می‌دهد.
 نام روش: نام متد اولیه یا نقطه ورود.
```bash
{
    "Lcom/example/MyClass;": {
        12: "Lcom/example/MyClass;-><init>()V",  # Constructor
        15: "Lcom/example/MyClass;->run()V"      # Runnable entry point
    },
    "Lcom/example/MyOtherClass;": {
        20: "Lcom/example/MyOtherClass;->doWork()V"
    }
} 
```

### **کلاس Tpl چیست؟**

کلاس **`Tpl`** بخشی از چارچوب تحلیل است که برای بررسی **گره‌های حساس الگو** در گراف تماس (Call Graph) یک فایل APK طراحی شده است. **گره‌های حساس الگو** به گره‌هایی اشاره دارند که در آن‌ها از الگوها یا قالب‌های رایج فریم‌ورک اندروید (مانند `AsyncTask`، `Handler`، یا `Thread`) استفاده شده و اغلب با APIهای حساس ترکیب می‌شوند.

---

### **هدف از کلاس Tpl**

این کلاس برای موارد زیر طراحی شده است:

1. **تحلیل استفاده از الگوهای حساس**:
    - ردیابی قالب‌های اندروید مانند `AsyncTask`، `Handler` یا `Runnable` و تعاملات متدهای آن‌ها (مانند `doInBackground`، `handleMessage`، یا `run`).
2. **تشخیص فراخوانی‌های API حساس**:
    - لینک دادن APIهای حساس به این الگوها برای شناسایی ریسک‌های امنیتی.
3. **یکپارچه‌سازی با تحلیل مجوزها**:
    - تولید خروجی شامل مجوزهای مورد نیاز برای عملیات حساس شناسایی‌شده.
4. **تولید داده‌های ویژگی**:
    - خروجی به صورت ساختار‌یافته در قالب CSV تولید می‌شود که شامل الگوها، APIهای حساس و مجوزها است.

----
### **اجزای کلیدی**

#### **سازنده (`__init__`)**

```python
def __init__(self, tpl_list, G, outpath, sensitiveapimap, permission, class2init, deepth):
```

1. **`tpl_list`**:
    
    - لیستی از نام پکیج‌ها یا الگوهایی که باید ردیابی شوند (مانند `AsyncTask`، `Handler`).
    - اگر به صورت لیست ارائه نشود، از متد `get_pkg()` برای شناسایی آن‌ها استفاده می‌شود.
2. **`G`**:
    
    - گراف تماس که تعاملات متدهای فایل APK را نشان می‌دهد.
3. **`outpath`**:
    
    - مسیر ذخیره فایل خروجی CSV.
4. **`sensitiveapimap`**:
    
    - یک نگاشت کلی از گره‌های گراف تماس به APIهای حساس و مجوزهای مورد نیاز آن‌ها.
5. **`permission`**:
    
    - لیستی از تمام مجوزهایی که باید ردیابی شوند.
6. **`class2init`**:
    
    - نگاشت کلاس‌ها به متدهای ابتدایی یا نقاط ورودی آن‌ها (مانند `run()`).
7. **`replacemap`**:
    
    - تعریف نگاشت متدهای فریم‌ورک به متدهای بازنویسی‌شده یا نقاط ورودی.
    - مثال:
        
        ```python
        replacemap = {
            'android/os/AsyncTask;->execute': ('android/os/AsyncTask;->onPreExecute', 'android/os/AsyncTask;->doInBackground'),
            'android/os/Handler;->sendMessage': ('android/os/Handler;->handleMessage'),
            'java/lang/Thread;->start': ('java/lang/Runnable;->run')
        }
        ```
        
8. **`deepth`**:
    
    - عمق جستجو در گراف تماس برای عملیات حساس.

---

### **متدهای کلیدی**

#### **1. `get_pkg(apk_path, ratio=0.6)`**

- شناسایی نام پکیج‌ها یا کتابخانه‌های استفاده‌شده در فایل APK با استفاده از **LibRadarLite**.
- فیلتر کردن پکیج‌ها بر اساس نسبت تطبیق.

---

#### **2. `dfs(nodeid)`**

اجرای یک **جستجوی عمق-اول (DFS)** از یک گره در گراف تماس برای:

- شناسایی تمام گره‌های قابل‌دسترسی تا عمق مشخص‌شده.
- ردیابی متدهای بازنویسی‌شده یا جایگزین‌شده با استفاده از `replacemap`.

**خروجی مثال**:

```python
leafs = {10, 12, 15}  # گره‌های دسترسی‌پذیر در طول جستجوی DFS
```

---

#### **3. `getTplSensitiveNode(nodeid)`**

- فراخوانی `dfs(nodeid)` برای پیمایش گراف تماس.
- فیلتر کردن گره‌های برگ که با APIهای حساس در `sensitiveapimap` مرتبط هستند.

**خروجی مثال**:

```python
TplSensitiveNodes = {15, 20}
```

---

#### **4. `writefile()`**

نوشتن نتایج گره‌های حساس الگو به فایل خروجی CSV:

- **سطرها**: گره‌های حساس الگو.
- **ستون‌ها**: مجوزها.
- **مقادیر**: فراوانی هر مجوز برای گره‌های حساس.

**خروجی مثال**:

```csv
node_id,READ_CONTACTS,SEND_SMS
10,0.5,0.2
```

---

#### **5. `generate()`**

نقطه اصلی ورود برای تحلیل:

1. استخراج ویژگی‌ها (`id`، `label`) از گره‌های گراف تماس.
2. پیمایش گره‌ها و بررسی اینکه آیا آن‌ها به الگوهای ردیابی‌شده (`tpl_list`) تعلق دارند.
3. برای هر گره تطبیق‌یافته، فراخوانی `getTplSensitiveNode()` برای شناسایی گره‌های برگ حساس.
4. به‌روزرسانی `TplSensitiveNodeMap` با گره‌های شناسایی‌شده.
5. فراخوانی `writefile()` برای ذخیره نتایج.

---

### **جریان اجرایی**

#### **ورودی**

- گراف تماس `G`.
- لیست الگوهای ردیابی‌شده (`tpl_list`).
- نقشه APIهای حساس (`sensitiveapimap`).

#### **مراحل**

1. **شناسایی گره‌های الگو**:
    - تطبیق گره‌ها در گراف تماس با الگوها (مانند `AsyncTask`، `Handler`).
2. **DFS برای APIهای حساس**:
    - پیمایش از هر گره الگو برای شناسایی APIهای حساس در گره‌های قابل‌دسترسی.
3. **تجمیع مجوزها**:
    - گره `10`: `READ_CONTACTS`.
    - گره `15`: `SEND_SMS`.
4. **ذخیره نتایج**:
    - ذخیره مجوزهای تجمیع‌شده برای هر گره حساس الگو در فایل CSV.

---

### **نمونه گردش کار**

#### **گراف تماس ورودی**

گره‌ها:

1. `Lcom/example/MyTask;->doInBackground()`
2. `Lcom/example/MyHandler;->handleMessage()`
3. `Lcom/example/MyRunnable;->run()`

#### **الگوهای ردیابی‌شده**

- `AsyncTask`
- `Handler`

#### **اجرای مراحل**

1. **DFS**:
    - شروع از گره‌هایی مانند `AsyncTask.execute` و پیمایش تا متدهایی مانند `doInBackground()`.
2. **تشخیص APIهای حساس**:
    - شناسایی فراخوانی‌های API حساس مانند:
        - `ContentResolver.query()` → نیاز به `READ_CONTACTS`.
3. **تجمیع مجوزها**:
    - گره `10`: `READ_CONTACTS`.
    - گره `15`: `SEND_SMS`.

#### **خروجی CSV**

```csv
node_id,READ_CONTACTS,SEND_SMS
10,1,0
15,0,1
```

---

### **اهمیت کلاس Tpl**

1. **ردیابی عملیات حساس الگوها**:
    
    - شناسایی و تحلیل الگوهای رایج اندروید مانند `AsyncTask`، `Handler`، و `Runnable`.
2. **ارتباط الگوها با APIهای حساس**:
    
    - ردیابی نحوه تعامل این الگوها با عملیات حساس.
3. **تسهیل تحلیل امنیتی**:
    
    - شناسایی سوءاستفاده یا استفاده بیش از حد از APIهای حساس در الگوهای مبتنی بر قالب‌ها.
4. **تولید داده‌های ویژگی**:
    
    - خروجی ساختار‌یافته برای تحلیل بیشتر یا مدل‌های یادگیری ماشین.

---

### **نتیجه‌گیری**

کلاس **`Tpl`** نقش مهمی در ارتباط الگوهای اندروید با استفاده از APIهای حساس ایفا می‌کند. این کلاس با استفاده از گراف تماس، نگاشت مجوزها و تحلیل APIهای حساس، دید جامعی از عملیات حساس مبتنی بر الگو در فایل APK ارائه می‌دهد. خروجی آن برای ارزیابی امنیت، تحلیل رفتار و استخراج ویژگی‌ها حیاتی است.

## اتمام تولید ویژگی‌ها
با استخراج tplها، ویژگی‌های مربوط به گراف استخراج شدند. حال به تابع `generate_behavior_subgraph` می‌پردازیم. 
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

در ادامه داریم که:
```python
call_graphs.sort()
```
در واقع مسیر این گراف‌های فراخوانی به صورت الفبایی مرتب می‌گردد.
در ادامه باید تابع `generate_graph` بررسی گردد.

# تابع `generate_graph`
جریان اجرا در این تابع به صورت زیر است:
```bash
generate_graph()
   ├── Compute `exp_dir`
   ├── Initialize `MyOwnDataset`
   │     ├── Process raw `.gml` files
   │     ├── Generate subgraphs for each call graph
   │     ├── Map APIs and prune TPL nodes if required
   │     └── Save subgraphs as `.pt` files
   └── Return control
```

و کد این تابع به صورت زیر است:
```python
def generate_graph(call_graphs, output_dir, apk_base, db_name, label, hop=2, tpl=True, training=False, api_map=False):
    exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}' if training else osp.join(output_dir, db_name)
    MyOwnDataset(root=exp_dir, label=label, tpl=tpl, hop=hop, db=db_name, base_dir=apk_base, apks=call_graphs, api_map=api_map)
```

در ابتدا متغیر `exp_dir` محاسبه می‌گردد که گراف‌ها و زیرگراف‌ها(نتایج) در این دایرکتوری ذخیره می‌گردند. یعنی در واقع برای training، این دایرکتوری شامل دیتاستی از گراف‌هاست.

پس از آن `MyOwnDataset` را داریم که فرایند استخراج زیرگراف‌ها را شامل می‌شود. این کلاس باعث ایجاد دیتاست می‌گردد و تنظیمات ایجاد این دیتاست با تنظیم ویژگی‌ها کلیدی `tpl`، `hop`، `label` و `api_map` است.
نتایج پردازش گراف‌ها و زیرگراف‌ها در آدرسی به فرمت زیر ذخیره می‌گردند:
```bash
./training/Graphs/<db_name>/HOP_<hop>/TPL_<tpl>/processed/data_<apk_id>_<subgraph_id>.pt
```

## بررسی MyOwnDataset
کلاس MyOwnDataset که در فایل behavior_subgraph.py تعریف شده است، یک کلاس داده‌ای سفارشی برای مدیریت پردازش و ذخیره‌سازی داده‌های مبتنی بر گراف است. این کلاس از Dataset در PyTorch Geometric به عنوان پایه استفاده می‌کند و متد initialization (`__init__`) آن عملیات مختلفی را برای تولید، پردازش و مدیریت زیرگراف‌ها از گراف‌های فراخوانی ورودی انجام می‌دهد.
خروجی‌ها در قالب دادۀ PyG (فایل‌های .pt) ذخیره می‌گردند.

کد فایل `behavior_hraph.py` در زیر آمده است:
```python
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
import torch
from torch_geometric.data import Dataset
import logging
import glob
import os
import os.path as osp
import time
from tqdm import tqdm

from graph.subgraph import api_subgraph
from utils import makedirs, find_all_apk

from functools import partial
from multiprocessing import Pool, cpu_count
import asyncio


class MyOwnDataset(Dataset):
    def __init__(self, root, tpl, hop, db, base_dir, transform=None, pre_transform=None, label=1, apks=None, layer=None, api_map=False):
        self.lens = 0
        self.samples = 0
        self.label = label
        self.base_dir = base_dir
        self.tpl = tpl
        self.hop = hop
        self.db = db
        self.apks = apks
        self.api_map = api_map
        if apks is None:
            self.layer = layer
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        if self.apks is None:
            db = self.db # self.root.split('/')[1]
            db_record = f'{self.base_dir}/{db}.csv'
            if not osp.isfile(db_record):
                print('[GraphDroid] Searching db for `.gml` files.')
                self.apks = get_db_gml(base_dir=self.base_dir, db=db, layer=self.layer)
                self.apks.to_csv(db_record, header=False, index=None)
            else:
                print(f'[GraphDroid] Read existing data csv: {db_record}')
                self.apks = pd.read_csv(db_record, header=None)[0]
        else:
            self.apks = pd.Series(self.apks)
        return self.apks

    @property
    def processed_file_names(self):
        r'''The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing.'''
        ex_map = f'./mappings/{self.db}_2_True.csv'
        if osp.exists(ex_map):
            print(f'[GraphDroid] Read existing mapping csv: {ex_map}')
            df = pd.read_csv(ex_map)
            return [f'data_{v.graph_id}_{v.subgraph_id}.pt' for _,v in df.iterrows()]
        return [f'data_{i}_0.pt' for i in range(len(self.apks))]
  
    def _exclude_exists(self):
        print('[GraphDroid] Finding break points...')
        graph_ids = []
        apps = self.apks
        for i in tqdm(range(len(apps))):
            data_file = f'{self.root}/processed/data_{i}_0.pt'
            if osp.exists(data_file):
                a = torch.load(data_file).app
                # '''
                gml = apps[apps.str.contains(a)]
                assert len(gml) == 1
                apps = apps.drop(gml.index)
                # '''
                '''
                a_api = glob.glob(f'{self.root}/processed/data_{i}_*.pt')
                if len(pd.read_csv(apps[i].replace(f'{self.db}/decompile', f'{self.db}/result/permission').replace('/call.gml', '.csv'))) == len(a_api):
                    gml = apps[apps.str.contains(a)]
                    assert len(gml) == 1
                    apps = apps.drop(gml.index)
                    # print(f'[GraphDroid] Found {gml.item()}')
                else:
                    graph_ids.append(i)
                    for api in a_api:
                        os.remove(api)
                '''
            else:
                graph_ids.append(i)
  
        assert len(graph_ids) >= len(apps)
        return apps, graph_ids
    def _process(self):
        def files_exist(files):
            return len(files) != 0 and all([osp.exists(f) for f in files])
        apps = self.raw_paths
        if files_exist(self.processed_paths):
            print(f'[GraphDroid] Data found in `{self.root}/processed`. Skip processing.')
        else:
            if glob.glob(f'{self.root}/processed/data_*.pt'):
                apps, graph_ids = self._exclude_exists()
            else:
                graph_ids = list(range(len(apps)))
            makedirs(self.processed_dir)
            self.process(apps, graph_ids)
            if self.api_map:
                from graph.dbmap import form_dataset
                max_gid, _ = self.len()
                form_dataset(self.db, self.hop, self.tpl, max_gid)
            tqdm.write(f'[GraphDroid] Data generated in `{self.root}/processed/`.')
  
    def get_sep(self, example):
        fwords = ['permission','opcode']
        flens = get_feature(example, fwords=fwords, getsep=True, base_dir=self.base_dir)
        with open(f'{self.root}/FeatureLen.txt', 'w') as f:
            f.write(str(flens))

    def process(self, apps, graph_ids):
        print("apps content:", apps)
        print("apps index:", apps.index)
        self.get_sep(apps[0])
        apps = apps.sort_values()
        zip_args = list(zip(apps, graph_ids))
        logging.info(f'Processing {len(zip_args)} apps...')

        partial_func = partial(process_apk_wrapper, label=self.label, tpl=self.tpl, hop=self.hop, base_dir=self.base_dir, processed_dir=self.processed_dir) # fixed params
        self.samples, self.lens = mp_process(partial_func, zip_args)
        logging.info(f'Total app samples: {self.samples}, total behavior subgraphs: {self.lens}')

    def len(self):
        if not(self.samples & self.lens):
            pt_files = glob.glob(f"{self.processed_dir}/data_*.pt")
            self.lens = len(pt_files)
            gids = []
            for p in pt_files:
                gids.append(int(p.split('data_')[-1].split('_')[0]))
            self.samples = max(gids)
        return self.samples, self.lens
  
    def get(self, graph_id, subgraph_id):
        data = torch.load(osp.join(self.processed_dir, 'data_{}_{}.pt'.format(graph_id, subgraph_id)))
        return data


def mp_process(func, argument_list):
    num_pool = int(cpu_count() / 8)
    print('Number of pools:', num_pool)
    glen = 0
    slen = 0
    pool = Pool(processes=num_pool)
    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    T1 = time.process_time()
    for job in tqdm(jobs, desc='[GraphDroid GraphGenerator]'):
        gl, sl = job.get()
        glen += gl
        slen += sl
    pool.join()
    T2 = time.process_time()
    logging.info(f'[Timer] {T2-T1}')
    return glen, slen
  
def process_apk_wrapper(*args, **kwargs): # multiple arguments
    label = kwargs['label']
    tpl = kwargs['tpl']
    hop = kwargs['hop']
    base_dir = kwargs['base_dir']
    processed_dir = kwargs['processed_dir']
    app = args[0]
    graph_id = args[1]
  
    flag = 0
    num_subgraph = 0
    logging.info(app)
    try:
        data_list = asyncio.run(gml2Data(app, label, tpl=tpl, hop=hop, base_dir=base_dir))
        dlen = len(data_list)
        if dlen:
            for i in range(dlen):
                data = data_list[i]
                data_path = osp.join(processed_dir, 'data_{}_{}.pt'.format(graph_id,i))
                assert not osp.exists(data_path)
                torch.save(data, data_path)
                num_subgraph += 1
            flag = 1
        logging.info(f'[Success] {app}')
    except Exception:
        logging.exception(f'{app}')
    finally:
        return flag, num_subgraph
  

def get_feature(gmlfile, base_dir, fwords=['permission','opcode','tpl'], getsep=False):
    feature_file = gmlfile.replace(f'{base_dir}/decompile/', f'{base_dir}/result/%s/').replace('/call.gml', '.csv')
    print(feature_file)
    features = [feature_file % i for i in fwords]
    '''
    [node type]
        external: ('undefined'=0),'permission'=1
        'opcode'=2, 'tpl'=3
    '''
    if getsep:
        return [pd.read_csv(features[i]).shape[1]-1 for i in range(len(features))]
    return [pd.read_csv(features[i]).assign(type=i+1) for i in range(len(features))]


def convert_subgraph_edge(edge_index, feature_df, p, map_only=False):
    mapping = {int(row['id']):index for index,row in feature_df.iterrows()}
    center = mapping[p] # new
    if map_only:
        return mapping # (old, new)

    result=[]
    for l in edge_index:
        rep = [mapping[x] for x in l]
        result.append(rep)

    return result, center, mapping
  
async def prepare_file(gmlfile, base_dir, fwords):
    single_graph = nx.read_gml(gmlfile, label='id')
    x = get_feature(gmlfile, base_dir, fwords)
    return single_graph, x
  
async def generate_behavior_subgraph(p, features, single_graph, hop, debug, gmlfile, apk_name, y):
    nodes_type = features[['id', 'type']]
    subgraph_nodes, subgraph_edges, apimap = api_subgraph(p, single_graph, nodes_type, hop=hop, debug=debug)

    if len(subgraph_nodes) <= 1:
        logging.warning(f'[IsolateNode] {gmlfile}: isolated node@{p}')
        return None

    subtypes = nodes_type[nodes_type['id'].isin(subgraph_nodes)]
    subgraph_features = features[features.id.isin(subgraph_nodes)].reset_index(drop=True)
    assert subgraph_features.shape[0]==len(subgraph_nodes)

    edges = subgraph_edges # [(source, target, key), ...]
    edges_df = pd.DataFrame(edges).iloc[:,:-1].T

    edge_list, center, m = convert_subgraph_edge(edges_df.values.tolist(), subgraph_features, p)
    assert len(apimap)==len(m)
    mapping = [apimap[i] for i in m]
    labels = [subtypes[subtypes.id==i].type.tolist()[0] for i in m]

    data = Data(x=torch.tensor(subgraph_features.iloc[:,1:-1].values.tolist(), dtype=torch.float)
                , edge_index=torch.tensor(edge_list, dtype=torch.long)
                , y=torch.tensor([y], dtype=torch.long)
                , num_nodes=len(subgraph_nodes), labels=labels
                , center=center, mapping=mapping, app=apk_name)
    return data 

async def gml2Data(gmlfile, y, base_dir, tpl=True, sub=True, hop=2, debug=False):
    fwords = ['permission','opcode','tpl'] if tpl else ['permission','opcode']
    single_graph, x = await prepare_file(gmlfile, base_dir, fwords)
    apk_name = gmlfile.split('/decompile/')[-1][:-9]
    all_nodes = pd.DataFrame(single_graph.nodes,columns=['id'])
    permission, opcodes = x[:2]
    if tpl:
        opcodes = pd.merge(opcodes, x[2], how='outer', on='id', suffixes=['_','']).drop(['type_'],axis=1)
        opcodes['type'] = opcodes['type'].fillna(2)
        opcodes = opcodes.fillna(0)
    features_exist = pd.merge(permission.astype('float'), opcodes, how='outer').fillna(0).drop_duplicates('id', keep='first')   # keep type = 1
    features = pd.merge(all_nodes, features_exist, how='outer').fillna(0)
    features['type'] = features['type'].astype('int')

    p_list = x[0].id.tolist()
    data_list = []
    if sub:
        tasks = []
        for p in p_list:
            partial_func = partial(generate_behavior_subgraph, features=features, single_graph=single_graph, hop=hop, debug=debug, gmlfile=gmlfile, apk_name=apk_name, y=y)
            tasks.append(partial_func(p))
        data_list = await asyncio.gather(*tasks)
        while None in data_list:
            data_list.remove(None)
    else:
        nodes = single_graph.nodes
        edges = single_graph.edges # [(source, target, key), ...]
        edge_list = pd.DataFrame(edges).iloc[:,:-1].T.values.tolist()
        data = Data(x=torch.tensor(features.iloc[:,1:].values.tolist(), dtype=torch.long)
                    , edge_index=torch.tensor(edge_list, dtype=torch.long)
                    , y=torch.tensor([y], dtype=torch.long) # y:list
                    , num_nodes=len(nodes))
        data_list.append(data)
    return data_list

def get_db_gml(base_dir, db='Drebin', check=False, layer=None):
    base = f'{base_dir}/{db}/'
    gmls = []
    if check:
        apks = find_all_apk(osp.join(base, db), end='.apk', layer=layer)
        for a in apks:
            rpath = a.split(db)[-1].split('.apk')[0]
            gmls.append(base+'decompile'+'%s/call.gml' % rpath)
        gmls = check_gml(gmls)
    else:
        gmls = find_all_apk(osp.join(base, 'decompile'), end='.gml', layer=layer)
    return pd.Series(gmls)

def check_gml(gmllist):
    tmp = []
    for a in gmllist:
        if not osp.exists(a):
            logging.warning(f'[NoGML] {a}')
        else:
            tmp.append(a)
    return tmp

if __name__ == '__main__':
    makedirs('loggings'); makedirs('mappings')

    import argparse
    parser = argparse.ArgumentParser(description='GraphDroid Data Generator.')
    parser.add_argument('db', type=str, help='Choose a decompiled APK dataset.')
    parser.add_argument('--tpl', type=str, default=True, help='Simpilfy third party library API nodes.')
    parser.add_argument('--hop', type=int, default=2, help='Subgraph based on k hop neighborhood.')
    parser.add_argument('--label', type=int, default=None, help='Dataset label: 1 for Malicious, 0 for Benign.')
    parser.add_argument('--base', type=str, default=None, help='Call graph and feature files directory.')
    parser.add_argument('--layer', type=int, default=1, help='Speed up gml searching.')
    args = parser.parse_args()

    LOG_FORMAT = '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s - %(message)s'
    current_milli_time = lambda: int(round(time.time() * 1000))
    db = args.db
    tpl = args.tpl
    hop = args.hop
    exp_dir = f'./Datasets/{db}/HOP_{hop}/TPL_{tpl}'
    makedirs(exp_dir)
    logging.basicConfig(filename=f'./loggings/[HOP_{hop}-TPL_{tpl}-{db}]{current_milli_time()}.log', level=logging.INFO, format=LOG_FORMAT)
    logging.debug(exp_dir)
    if args.label is None:
        try:
            db_labels = {'Drebin':1, 'Genome':1, 'AMD':1, 'Benign':0}
            label = db_labels[db.split('_')[0]]
        except Exception:
            logging.error('Label must be specified for unkown dataset')
    else:
        label = args.label

    layer = None if args.layer < 0 else args.layer
    dataset = MyOwnDataset(root=exp_dir, label=label, tpl=tpl, hop=hop, db=db, base_dir=args.base, layer=layer)
```

در ابتدا بهتر است که به کلاس `Dataset` بپردازیم. کلاس `Dataset` در PyTorch Geometric (PyG) یک کلاس پایه انتزاعی برای ایجاد و مدیریت مجموعه داده‌های گراف است. این کلاس به منظور ساده‌سازی مدیریت، پردازش و بارگذاری داده‌های گراف برای وظایفی مانند طبقه‌بندی گراف، طبقه‌بندی گره و پیش‌بینی لینک طراحی شده است.
### مروری بر کلاس Dataset در PyG

کلاس Dataset به‌عنوان یک چارچوب برای موارد زیر عمل می‌کند:
- **مدیریت داده‌های خام**: مدیریت فایل‌های گراف خام یا فرمت‌های دیگر.
- **پردازش داده‌ها**: تبدیل داده‌های خام به اشیای گراف در قالب Data متعلق به PyG.
- **بارگذاری داده‌ها**: فراهم کردن دسترسی آسان به گراف‌های تکی یا دسته‌ها در حین آموزش یا ارزیابی.

### متدهای کلیدی در کلاس `Dataset`
1. مقداردهی اولیه (`__init__`)
متد `__init__` با تعریف ویژگی‌ها و مسیرهای ضروری مجموعه داده را تنظیم می‌کند.
```python
class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
```
- **`root`**: مسیری که مجموعه داده در آن ذخیره یا پردازش می‌شود.
- **`transform`**: تبدیل‌های اختیاری که هنگام بارگذاری بر هر شیء گراف اعمال می‌شوند.
- **`pre_transform`**: تبدیل‌های اختیاری که قبل از ذخیره داده‌های پردازش‌شده اعمال می‌شوند (مانند نرمال‌سازی ویژگی‌ها).

#### متدهای `raw_file_names` و `processed_file_names`

- **`raw_file_names`**: لیستی از فایل‌های ورودی خام موردنیاز برای پردازش را بازمی‌گرداند.
- **`processed_file_names`**: لیستی از فایل‌های پردازش‌شده را که بعداً بارگذاری می‌شوند بازمی‌گرداند.

```python
@property
def raw_file_names(self):
    return ['graph1.gml', 'graph2.gml']

@property
def processed_file_names(self):
    return ['data_0.pt', 'data_1.pt']
```

#### 3. `download`
اگر مجموعه داده به‌صورت محلی موجود نباشد، این متد آن را دانلود می‌کند.
```python
def download(self):
    # دانلود مجموعه داده از یک URL
    download_url('http://example.com/dataset.zip', self.raw_dir)
    extract_zip(self.raw_dir)
```

#### 4. `process`
متد `process` داده‌های خام را به اشیای پردازش‌شده `Data` در PyG تبدیل می‌کند. این مرحله مهم‌ترین بخش است.
```python
def process(self):
    for raw_path in self.raw_paths:
        # بارگذاری داده‌های خام (مانند GML یا JSON)
        graph = nx.read_gml(raw_path)
        # تبدیل به قالب Data در PyG
        data = Data(
            x=torch.tensor(...),   # ویژگی‌های گره‌ها
            edge_index=torch.tensor(...),  # شاخص‌های یال‌ها
            y=torch.tensor(...)    # برچسب‌ها
        )
        # ذخیره داده‌های پردازش‌شده
        torch.save(data, self.processed_paths[index])
```
#### 5. `len`
تعداد گراف‌های موجود در مجموعه داده را بازمی‌گرداند.
```python
def len(self):
    return len(self.processed_file_names)
```

#### 6. `get`
یک شیء گراف تکی را از مجموعه داده پردازش‌شده بارگذاری و بازمی‌گرداند.
```python
def get(self, idx):
    return torch.load(self.processed_paths[idx])
```

### نمونه پیاده‌سازی یک مجموعه داده در PyG
```python
from torch_geometric.data import Dataset, Data

class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['graph1.gml', 'graph2.gml']

    @property
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt']

    def process(self):
        for i, raw_path in enumerate(self.raw_paths):
            # تبدیل گراف خام به قالب Data در PyG
            graph = nx.read_gml(raw_path)
            data = Data(
                x=torch.tensor(...),  # ویژگی‌های گره‌ها
                edge_index=torch.tensor(...),  # شاخص‌های یال‌ها
                y=torch.tensor(...)  # برچسب‌ها
            )
            torch.save(data, self.processed_paths[i])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])

dataset = MyDataset(root='./data')
```

### کمی با transform و pre_transform
در PyTorch Geometric (PyG)، `transform` و `pre_transform` مکانیزم‌هایی هستند که به شما امکان می‌دهند تغییراتی را بر روی اشیای داده‌ی گراف (`Data`) اعمال کنید، چه قبل از استفاده و چه در حین استفاده. این تغییرات معمولاً برای پیش‌پردازش داده‌ها، تقویت ویژگی‌ها یا تغییر ساختار گراف‌ها به کار می‌روند.
### 1. `transform`
- **زمان اعمال**: در زمان اجرا، هر بار که یک شیء داده بارگذاری می‌شود (مانند حین آموزش یا استنتاج).
- **هدف**: به صورت پویا تغییراتی را بر روی هر شیء گراف در هنگام بازیابی آن از مجموعه داده اعمال می‌کند.
#### نمونه موارد استفاده:
1. نرمال‌سازی ویژگی‌های گره‌ها.
2. حذف تصادفی یال‌ها (افزایش داده).
3. تبدیل داده‌های گراف به قالب خاص مورد نیاز مدل.

مثال:
```python
from torch_geometric.transforms import NormalizeFeatures

dataset = MyDataset(root='./data', transform=NormalizeFeatures())
# ویژگی‌های هر گراف در هنگام دسترسی نرمال‌سازی می‌شوند.
```

### **2. `pre_transform`**

- **زمان اعمال**: در مرحله‌ی `process`، قبل از ذخیره داده‌های پردازش‌شده.
- **هدف**: تغییراتی را یک بار بر روی داده‌های خام اعمال کرده و داده‌های تغییر یافته را برای استفاده‌ی مجدد ذخیره می‌کند.
#### نمونه موارد استفاده:
1. اضافه کردن ویژگی‌های جدید به گره‌ها یا یال‌ها.
2. هرس کردن گراف (مانند حذف گره‌ها یا یال‌های غیرضروری).
3. محاسبه خصوصیات گراف در سطح کل مانند مقادیر ویژه لاپلاسین.

```python
from torch_geometric.transforms import AddSelfLoops

dataset = MyDataset(root='./data', pre_transform=AddSelfLoops())
# یال‌های خود حلقه‌ای به تمام گراف‌ها در طول پیش‌پردازش اضافه می‌شوند.
```

### **نحوه استفاده از آن‌ها**

#### **مثال `transform`**
```python
from torch_geometric.transforms import RandomLinkSplit

# تقسیم پویا یال‌ها به مجموعه‌های آموزش، تست و اعتبارسنجی
dataset = MyDataset(root='./data', transform=RandomLinkSplit())
```

مثال `pre_transform`
```python
from torch_geometric.transforms import RemoveIsolatedNodes

# پیش‌پردازش داده برای حذف گره‌های جدا شده قبل از ذخیره
dataset = MyDataset(root='./data', pre_transform=RemoveIsolatedNodes())
```

### **ترکیب `transform` و `pre_transform`**

می‌توانید از هر دو `transform` و `pre_transform` در یک مجموعه داده استفاده کنید:

- از `pre_transform` برای اعمال پیش‌پردازش دائمی در هنگام ایجاد اولیه مجموعه داده استفاده کنید.
- از `transform` برای اعمال افزایش یا تنظیمات پویا در حین آموزش یا استنتاج استفاده کنید.

## بررسی گام به گام  کلاس `MyOwnDataset(Dataset)`
کلاس `MyOwnDataset` یک پیاده‌سازی سفارشی از کلاس `Dataset` در PyTorch Geometric است. در ادامه، مرحله‌ی مقداردهی اولیه (`__init__`) را به صورت گام‌به‌گام توضیح می‌دهیم.
### 1. سازنده: `__init__`
```python
class MyOwnDataset(Dataset):
    def __init__(self, root, tpl, hop, db, base_dir, transform=None, pre_transform=None, label=1, apks=None, layer=None, api_map=False):
        self.lens = 0            # تعداد کل زیرگراف‌های پردازش‌شده را ردیابی می‌کند
        self.samples = 0         # تعداد گراف‌های موجود در مجموعه داده را ردیابی می‌کند
        self.label = label       # برچسب مجموعه داده (مثلاً 1 برای بدافزار، 0 برای برنامه‌های سالم)
        self.base_dir = base_dir # مسیر پایه برای داده‌های خام و پردازش‌شده
        self.tpl = tpl           # ساده‌سازی گره‌های کتابخانه‌های شخص ثالث در صورت True بودن
        self.hop = hop           # شعاع همسایگی برای استخراج زیرگراف
        self.db = db             # نام پایگاه داده (مانند Drebin، AMD)
        self.apks = apks         # لیستی از APKها برای پردازش، در صورت ارائه
        self.api_map = api_map   # در صورت True بودن، نگاشت API به گره‌های گراف را انجام می‌دهد
        if apks is None:
            self.layer = layer   # پارامتر لایه برای کشف سریع‌تر فایل‌های APK
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
```

#### **ویژگی‌های کلیدی:**

- **`root`**: مسیر ریشه که مجموعه داده در آن ذخیره می‌شود.
- **`tpl`**: آیا گره‌های کتابخانه‌های شخص ثالث ساده شوند یا خیر.
- **`hop`**: اندازه همسایگی k-hop برای تولید زیرگراف.
- **`db`**: نام پایگاه داده (مثلاً "Drebin" یا "AMD").
- **`base_dir`**: مسیر دایرکتوری که فایل‌های خام `.gml` و ویژگی‌ها را نگه می‌دارد.
- **`transform`**: تابعی برای تغییر پویا که در زمان اجرا به هر گراف اعمال می‌شود.
- **`pre_transform`**: تابعی برای پیش‌پردازش که فقط یک بار در مرحله‌ی `process` اعمال می‌شود.
- **`label`**: برچسب مجموعه داده، معمولاً برای مشخص کردن بدافزار (1) یا برنامه‌های سالم (0).
- **`apks`**: لیستی از مسیرهای APK برای پردازش. اگر `None` باشد، کلاس به دنبال آن‌ها می‌گردد.
- **`layer`**: پارامتر اختیاری برای کشف سریع‌تر فایل‌های `.gml`.
- **`api_map`**: آیا نگاشت API به گره‌های گراف در زیرگراف‌ها گنجانده شود یا خیر.

### **2. جریان مقداردهی اولیه**

#### 1. تنظیم ویژگی‌ها

- سازنده تمام پارامترهای ارائه‌شده را به‌عنوان ویژگی‌های نمونه تنظیم می‌کند (`self.label`, `self.tpl` و غیره).
- ویژگی‌هایی مانند `self.lens` و `self.samples` برای ردیابی آمار مجموعه داده مقداردهی می‌شوند.

#### 2. مقداردهی اولیه کلاس والد

- فراخوانی `super().__init__(root, transform, pre_transform)` کلاس والد `Dataset` را مقداردهی می‌کند.
    - مسیرهای داده خام و پردازش‌شده را تنظیم می‌کند:
        - `self.raw_dir`: مسیر داده‌های خام.
        - `self.processed_dir`: مسیر داده‌های پردازش‌شده.
    - مدیریت مناسب تبدیل‌ها از طریق `transform` و `pre_transform` را تضمین می‌کند.

### 3. نمونه استفاده

#### ایجاد نمونه از مجموعه داده
```python
dataset = MyOwnDataset(
    root='./data',
    tpl=True,
    hop=2,
    db='Androzoo',
    base_dir='./datasets',
    label=1,
    transform=NormalizeFeatures(),
    pre_transform=AddSelfLoops()
)
```

## بررسی `raw_paths`
متد `raw_paths` در کلاس `MyOwnDataset` یک **ویژگی (property)** است که لیستی از فایل‌های ورودی خام (مثلاً فایل‌های `.gml`) را که باید برای ایجاد مجموعه داده پردازش شوند، ارائه می‌دهد. این متد بخشی حیاتی از جریان کاری کلاس `Dataset` در PyTorch Geometric است، زیرا مشخص می‌کند مجموعه داده چگونه داده‌های خام خود را مکان‌یابی و مدیریت کند.
```python
@property
def raw_paths(self):
    """
    فایل‌های خام مورد نیاز برای تبدیل به فرمت PyTorch Geometric Data را بازمی‌گرداند.
    """
    if self.apks is None:
        db = self.db  # استفاده از نام پایگاه داده
        db_record = f'{self.base_dir}/{db}.csv'  # مسیر فایل CSV که لیست فایل‌های `.gml` را دارد
        if not osp.isfile(db_record):  # اگر فایل وجود نداشته باشد
            print('[GraphDroid] Searching db for `.gml` files.')
            self.apks = get_db_gml(base_dir=self.base_dir, db=db, layer=self.layer)  # جستجوی فایل‌های `.gml`
            self.apks.to_csv(db_record, header=False, index=None)  # ذخیره مسیرها برای استفاده مجدد
        else:
            print(f'[GraphDroid] Read existing data csv: {db_record}')
            self.apks = pd.read_csv(db_record, header=None)[0]  # بارگذاری مسیرهای `.gml` از فایل CSV
    else:
        self.apks = pd.Series(self.apks)  # اگر `apks` ارائه شده باشد، مستقیماً از آن استفاده می‌شود
    return self.apks
```

1. **هدف**:
    
    - این متد فایل‌های ورودی خام (مانند `.gml`) مورد نیاز برای پردازش و ایجاد مجموعه داده را شناسایی می‌کند.
    - بررسی می‌کند که آیا لیستی از فایل‌های خام از قبل وجود دارد یا خیر و در صورت عدم وجود، آن‌ها را به صورت پویا کشف می‌کند.
2. **مراحل**:
    
    - **اگر `self.apks` برابر `None` باشد**:
        
        - بررسی می‌کند که آیا فایل CSV (`db_record`) شامل لیست مسیرهای فایل‌های `.gml` وجود دارد یا خیر.
        - اگر فایل CSV وجود نداشته باشد:
            - تابع `get_db_gml` برای جستجوی فایل‌های `.gml` در دایرکتوری پایگاه داده (`self.base_dir`) فراخوانی می‌شود.
            - مسیرهای پیدا شده در یک فایل CSV ذخیره می‌شوند.
        - اگر فایل CSV وجود داشته باشد:
            - مسیرهای فایل‌های `.gml` از CSV خوانده می‌شوند.
    - **اگر `self.apks` ارائه شده باشد**:
        
        - مستقیماً از لیست داده شده به‌عنوان مسیر فایل‌های خام استفاده می‌شود.
3. **خروجی**:
    - یک سری از مسیرهای فایل‌های `.gml` در قالب `Pandas Series`.

خروجی این قسمت به صورت زیر است که یک `Pandas Series` شامل آدرس call graph ها است:
```bash
0    /home/user/MsDroid2/MsDroid-main/src/Output/Te...
1    /home/user/MsDroid2/MsDroid-main/src/Output/Te...
2    /home/user/MsDroid2/MsDroid-main/src/Output/Te...
```

پس از اجرای متد raw_paths و بدست آوردن آدرس فایل‌های `.gml`، متد `processed_file_names` اجرا می‌شود که آیا این فایل‌‌های خام پردازش شده‌اند یا نه!
## بررسی `processed_file_names`
متد **`processed_file_names`** در کلاس `MyOwnDataset` یک ویژگی (property) است که نام فایل‌های پردازش‌شده (مثلاً فایل‌های `.pt`) را که مجموعه داده انتظار دارد در دایرکتوری `processed` وجود داشته باشند، مشخص می‌کند. این فایل‌ها در مرحله‌ی `process()` مجموعه داده ایجاد می‌شوند و برای بارگذاری داده‌های پیش‌پردازش‌شده به صورت کارآمد استفاده می‌شوند.
```python
@property
def processed_file_names(self):
    """
    نام فایل‌های پردازش‌شده `.pt` در دایرکتوری processed را بازمی‌گرداند.
    این فایل‌ها پس از پردازش فایل‌های خام `.gml` تولید می‌شوند.
    """
    ex_map = f'./mappings/{self.db}_2_True.csv'
    if osp.exists(ex_map):
        print(f'[GraphDroid] Read existing mapping csv: {ex_map}')
        df = pd.read_csv(ex_map)
        return [f'data_{v.graph_id}_{v.subgraph_id}.pt' for _, v in df.iterrows()]
    return [f'data_{i}_0.pt' for i in range(len(self.apks))]
```

این متد لیستی از فایل‌های `.pt` را که باید در دایرکتوری `processed` وجود داشته باشند، مشخص می‌کند. این فایل‌ها به زیرگراف‌هایی مربوط هستند که از فایل‌های خام `.gml` در مرحله‌ی `process()` ایجاد شده‌اند.
به دنبال یک فایل CSV (`./mappings/{self.db}_2_True.csv`) می‌گردد که مشخص می‌کند کدام فایل‌های پردازش‌شده مورد نیاز هستند.
اگر فایل وجود داشته باشد:
- فایل CSV را به یک DataFrame می‌خواند.
- نام فایل‌ها را بر اساس ستون‌های `graph_id` و `subgraph_id` در فایل CSV می‌سازد.
در مثال ما نام این فایل csv عبارت است از:
```bash
./mappings/Test_DB_2_True.csv
```

اگر فایل CSV وجود نداشته باشد:
- لیستی پیش‌فرض از نام فایل‌ها تولید می‌کند که فرض می‌کند هر فایل خام `.gml` یک زیرگراف تولید می‌کند:
```python
['data_0_0.pt', 'data_1_0.pt', ..., 'data_N_0.pt']
```
## متد `process_`
بعد از **`processed_file_names`** در کلاس `MyOwnDataset` متد **`process_`** است که مسئول پردازش داده‌های خام به فرمت `Data` در PyTorch Geometric و ذخیره آن‌ها به‌صورت فایل‌های پردازش‌شده است.
این متد بررسی می‌کند که آیا فایل‌های پردازش‌شده از قبل وجود دارند یا خیر. در صورت عدم وجود، فایل‌های خام را به زیرگراف‌ها پردازش می‌کند. زیرگراف‌های پردازش‌شده را به صورت فایل‌های `.pt` در دایرکتوری `processed` ذخیره می‌کند.
```python
def _process(self):
    def files_exist(files):
        return len(files) != 0 and all([osp.exists(f) for f in files])

    apps = self.raw_paths  # دریافت مسیر فایل‌های خام `.gml`
    if files_exist(self.processed_paths):  # بررسی وجود تمام فایل‌های پردازش‌شده
        print(f'[GraphDroid] Data found in `{self.root}/processed`. Skip processing.')
    else:
        if glob.glob(f'{self.root}/processed/data_*.pt'):  # اگر برخی فایل‌های پردازش‌شده وجود داشته باشند
            apps, graph_ids = self._exclude_exists()  # حذف فایل‌های پردازش‌شده از پردازش بیشتر
        else:
            graph_ids = list(range(len(apps)))  # اختصاص ID‌های منحصربه‌فرد به فایل‌های خام

        makedirs(self.processed_dir)  # اطمینان از وجود دایرکتوری `processed`
        self.process(apps, graph_ids)  # پردازش فایل‌های خام به فایل‌های پردازش‌شده
        
        if self.api_map:  # در صورت فعال بودن: نگاشت API به گره‌ها
            from graph.dbmap import form_dataset
            max_gid, _ = self.len()  # محاسبه تعداد فایل‌های پردازش‌شده
            form_dataset(self.db, self.hop, self.tpl, max_gid)
        
        tqdm.write(f'[GraphDroid] Data generated in `{self.root}/processed/`.')
```

زمانی که به دنبال وجود یا عدم وجود `self.processed_paths` می‌گردد، این مقدار برای مثال ما که یک دیتابیس 3 تایی است، برابر است با:
```bash
['training/Graphs/Test_DB/HOP_2/TPL_True/processed/data_0_0.pt', 'training/Graphs/Test_DB/HOP_2/TPL_True/processed/data_1_0.pt', 'training/Graphs/Test_DB/HOP_2/TPL_True/processed/data_2_0.pt']
```
چرا که در نظر دارد اگر پردازشی اتفاق افتاده باشد، برای هر گراف، حداقل یک زیرگراف تولید شده است و آن هم زیرگراف `data_N_0` است.
اگر برخی فایل‌های پردازش‌شده وجود داشته باشند (`data_*.pt`)، متد **`_exclude_exists`** فایل‌های خامی را که هنوز پردازش نشده‌اند شناسایی می‌کند.
اگر هیچ فایل پردازش‌شده‌ای وجود نداشته باشد، به تمام فایل‌های خام ID‌های منحصربه‌فرد اختصاص می‌دهد.

پس از بررسی فایل‌های پردازش موجود و بررسی وجود دایرکتوری، متد `process` فراخوانی می‌گردد.
## بررسی متد `process`
متد **`process`** در کلاس `MyOwnDataset` بخش حیاتی از چرخه حیات مجموعه داده است. این متد وظیفه تبدیل فایل‌های داده خام (مانند فایل‌های `.gml` که نمایانگر گراف‌های فراخوانی هستند) به اشیای داده‌ای پردازش‌شده در قالب `Data` در PyTorch Geometric و ذخیره آن‌ها در دایرکتوری `processed` را بر عهده دارد.
```python
def process(self, apps, graph_ids):
    print("apps content:", apps)
    print("apps index:", apps.index)
    
    # بررسی طول ویژگی‌ها برای اشکال‌زدایی یا تحلیل
    self.get_sep(apps[0])
    
    # مرتب‌سازی فایل‌های خام برای یکپارچگی در پردازش
    apps = apps.sort_values()
    zip_args = list(zip(apps, graph_ids))  # جفت کردن هر اپ با ID گراف مربوطه
    
    logging.info(f'Processing {len(zip_args)} apps...')
    
    # تابع جزئی برای پردازش فایل‌های APK به‌صورت جداگانه
    partial_func = partial(
        process_apk_wrapper, 
        label=self.label, 
        tpl=self.tpl, 
        hop=self.hop, 
        base_dir=self.base_dir, 
        processed_dir=self.processed_dir
    )
    
    # استفاده از پردازش موازی برای پردازش فایل‌های APK
    self.samples, self.lens = mp_process(partial_func, zip_args)
    
    logging.info(f'Total app samples: {self.samples}, total behavior subgraphs: {self.lens}')
```

پارامترهای ورودی:
- آرگومان `apps`: لیست یا سری Pandas شامل مسیرهای فایل‌های خام `.gml` برای پردازش.
- آرگومان `graph_ids`: شناسه‌های منحصربه‌فرد اختصاص داده شده به هر فایل گراف خام برای نام‌گذاری فایل‌های پردازش‌شده.

در ابتدای این متد، تابع `get_sep` اجرا می‌گردد. کد این تابع:
```python
def get_sep(self, example):
    fwords = ['permission', 'opcode']  # Define feature types to analyze
    flens = get_feature(example, fwords=fwords, getsep=True, base_dir=self.base_dir)
    with open(f'{self.root}/FeatureLen.txt', 'w') as f:
        f.write(str(flens))
```
ورودی این تابع آدرس یک فایل `.gml` است و خروجی آن فایل `FeatureLen.txt` است که ابعاد ویژگی‌ها را عنوان می‌کند:
```bash
[length_of_permissions, length_of_opcodes]
```

در ادامۀ متد `process`، یک لیست از تاپل‌ها ایجاد می‌شود که هر تاپل شامل یک مسیر فایل خام و ID گراف مربوطه است:
```bash
zip_args = [(apps[0], graph_ids[0]), (apps[1], graph_ids[1]), ...]
```

و سپس با استفاده از `functools.partial` تابع `partial_func` ایجاد می‌شود که پارامترهای ثابت (مانند `tpl`, `hop`) را به تابع `process_apk_wrapper` ارسال می‌کند. تابع `process_apk_wrapper` یک فایل APK یا فایل خام را پردازش می‌کند. 

### بررسی تابع `process_apk_wrapper`
تابع **`process_apk_wrapper`** در کلاس `MyOwnDataset` مسئول پردازش یک فایل APK یا گراف خام است. این تابع منطق لازم برای مدیریت فایل‌های خام `.gml` را شامل می‌شود، از جمله:
1. خواندن فایل گراف خام.
2. استخراج زیرگراف‌ها با تمرکز بر گره‌های خاص (مانند APIهای حساس).
3. تبدیل زیرگراف‌ها به اشیای `Data` در PyTorch Geometric.
4. ذخیره زیرگراف‌های پردازش‌شده به صورت فایل‌های `.pt`.

```python
def process_apk_wrapper(*args, **kwargs):
    label = kwargs['label']
    tpl = kwargs['tpl']
    hop = kwargs['hop']
    base_dir = kwargs['base_dir']
    processed_dir = kwargs['processed_dir']
    app = args[0]
    graph_id = args[1]

    flag = 0
    num_subgraph = 0
    logging.info(app)

    try:
        # پردازش فایل خام `.gml` و تولید زیرگراف‌ها
        data_list = asyncio.run(gml2Data(app, label, tpl=tpl, hop=hop, base_dir=base_dir))
        dlen = len(data_list)

        # ذخیره هر زیرگراف تولیدشده به صورت فایل `.pt`
        if dlen:
            for i in range(dlen):
                data = data_list[i]
                data_path = osp.join(processed_dir, 'data_{}_{}.pt'.format(graph_id, i))
                assert not osp.exists(data_path)
                torch.save(data, data_path)
                num_subgraph += 1

            flag = 1  # پردازش موفقیت‌آمیز

        logging.info(f'[Success] {app}')

    except Exception:
        logging.exception(f'{app}')  # ثبت استثناها برای اشکال‌زدایی

    finally:
        return flag, num_subgraph  # بازگشت وضعیت موفقیت و تعداد زیرگراف‌ها
```

متغیر `flag`: مشخص می‌کند که آیا پردازش موفق بوده است (0 = شکست، 1 = موفقیت). `num_subgraph`: تعداد زیرگراف‌های تولیدشده را شمارش می‌کند.
تابع `gml2Data` فراخوانی می‌شود که:
فایل `.gml` را می‌خواند. زیرگراف‌های k-hop را حول APIهای حساس استخراج می‌کند. لیستی از اشیای `Data` را بازمی‌گرداند که هر کدام نمایانگر یک زیرگراف هستند. در انتها زیرگراف به صورت فایل `.pt` ذخیره می‌شود.

### بررسی تابع `gml2Data`
تابع **`gml2Data`** یکی از اجزای کلیدی در پردازش است که فایل خام `.gml` (نمایانگر یک گراف فراخوانی) را به زیرگراف‌هایی تبدیل می‌کند که حول گره‌های خاص (مانند APIهای حساس) متمرکز شده‌اند. این زیرگراف‌ها در قالب اشیای `Data` در PyTorch Geometric قالب‌بندی شده‌اند.

```python
async def gml2Data(gmlfile, y, base_dir, tpl=True, sub=True, hop=2, debug=False):
    fwords = ['permission', 'opcode', 'tpl'] if tpl else ['permission', 'opcode']
    single_graph, x = await prepare_file(gmlfile, base_dir, fwords)  # خواندن گراف و ویژگی‌ها
    apk_name = gmlfile.split('/decompile/')[-1][:-9]  # استخراج نام APK
    all_nodes = pd.DataFrame(single_graph.nodes, columns=['id'])  # لیست تمام گره‌های گراف

    # استخراج ویژگی‌ها
    permission, opcodes = x[:2]
    if tpl:
        opcodes = pd.merge(opcodes, x[2], how='outer', on='id', suffixes=['_', '']).drop(['type_'], axis=1)
        opcodes['type'] = opcodes['type'].fillna(2)  # نوع پیش‌فرض برای گره‌های opcode
        opcodes = opcodes.fillna(0)

    features_exist = pd.merge(permission.astype('float'), opcodes, how='outer').fillna(0).drop_duplicates('id', keep='first')
    features = pd.merge(all_nodes, features_exist, how='outer').fillna(0)
    features['type'] = features['type'].astype('int')  # اطمینان از اینکه نوع داده عدد صحیح است

    p_list = x[0].id.tolist()  # لیست گره‌های مربوط به مجوزها
    data_list = []

    # تولید زیرگراف‌ها
    if sub:
        tasks = []
        for p in p_list:
            partial_func = partial(generate_behavior_subgraph, features=features, single_graph=single_graph, hop=hop, debug=debug, gmlfile=gmlfile, apk_name=apk_name, y=y)
            tasks.append(partial_func(p))
        data_list = await asyncio.gather(*tasks)
        while None in data_list:
            data_list.remove(None)  # حذف زیرگراف‌های ناموفق
    else:
        # پردازش کل گراف در صورتی که زیرگراف نیاز نباشد
        nodes = single_graph.nodes
        edges = single_graph.edges  # [(مبدا، مقصد، کلید)، ...]
        edge_list = pd.DataFrame(edges).iloc[:, :-1].T.values.tolist()
        data = Data(
            x=torch.tensor(features.iloc[:, 1:].values.tolist(), dtype=torch.long),
            edge_index=torch.tensor(edge_list, dtype=torch.long),
            y=torch.tensor([y], dtype=torch.long),
            num_nodes=len(nodes)
        )
        data_list.append(data)

    return data_list
```
در ابتدا تابع `prepare_file` فراخوانی می‌شود و یک گراف در قالب NetworkX همراه ویژگی‌های آن در متغیر `x` خروجی داده می‌شود. مثلا:
```bash
single_graph:

MultiDiGraph with 17282 nodes and 36765 edges
```

```bash
x:

[       id  Permission:android.car.permission.CAR_CAMERA  ...  Permission:ti.permission.FMRX_ADMIN  type
0    1344                                             0  ...                                    0     1
1    1803                                             0  ...                                    0     1
2    2340                                             0  ...                                    0     1
3    2346                                             0  ...                                    0     1
4    2754                                             0  ...                                    0     1
..    ...                                           ...  ...                                  ...   ...
58  16784                                             0  ...                                    0     1
59  17050                                             0  ...                                    0     1
60  17052                                             0  ...                                    0     1
61  17054                                             0  ...                                    0     1
62  17055                                             0  ...                                    0     1

[63 rows x 270 columns],          id  nop  move  move/from16  ...  invoke-custom/range  const-method-handle  const-method-type  type
0         0    0     0            0  ...                    0                    0                  0     2
1         3    0     0            0  ...                    0                    0                  0     2
2         6    0     0            0  ...                    0                    0                  0     2
3         7    0     0            0  ...                    0                    0                  0     2
4         9    0     0            0  ...                    0                    0                  0     2
...     ...  ...   ...          ...  ...                  ...                  ...                ...   ...
12175  6339    0     0            0  ...                    0                    0                  0     2
12176  6341    0     0            0  ...                    0                    0                  0     2
12177  6343    0     0            0  ...                    0                    0                  0     2
12178  6346    0     0            0  ...                    0                    0                  0     2
12179  6349    0     0            0  ...                    0                    0                  0     2

[12180 rows x 226 columns],          id  Permission:android.car.permission.CAR_CAMERA  ...  Permission:ti.permission.FMRX_ADMIN  type
0       276                                           0.0  ...                                  0.0     3
1       277                                           0.0  ...                                  0.0     3
2       278                                           0.0  ...                                  0.0     3
3       279                                           0.0  ...                                  0.0     3
4       280                                           0.0  ...                                  0.0     3
...     ...                                           ...  ...                                  ...   ...
8910  17266                                           0.0  ...                                  0.0     3
8911  17267                                           0.0  ...                                  0.0     3
8912  17268                                           0.0  ...                                  0.0     3
8913  17274                                           0.0  ...                                  0.0     3
8914  17275                                           0.0  ...                                  0.0     3

[8915 rows x 270 columns]]
```

در ادامه تمام ویژگی‌ها با هم ادغام می‌گردند. برای مثال داریم اگر ویژگی‌های استخراج‌شده به صورت زیر باشند:
```bash
Permissions:
id   READ_SMS  ACCESS_FINE_LOCATION
0    1         0
1    0         1
2    0         0
```

```bash
Opcodes:
id   const/4  invoke-direct  return-void
0    2        1             0
1    0        2             1
2    1        0             1
```

```bash
TPL:
id   tpl_1  tpl_2
0    0      1
1    1      0
2    0      0
```

خواهیم داشت:
```bash
id   READ_SMS  ACCESS_FINE_LOCATION  const/4  invoke-direct  return-void  tpl_1  tpl_2
0    1.0       0.0                  2        1             0           0      1
1    0.0       1.0                  0        2             1           1      0
2    0.0       0.0                  1        0             1           0      0
```

در مثال خودمان داریم:
```bash
           id  Permission:android.car.permission.CAR_CAMERA  Permission:android.car.permission.CAR_HVAC  ...  invoke-custom/range  const-method-handle  const-method-type
0          0                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
1          1                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
2          2                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
3          3                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
4          4                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
...      ...                                           ...                                         ...  ...                  ...                  ...                ...
17277  17277                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
17278  17278                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
17279  17279                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
17280  17280                                           0.0                                         0.0  ...                  0.0                  0.0                0.0
17281  17281                                           0.0                                         0.0  ...                  0.0                  0.0                0.0

[17282 rows x 494 columns]
```

 در ادامه گره‌های مرتبط با APIهای حساس یا مجوزها از ویژگی‌ها استخراج می‌شوند:
 ```python
 p_list = x[0].id.tolist()
```

خروجی این قسمت از کد به صورت زیر خواهد بود:
```bash
[1344, 1803, 2340, 2346, 2754, 4268, 4269, 4278, 4280, 4281, 4282, 4284, 4285, 4287, 4288, 4289, 4641, 7723, 7725, 7733, 7734, 9384, 9534, 9811, 9812, 9813, 10029, 10378, 10890, 10897, 10898, 11280, 11282, 11285, 11412, 12344, 12345, 12349, 12364, 12365, 12367, 12369, 13511, 13514, 13516, 13542, 13546, 13568, 13598, 13680, 13877, 14105, 14663, 15075, 15086, 15100, 16780, 16783, 16784, 17050, 17052, 17054, 17055]
```

برای هر گره حساس در `p_list`، یک زیرگراف k-hop با استفاده از `generate_behavior_subgraph` استخراج می‌شود. 
```python
for p in p_list:
            partial_func = partial(generate_behavior_subgraph, features=features, single_graph=single_graph, hop=hop, debug=debug, gmlfile=gmlfile, apk_name=apk_name, y=y)
            tasks.append(partial_func(p))
        data_list = await asyncio.gather(*tasks)
```

تابع **`generate_behavior_subgraph`** زیرگراف‌های **k-hop** را از یک گراف فراخوانی (call graph) حول یک **گره مرکزی** خاص (مثلاً یک API حساس) استخراج می‌کند. این زیرگراف استخراج‌شده به یک شیء `Data` در PyTorch Geometric تبدیل می‌شود.
```python
async def generate_behavior_subgraph(p, features, single_graph, hop, debug, gmlfile, apk_name, y):
    # داده نوع گره برای فیلتر زیرگراف‌ها
    nodes_type = features[['id', 'type']]

    # استخراج گره‌ها، یال‌ها و نگاشت API زیرگراف
    subgraph_nodes, subgraph_edges, apimap = api_subgraph(
        p, single_graph, nodes_type, hop=hop, debug=debug
    )

    # مدیریت گره‌های جدا شده
    if len(subgraph_nodes) <= 1:
        logging.warning(f'[IsolateNode] {gmlfile}: isolated node@{p}')
        return None

    # فیلتر ویژگی‌های گره زیرگراف
    subtypes = nodes_type[nodes_type['id'].isin(subgraph_nodes)]
    subgraph_features = features[features.id.isin(subgraph_nodes)].reset_index(drop=True)

    # اطمینان از همخوانی تعداد گره‌ها
    assert subgraph_features.shape[0] == len(subgraph_nodes)

    # تبدیل یال‌ها به DataFrame
    edges_df = pd.DataFrame(subgraph_edges).iloc[:, :-1].T

    # تبدیل یال‌ها به فرمت PyTorch Geometric
    edge_list, center, mapping = convert_subgraph_edge(edges_df.values.tolist(), subgraph_features, p)

    # اطمینان از همخوانی نگاشت و لیست API
    assert len(apimap) == len(mapping)
    mapping = [apimap[i] for i in mapping]
    labels = [subtypes[subtypes.id == i].type.tolist()[0] for i in mapping]

    # ایجاد شیء `Data` در PyTorch Geometric
    data = Data(
        x=torch.tensor(subgraph_features.iloc[:, 1:-1].values.tolist(), dtype=torch.float),
        edge_index=torch.tensor(edge_list, dtype=torch.long),
        y=torch.tensor([y], dtype=torch.long),
        num_nodes=len(subgraph_nodes),
        labels=labels,
        center=center,
        mapping=mapping,
        app=apk_name
    )
    return data
```

در ابتدا با استفاده از تابع `api_subgraph` یک زیرگراف به عمق `hop` حول گره حساس `p` استخراج می‌گردد. خروجی این قسمت برابر است با:
```bash
subgraph nodes:  [2050, 11269, 11399, 11274, 2059, 11276, 11277, 11278, 11283, 11412, 4760, 4761, 28, 30, 4511, 32, 9767, 9926, 2147, 617, 1651]
```

```bash
subgraph edges:  [(11269, 11399, 108), (11399, 11283, 58), (11399, 11283, 380), (11399, 11283, 296), (11399, 4761, 328), (11399, 4761, 456), (11399, 4761, 412), (11399, 4761, 100), (11399, 11412, 108), (11399, 2050, 70), (11399, 2050, 304), (11399, 2050, 388), (11399, 32, 12), (11399, 32, 262), (11399, 32, 346), (11399, 32, 138), (11399, 32, 214), (11399, 11277, 182), (11399, 4511, 88), (11399, 4511, 316), (11399, 4511, 444), (11399, 4511, 400), (11399, 28, 372), (11399, 28, 164), (11399, 28, 240), (11399, 28, 38), (11399, 28, 288), (11399, 9767, 120), (11399, 11278, 174), (11399, 2059, 76), (11399, 11276, 196), (11399, 11274, 406), (11399, 11274, 94), (11399, 11274, 322), (11399, 11274, 450), (11399, 30, 224), (11399, 30, 352), (11399, 30, 364), (11399, 30, 144), (11399, 30, 22), (11399, 30, 156), (11399, 30, 268), (11399, 30, 280), (11399, 2147, 310), (11399, 617, 232), (11399, 617, 30), (11399, 4760, 394), (11399, 9926, 46), (11399, 1651, 248), (11283, 32, 4), (11283, 28, 46), (11283, 30, 38), (11283, 30, 30), (11283, 617, 18)]
```

```bash
api map:  {2050: 'Ljava/io/File;-><init>(Ljava/lang/String;)V', 11269: 'Lcom/startapp/android/publish/ads/video/g;->a()V', 11399: 'Lcom/startapp/android/publish/ads/video/h;->a(Landroid/content/Context; Ljava/net/URL; Ljava/lang/String;)Ljava/lang/String;', 11274: 'Ljava/io/DataInputStream;->close()V', 2059: 'Ljava/io/File;->exists()Z', 11276: 'Ljava/io/FileOutputStream;->write([B I I)V', 11277: 'Ljava/io/DataInputStream;->read([B)I', 11278: 'Landroid/content/Context;->openFileOutput(Ljava/lang/String; I)Ljava/io/FileOutputStream;', 11283: 'Lcom/startapp/android/publish/ads/video/h;->a(Landroid/content/Context; Ljava/lang/String;)Ljava/lang/String;', 11412: 'Ljava/net/URL;->openStream()Ljava/io/InputStream;', 4760: 'Ljava/io/File;->renameTo(Ljava/io/File;)Z', 4761: 'Ljava/io/FileOutputStream;->close()V', 28: 'Ljava/lang/StringBuilder;->toString()Ljava/lang/String;', 30: 'Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;', 4511: 'Ljava/io/InputStream;->close()V', 32: 'Ljava/lang/StringBuilder;-><init>()V', 9767: 'Ljava/io/DataInputStream;-><init>(Ljava/io/InputStream;)V', 9926: 'Lcom/startapp/common/a/g;->a(Ljava/lang/String; I Ljava/lang/String;)V', 2147: 'Ljava/io/File;->delete()Z', 617: 'Ljava/lang/StringBuilder;->append(Ljava/lang/Object;)Ljava/lang/StringBuilder;', 1651: 'Landroid/util/Log;->e(Ljava/lang/String; Ljava/lang/String; Ljava/lang/Throwable;)I'}
```

برمی‌گردیم به تابع `gml2Data` و داریم که `data_list` برابر مقدار زیر است:
```bash
[Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=1, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=1, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[10, 492], edge_index=[2, 9], y=[1], num_nodes=10, labels=[10], center=9, mapping=[10], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[12, 492], edge_index=[2, 12], y=[1], num_nodes=12, labels=[12], center=10, mapping=[12], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[6, 492], edge_index=[2, 5], y=[1], num_nodes=6, labels=[6], center=4, mapping=[6], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[2, 492], edge_index=[2, 1], y=[1], num_nodes=2, labels=[2], center=0, mapping=[2], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[2, 492], edge_index=[2, 1], y=[1], num_nodes=2, labels=[2], center=1, mapping=[2], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[4, 492], edge_index=[2, 3], y=[1], num_nodes=4, labels=[4], center=3, mapping=[4], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[12, 492], edge_index=[2, 15], y=[1], num_nodes=12, labels=[12], center=3, mapping=[12], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[45, 492], edge_index=[2, 80], y=[1], num_nodes=45, labels=[45], center=12, mapping=[45], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[4, 492], edge_index=[2, 3], y=[1], num_nodes=4, labels=[4], center=3, mapping=[4], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[12, 492], edge_index=[2, 15], y=[1], num_nodes=12, labels=[12], center=5, mapping=[12], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[4, 492], edge_index=[2, 3], y=[1], num_nodes=4, labels=[4], center=3, mapping=[4], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=0, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=1, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=2, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[2, 492], edge_index=[2, 1], y=[1], num_nodes=2, labels=[2], center=1, mapping=[2], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=0, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=2, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[2, 492], edge_index=[2, 1], y=[1], num_nodes=2, labels=[2], center=0, mapping=[2], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[2, 492], edge_index=[2, 1], y=[1], num_nodes=2, labels=[2], center=1, mapping=[2], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[109, 492], edge_index=[2, 267], y=[1], num_nodes=109, labels=[109], center=19, mapping=[109], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[18, 492], edge_index=[2, 21], y=[1], num_nodes=18, labels=[18], center=4, mapping=[18], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[5, 492], edge_index=[2, 4], y=[1], num_nodes=5, labels=[5], center=4, mapping=[5], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[5, 492], edge_index=[2, 4], y=[1], num_nodes=5, labels=[5], center=4, mapping=[5], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[5, 492], edge_index=[2, 4], y=[1], num_nodes=5, labels=[5], center=4, mapping=[5], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[110, 492], edge_index=[2, 231], y=[1], num_nodes=110, labels=[110], center=23, mapping=[110], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[50, 492], edge_index=[2, 116], y=[1], num_nodes=50, labels=[50], center=31, mapping=[50], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[199, 492], edge_index=[2, 514], y=[1], num_nodes=199, labels=[199], center=84, mapping=[199], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[97, 492], edge_index=[2, 171], y=[1], num_nodes=97, labels=[97], center=47, mapping=[97], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[54, 492], edge_index=[2, 129], y=[1], num_nodes=54, labels=[54], center=32, mapping=[54], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[119, 492], edge_index=[2, 420], y=[1], num_nodes=119, labels=[119], center=64, mapping=[119], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[30, 492], edge_index=[2, 105], y=[1], num_nodes=30, labels=[30], center=26, mapping=[30], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[30, 492], edge_index=[2, 105], y=[1], num_nodes=30, labels=[30], center=29, mapping=[30], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[21, 492], edge_index=[2, 54], y=[1], num_nodes=21, labels=[21], center=20, mapping=[21], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[15, 492], edge_index=[2, 20], y=[1], num_nodes=15, labels=[15], center=8, mapping=[15], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[15, 492], edge_index=[2, 20], y=[1], num_nodes=15, labels=[15], center=9, mapping=[15], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[15, 492], edge_index=[2, 20], y=[1], num_nodes=15, labels=[15], center=13, mapping=[15], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[11, 492], edge_index=[2, 23], y=[1], num_nodes=11, labels=[11], center=9, mapping=[11], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[20, 492], edge_index=[2, 35], y=[1], num_nodes=20, labels=[20], center=17, mapping=[20], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[22, 492], edge_index=[2, 47], y=[1], num_nodes=22, labels=[22], center=19, mapping=[22], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[18, 492], edge_index=[2, 33], y=[1], num_nodes=18, labels=[18], center=17, mapping=[18], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[9, 492], edge_index=[2, 15], y=[1], num_nodes=9, labels=[9], center=7, mapping=[9], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[32, 492], edge_index=[2, 63], y=[1], num_nodes=32, labels=[32], center=20, mapping=[32], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[9, 492], edge_index=[2, 13], y=[1], num_nodes=9, labels=[9], center=8, mapping=[9], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[10, 492], edge_index=[2, 9], y=[1], num_nodes=10, labels=[10], center=7, mapping=[10], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[8, 492], edge_index=[2, 7], y=[1], num_nodes=8, labels=[8], center=6, mapping=[8], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[57, 492], edge_index=[2, 101], y=[1], num_nodes=57, labels=[57], center=33, mapping=[57], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[11, 492], edge_index=[2, 12], y=[1], num_nodes=11, labels=[11], center=10, mapping=[11], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[39, 492], edge_index=[2, 50], y=[1], num_nodes=39, labels=[39], center=30, mapping=[39], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[5, 492], edge_index=[2, 4], y=[1], num_nodes=5, labels=[5], center=4, mapping=[5], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[19, 492], edge_index=[2, 29], y=[1], num_nodes=19, labels=[19], center=10, mapping=[19], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[2, 492], edge_index=[2, 1], y=[1], num_nodes=2, labels=[2], center=1, mapping=[2], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[4, 492], edge_index=[2, 3], y=[1], num_nodes=4, labels=[4], center=3, mapping=[4], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[4, 492], edge_index=[2, 3], y=[1], num_nodes=4, labels=[4], center=3, mapping=[4], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[11, 492], edge_index=[2, 14], y=[1], num_nodes=11, labels=[11], center=10, mapping=[11], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[15, 492], edge_index=[2, 21], y=[1], num_nodes=15, labels=[15], center=13, mapping=[15], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=2, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[11, 492], edge_index=[2, 13], y=[1], num_nodes=11, labels=[11], center=10, mapping=[11], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=2, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[7, 492], edge_index=[2, 8], y=[1], num_nodes=7, labels=[7], center=6, mapping=[7], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=2, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3'), Data(x=[3, 492], edge_index=[2, 2], y=[1], num_nodes=3, labels=[3], center=2, mapping=[3], app='1EC3CAC448F523E6286176C6CF12BC1BD3EE485445B4A428FC2655DDBFE505F3')]
```

در تابع `process_apk_wrapper` هر کدام از این ساختارهای `Data` به صورت یک فایل `.pt` ذخیره می‌گردد.

----
## انطباق تولید زیرگراف با مقالۀ MsDroid
**عملکرد تولید زیرگراف** که در مقاله MsDroid توضیح داده شده است، با بخش‌های خاصی از کد که مسئول **مقداردهی اولیه**، **تقسیم‌بندی** و **کاهش** هستند، مطابقت دارد. در ادامه، مراحل نظری موجود در مقاله را با پیاده‌سازی آن‌ها در کد ارائه‌شده تطبیق می‌دهم.
### 1. مقداردهی اولیه (Initialization)
این مرحله مربوط به:
- شناسایی **APIهای حساس** به عنوان **گره‌های مرکزی**.
- استخراج **همسایگی k-hop** برای هر گره مرکزی است.
#### ارجاع به کد
این مرحله در تابع `api_subgraph` مدیریت می‌شود:
```python
def api_subgraph(node, graph, nodes_type, hop=2, debug=False, apimap=True):
    # استخراج همسایگی k-hop گره مرکزی
    ego_graph = nx.ego_graph(graph, node, radius=hop, undirected=True)

    if debug:
        print(f'{hop} neighborhood of {node}:')
        color = ['r' if n == node else 'b' for n in ego_graph.nodes]
        import matplotlib.pyplot as plt
        nx.draw(ego_graph, node_color=color, with_labels=True)
        plt.show()
```

### 2. تقسیم‌بندی (Partitioning)
تقسیم‌بندی در توابع **`generate_behavior_subgraph`** و **`api_subgraph`** انجام می‌شود:
```python
# تولید زیرگراف حول گره مرکزی
subgraph_nodes, subgraph_edges, apimap = api_subgraph(
    p, single_graph, nodes_type, hop=hop, debug=debug
)

# جداسازی نوع گره‌ها با استفاده از `nodes_type`
subtypes = nodes_type[nodes_type['id'].isin(subgraph_nodes)]
```

### 3. کاهش (Reduction)
کاهش در توابع **`api_subgraph`** و **`prune`** انجام می‌شود:
```python
if hop > 2:  # هرس برای همسایگی‌های بزرگ‌تر
    ori_num = len(ego_graph.nodes)
    ego_graph = prune(ego_graph, nodes_type, node, debug=debug)
    logging.info(f'[Prune] node {node}: {len(ego_graph.nodes)} / {ori_num}')
```

