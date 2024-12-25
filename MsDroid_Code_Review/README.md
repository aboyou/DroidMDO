
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
هدف آن است که بفهمیم چگونه این ویژگی‌ها (features) تولید می‌شوند. حال یا درون کلاس `AndroGen` تولید خواهند شد یا این که از طریق تابع `generate_hgraph()` بدست خواهند آمد.

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

	 دایرکتوری‌هایی را برای ذخیره گراف‌های تماس، کدهای عملیاتی، مجوزها و الگوها ایجاد می‌کند.
	 گراف تماس را با استفاده از شی تجزیه و تحلیل استخراج می‌کند و آن را با فرمت gml. با استفاده از networkx ذخیره می‌کند.
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

در واقع این نگاشت جایگزینی (replacement) برای هندل کردن کامپوننت‌های اندرویدی استفاده می‌شود که از الگوی خاصی در طراحی پیروی می‌کنند. در واقع کامپوننت‌های `AsyncTask`، `Handler` و `Runnable` متدهایی دارند که قابلیت اجرای asynchronously را دارا هستند و یا این که override می‌شوند و تسک خاصی را انجام می‌دهند. این متدها برای فهم عملکرد اپلیکیشن‌های اندرویدی حیاتی هستند، به خصوص در آنالیز عملیات‌ههای حیاتی API callها! 
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
این opcode ها تنها بخشی از opcode های Android Dalvik نیستند بلکه تمام opcode ها هستند. پس فقط بخشی از opcode ها به عنوان حساس در نظر گرفته نشده‌اند و استخراجج اطلاعات مربوط به تمامی opcode ها مدنظر است! 
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
opcodePAth:  /home/user/MsDroid2/MsDroid-main/src/Output/Test_DB/result/opcode
```
