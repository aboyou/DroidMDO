# **MsDroid Code Review - Training**
در یادداشت قبلی فرایند تولید زیرگراف‌ها توضیح داده شد ([لینک](https://github.com/aboyou/DroidMDO/tree/main/MsDroid_Code_Review))، در این یادداشت مدل یادگیری و فرایند آموزش آن توضیح داده خواهد شد.
```python
#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@File    :   train.py
@Time    :   2020/12/26 18:12:18
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''
# here put the import lib
import sys
import os
import logging
from training import set_train_config, GraphDroid
from graph import generate_graph
from utils import makedirs, set_logger
from main import generate_behavior_subgraph

exp_base = './training/Experiments'
graph_base = f'./training/Graphs'
logger = set_logger(logging.getLogger())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MsDroid Trainer.')
    # Generate behavior subgraphs
    parser.add_argument('--input', '-i', help='APK directory')
    parser.add_argument('--output', '-o', help='output directory', default=f'{sys.path[0]}/Output')
    parser.add_argument('--device', '-d', help='device for model test', default='cpu')
    parser.add_argument('--batch', '-b', help='batch size for model test', default=16)
    parser.add_argument('--label', '-l', help='dataset label: malware(1) / benign(0), unnecessary if only prediction needed.', default=1)
    parser.add_argument('--deepth', '-dp', help='deepth of tpl searching', default=3)
    # Training
    parser.add_argument(
    '--dbs',                   # Argument name
    nargs='+',                 # Accept one or more values as a list
    default=['TestAPK'],       # Default value when no argument is provided
    help='Datasets to train (space-separated).'  # Description of the argument
)
    #parser.add_argument('--dbs', type=list, default=['TestAPK'], help='Datasets to train.')
    parser.add_argument('--tpl', type=bool, default=True, help='TPL simplified subgraphs.')
    parser.add_argument('--hop', type=int, default=2, help='K-hop based subgraphs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for Dataloader.')
    parser.add_argument('--train_rate', type=float, default=0.8, help='Training rate.')
    parser.add_argument('--norm_op', type=bool, default=False, help='Normalize opcodes feature.')
    parser.add_argument('--mask', type=int, default=-1, help='Mask node features. 0: disable opcodes, 1: disable permission, 2: disable both')
    parser.add_argument('--global_pool', type=str, default='mix', help='Global pooling method for graph classification.')
    parser.add_argument('--lossfunc', type=int, default=0, help='Index of loss function.')
    parser.add_argument('--dimension', type=int, default=128, help='Hidden layer graph embedding dimension.')
    parser.add_argument('--dev', type=int, default=0, help='GPU device id.')
    parser.add_argument('--exp_base', type=str, default=exp_base, help='Dir to put exp results.')
    parser.add_argument('--graph_base', type=str, default=graph_base, help='Dir for graphs.')
    # For Train (`train_and_test`)
    parser.add_argument('--epoch', type=int, default=1, help='Training epoches.')
    parser.add_argument('--force', type=bool, default=False, help='Force new train in exp_base with same config.')
    parser.add_argument('--continue_train', type=bool, default=False, help='Continue to train from last checkpoint.')
    args = parser.parse_args()
    input_dir = args.input
    apk_base = os.path.abspath(os.path.join(input_dir,'../'))
    db_name = input_dir.split(apk_base)[-1].strip('/')
    output_dir = args.output
    makedirs(output_dir)
    label = args.label
    dbs = args.dbs
    tpl = args.tpl
    hop = args.hop
    batch_size = args.batch_size
    train_rate = args.train_rate
    norm_opcode = args.norm_op
    mask = args.mask
    global_pool = args.global_pool
    dimension = args.dimension
    lossfunc = args.lossfunc
    dev = args.dev
    epoch = args.epoch
    force = args.force
    continue_train = args.continue_train
    exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}'
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

    model_config = set_train_config(batch_size=batch_size, train_rate=train_rate, global_pool=global_pool, lossfunc=lossfunc, dimension=dimension)
    graph_droid = GraphDroid(hop, tpl, dbs, norm_opcode=norm_opcode, mask=mask, model_config=model_config, exp_base=args.exp_base, graph_base=args.graph_base, logger=logger)
    graph_droid.train_and_test(epoch, force=force, continue_train=continue_train, dev=dev, testonly=testonly)
```

کد بالا فایل `train.py` است که مسئولیت اجرای آموزش مدل را دارد. در کد بالا در ابتدا شاهد آن هستیم که زیرگراف‌های رفتاری تولید می‌گردند.
```python
num_apk = generate_behavior_subgraph(apk_base, db_name, output_dir, args.deepth, label, hop=hop, tpl=tpl, training=True, api_map=True)
```

در ادامه خط زیر اجرا می‌گردد:
```python
odel_config = set_train_config(batch_size=batch_size, train_rate=train_rate, global_pool=global_pool, lossfunc=lossfunc, dimension=dimension)
```

انتظار می‌رود این تابع پیکربندی تنظیمات آموزش مدل را انجام دهد. در پارامترها، پارامتری به نام `training_rate` وجود دارد. که این پارامتر نسبت داده‌های استفاده شده برای آموزش به اعتبارسنجی/تست را مشخص می‌کند. مثلا `train_rate=0.8` به معنای استفاده از ۸۰٪ داده‌ها برای آموزش و ۲۰٪ باقی‌مانده برای اعتبارسنجی/تست است.
همچنین پارامتر دیگر `global_pool` است. این پارامتر نوع عملیات pooling روی گراف را مشخص می‌کند. همچنین پارامتر `dimension`، اندازه لایه‌های پنهان در مدل گراف امبدینگ را تنظیم می‌کند.

# جریان اجرای فرایند آموزش

```
                 Start
                    │
      ┌─────────────▼──────────────┐
      │ Initialize GraphDroid Class │
      │ (Loads Configuration)       │
      └─────────────┬──────────────┘
                    │
          Calls `train_and_test()`
                    │
                    ▼
  ┌───────────────────────────────────┐
  │   Initialize Experiment Setup      │
  │   (Check existing training data)   │
  └───────────────────────────────────┘
                    │
      ┌────────────▼──────────────┐
      │ Train-Test Split           │
      │ (Separate Training & Test) │
      └────────────┬──────────────┘
                    │
        Calls `my_train()`
                    │
                    ▼
   ┌───────────────────────────────────┐
   │    Train Model using PyTorch       │
   │    (Run for `num_epoch` epochs)    │
   └───────────────────────────────────┘
                    │
      ┌────────────▼──────────────┐
      │ Save Best Model Weights    │
      └────────────┬──────────────┘
                    │
        Calls `__get_scores()`
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   Evaluate Model Performance       │
    │   (Compute Precision, Recall, F1)  │
    └───────────────────────────────────┘
                    │
      ┌────────────▼──────────────┐
      │  Generate Performance      │
      │  Reports & Save Results    │
      └────────────┬──────────────┘
                    │
       Calls `portability_test()`
                    │
                    ▼
     ┌───────────────────────────────┐
     │  Test on Different Datasets    │
     │  (Generalization Check)        │
     └───────────────────────────────┘
                    │
        Calls `explain_subgraphs()`
                    │
                    ▼
      ┌───────────────────────────────┐
      │ Explain Subgraph Predictions  │
      │ (Find Important APIs & Edges) │
      └───────────────────────────────┘
                    │
      ┌────────────▼──────────────┐
      │  Generate Explanation      │
      │  Reports & Visualizations  │
      └────────────┬──────────────┘
                    │
                Finished 🎉

```

# اولیه‌سازی آموزش (Initialization)
```
                      ┌──────────────────────────────┐
                      │  Initialize GraphDroid Class │
                      └──────────────┬──────────────┘
                                     │
      ┌──────────────────────────────▼─────────────────────────────┐
      │ Load Graph Configuration Parameters                        │
      │ - hop, tpl, train_dbs, norm_opcode, mask                   │
      └──────────────────────────────┬─────────────────────────────┘
                                     │
      ┌──────────────────────────────▼─────────────────────────────┐
      │ Set Up Model Training Configurations                        │
      │ - Load model_config or set default via __get_basic_train_config() │
      └──────────────────────────────┬─────────────────────────────┘
                                     │
      ┌──────────────────────────────▼─────────────────────────────┐
      │ Define Experiment & Graph Paths                            │
      │ - exp_base, graph_base                                     │
      └──────────────────────────────┬─────────────────────────────┘
                                     │
      ┌──────────────────────────────▼─────────────────────────────┐
      │ Set Up Logger                                                │
      └─────────────────────────────────────────────────────────────┘

```

