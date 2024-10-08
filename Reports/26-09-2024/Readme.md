# گزارش 09/26/2024
در این گزارش در مورد طرح توضیحاتی داده می‌شود. سپس قسمت‌های مختلفی که تا کنون پیاده‌سازی شده‌اند نیز شرح داده می‌شوند.

## استخراج گراف فراخوانی برنامه اندرویدی
برای پردازش اطلاعات مربوط به یک برنامه اندرویدی در این طرح لازم است که در ابتدا گراف فراخوانی آن را داشته باشیم. به همین دلیل در این طرح از تابع `generate_behavior_graph()` برای تولید گراف‌ فراخوانی و زیرگراف‌های رفتاری طبق طرح استفاده می‌شود.
در طی فراخوانی این تابع و زیرتابع‌های آن، دایرکتوری‌هایی تولید می‌شود که بنابر نیاز ذخیره‌سازی ایجاد می‌گردند.

## آنالیز برنامه اندرویدی
برای آنالیز برنامه اندرویدی از کلاس `auto` از `androguard.core.analysis` استفاده می‌گردد. برای استفاده از این کلاس باید تنظمیاتی را انجام داد که در این تنظیمات، مهمترین چیز معرفی کلاس آنالیز است. چیزی که در این طرح به نام `AndroGen` معرفی شده است.
```Python
settings = {
        # The directory `some/directory` should contain some APK files
        "my": AndroGen(APKpath=db_path, CGPath=cg_path, FeaturePath=feature_path, deepth=deepth),  # apkfile
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
```

### کلاس `()AndroGen`
این کلاس دارای توابع مختلفی است که هسته اصلی آن، تابع `analysis_app` قرار دارد.در پیاده‌سازی این تابع از توابع و کلاس‌های زیادی استفاده شده است که همگی آن‌ها در حال پیاده‌سازی است. هم‌اکنون در مرحله حال پیاده‌سازی کلاس `Permission` قرار داریم.
