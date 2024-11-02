## مروری بر مراحل پیاده‌سازی

پیاده‌سازی MsDroid شامل مراحل زیر است:
1. **پیش‌پردازش**: در این مرحله تمام اطلاعات مورد نیاز باید از دیتاست اپلیکیشن اندرویدی استخراج گردد.
2. **تولید زیرگراف‌ها**: در این مرحله زیرگراف‌هایی باید تولید گردند که هرکدام حول یک api حساس تشکیل شده باشند.
3. **شبکه عصبی گرافی**: در این مرحله با استفاده از GNN مدنظر مقاله، زیرگراف‌ها encode می‌گردند.
4. **طبقه‌بندی**: با استفاده از classifier مطرح‌شده در مقاله، طبقه‌بند با هدف binary classification کار خود را انجام می‌دهد.

## پیش‌پردازش

![preprocess](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/03.png)

این مرحله تمامی feature هایی که باید از فایل‌های APK استخراج گردد را خارج می‌کند. این featureها عبارتند از:
### 1. گراف فراخوانی هر فایل APK
در این گراف، هر گره یک method است و هرگاه فراخوانی بین method ها رخ دهد، یک یال بین این گره‌ها ایجاد می‌گردد.

![callgraph1](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/01.png)

![cg2](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/02.png)

![cg3](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/07.png)

### 2. آپکودها
در ابتدا یک تناسب بین هر گره با یک id یکتا ایجاد می‌شود. سپس بر اساس این ارتباط، برای method مرتبط با هر گره، opcode هایی که در آن method وجود دارند، علامت می‌خورند.

![oc1](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/04.png)

![oc2](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/08.png)

### 3. مجوزها
از قبل permission های حساس در نظر گرفته شده‌اند. بر اساس این permission ها، هر کدام از گره‌ها که شامل این permission ها باشد مشخص می‌گردد.
![per1](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/05.png)

![per2](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/09.png)

### 4. وجود و مجوزهای Third-Party Libraries
در این قسمت وجود third-party libraries در هر گره و این که شامل چه مجوزهایی می‌شوند مشخص می‌گردد. این کار برای بررسی گره‌هایی است به یک TPL اشاره دارند.

![tpl1](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/06.png)

![tpl2](https://github.com/aboyou/DroidMDO/blob/main/Reports/02-11-2024/images/10.png)


## تولید زیرگراف
کد این قسمت نوشته شده است اما دارای کمی باگ است که به محض رفع باگ آن هم گذاشته می‌شود.

---
## زمان!

یکی از مسائل جدی در این پیاده‌سازی مرحلۀ پیش‌پردازش است. برای بررسی فایل‌های APK از کتابخانه پایتونی Androguard استفاده می‌شود. در ابتدا باید هر APK با ساختار یک session باز شود و عملیات‌های دیگر بر روی آن انجام شود.

زمان محاسبه‌شده برای طی مرحلۀ پیش‌پردازش به ازای هر فایل APK برابر 102 ثانیه است که حدودا برابر 1.7 دقیقه می‌شود. با توجه به مجموعه داده 10000 عضوی، مقدار زمان مورد نیاز برای مرحلۀ پیش‌پردازش برابر 12 روز است. این در صورتی است که سیستم به صورت مداوم کار کند!

> 12 روز برای محاسبۀ preprocess!

