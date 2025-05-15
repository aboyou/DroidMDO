برای استفاده از Obfuscapk از دستور زیر استفاده می‌کنیم:
```bash
python3 -m obfuscapk.cli  --input  path/to/original.apk --output path/to/reflected.apk --obfuscators AdvancedReflection
```

```bash
python3 -m obfuscapk.cli -p -d /home/yousefi/apk.apk -o AdvancedReflection /home/yousefi/test_apks/dataset1/0003667d25c73cc8942df67ccfff075b0c4bc9a146ef1a7afd8cec5a56c73089.apk
```


```bash
python3 -m obfuscapk.cli -p -d /home/user/apk.apk -o AdvancedReflection /home/user/test_apks/dataset1/0003667d25c73cc8942df67ccfff075b0c4bc9a146ef1a7afd8cec5a56c73089.apk
```


```python
import os
import subprocess
import hashlib
import shutil
import argparse

APKTOOL = "apktool"
ZIPALIGN = "zipalign"
APKSIGNER = "apksigner"

KEYSTORE = "my-release-key.jks"
ALIAS = "mykey"
STOREPASS = "password"
KEYPASS = "password"

def run(cmd):
    print(f"🛠️ Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError("❌ Command failed")

def get_apk_hash(apk_path):
    with open(apk_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def ensure_keystore():
    if not os.path.exists(KEYSTORE):
        run(
            f"keytool -genkey -v -keystore {KEYSTORE} -alias {ALIAS} "
            f"-keyalg RSA -keysize 2048 -validity 10000 "
            f"-storepass {STOREPASS} -keypass {KEYPASS} "
            f'-dname "CN=Obfuscator, OU=Dev, O=Example, L=Tehran, S=Tehran, C=IR"'
        )

def obfuscate_apk(input_apk, output_apk, work_dir, obfuscators):
    apk_hash = get_apk_hash(input_apk)
    apk_work_dir = os.path.join(work_dir, apk_hash)
    unsigned_apk = f"{apk_hash}_unsigned.apk"
    signed_apk = f"{apk_hash}_signed.apk"

    # Run obfuscapk
    obfuscator_str = ' -o '.join(obfuscators)
    run(f"python3 -m obfuscapk.cli -p -d dummy.apk -o {obfuscator_str} {input_apk}")

    # Build APK
    run(f"{APKTOOL} b {apk_work_dir} -o {unsigned_apk}")

    # Sign APK
    run(
        f"{APKSIGNER} sign --ks {KEYSTORE} --ks-key-alias {ALIAS} "
        f"--ks-pass pass:{STOREPASS} --key-pass pass:{KEYPASS} "
        f"--out {signed_apk} {unsigned_apk}"
    )

    # Zipalign
    run(f"{ZIPALIGN} -v 4 {signed_apk} {output_apk}")

    # Clean temp files
    for f in [unsigned_apk, signed_apk]:
        if os.path.exists(f):
            os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="Obfuscate and sign APKs in batch")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory with APKs")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for final APKs")
    parser.add_argument("-w", "--work_dir", default="obfuscation_working_dir", help="Working directory for obfuscapk")
    parser.add_argument("-obf", "--obfuscators", nargs="+", required=True, help="List of obfuscators to apply")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    ensure_keystore()

    apks = [f for f in os.listdir(args.input_dir) if f.endswith(".apk")]
    print(f"📦 Found {len(apks)} APKs")

    for apk in apks:
        input_apk = os.path.join(args.input_dir, apk)
        output_apk = os.path.join(args.output_dir, apk)

        print(f"\n🚀 Processing: {apk}")
        try:
            obfuscate_apk(input_apk, output_apk, args.work_dir, args.obfuscators)
            print(f"✅ Done: {output_apk}")
        except Exception as e:
            print(f"❌ Failed on {apk}: {e}")

if __name__ == "__main__":
    main()
```

```bash
python3 batch_obfuscate_sign.py -i dataset1 -o obf_output -w obfuscation_working_dir -obf AdvancedReflection
```

```bash
-obf Rebuild ApktoolMove AdvancedReflection
```

```python
import os
import random
import subprocess
from pathlib import Path

from argparse import ArgumentParser

APKTOOL = "apktool"
ZIPALIGN = "zipalign"
APKSIGNER = "apksigner"
KEYSTORE = "my-release-key.jks"
ALIAS = "mykey"
STOREPASS = "password"
KEYPASS = "password"


def run(cmd):
    print(f"⚙️ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def ensure_keystore():
    if not os.path.exists(KEYSTORE):
        run(
            f"keytool -genkey -v -keystore {KEYSTORE} -alias {ALIAS} "
            f"-keyalg RSA -keysize 2048 -validity 10000 "
            f"-storepass {STOREPASS} -keypass {KEYPASS} "
            f'-dname "CN=Test, OU=Dev, O=Research, L=Tehran, S=Tehran, C=IR"'
        )


def filter_apks_by_size(apk_dir, min_size_kb=2048, max_size_kb=10240):
    apk_files = list(Path(apk_dir).glob("*.apk"))
    filtered = [apk for apk in apk_files if min_size_kb * 1024 <= apk.stat().st_size <= max_size_kb * 1024]
    return filtered


def obfuscate_apk(apk_path, output_path, obfuscators, work_dir):
    apk_hash = apk_path.stem
    work_apk_dir = Path(work_dir) / apk_hash
    unsigned_apk = work_apk_dir / f"{apk_hash}_unsigned.apk"
    signed_apk = work_apk_dir / f"{apk_hash}_signed.apk"

    # اجرای obfuscapk
    obf_flags = " -o ".join(obfuscators)
    run(f"python3 -m obfuscapk.cli -p -d dummy.apk -o {obf_flags} {apk_path}")

    # ساخت مجدد APK
    run(f"{APKTOOL} b {work_apk_dir} -o {unsigned_apk}")

    # امضا
    run(
        f"{APKSIGNER} sign --ks {KEYSTORE} --ks-key-alias {ALIAS} "
        f"--ks-pass pass:{STOREPASS} --key-pass pass:{KEYPASS} "
        f"--out {signed_apk} {unsigned_apk}"
    )

    # zipalign
    final_output = Path(output_path) / apk_path.name
    run(f"{ZIPALIGN} -v 4 {signed_apk} {final_output}")

    print(f"✅ Done: {final_output}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", default="obfuscation_working_dir")
    parser.add_argument("--obfuscators", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--min_kb", type=int, default=2048)
    parser.add_argument("--max_kb", type=int, default=10240)
    parser.add_argument("--output_list", default="selected_apks.txt")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    ensure_keystore()

    # گام ۱ و ۲: فیلتر بر اساس سایز
    candidates = filter_apks_by_size(args.input_dir, args.min_kb, args.max_kb)
    print(f"📦 Found {len(candidates)} APKs in size range")

    # گام ۳: انتخاب رندوم ۵۰۰۰ تای اول
    selected = random.sample(candidates, min(args.count, len(candidates)))

    # گام ۴: ذخیره لیست انتخاب شده
    with open(args.output_list, "w") as f:
        for apk in selected:
            f.write(str(apk) + "\n")
    print(f"📄 Saved selected APKs to {args.output_list}")

    # گام ۵: انجام obfuscation
    for apk in selected:
        try:
            obfuscate_apk(apk, args.output_dir, args.obfuscators, args.work_dir)
        except Exception as e:
            print(f"❌ Failed: {apk.name}, Error: {e}")


if __name__ == "__main__":
    main()
```

```bash
python3 obfuscator_random.py --input_dir ~/test_apks/dataset1 --output_dir ~/test_apks/obf_output2 --work_dir ~/test_apks/dataset1/obfuscation_working_dir2 --obfuscators AdvancedReflection --count 1 --min_kb 1 --max_kb 10240
```

```bash
 python3 obfuscator_modified.py -i ~/test_apks/dataset1 -o ~/test_apks/obf_output -w ~/test_apks/dataset1/obfuscation_working_dir -obf AdvancedReflection
```

```bash
python3 obf_updated.py -i /home/yousefi/Androzoo_Dataset/Downloaded_Malwares_50000 -o /home/yousefi/Androzoo_Dataset/Downloaded_Malwares_50000_reflectd -w /home/yousefi/Androzoo_Dataset/Downloaded_Malwares_50000_obfuscation_working_dir -obf AdvancedReflection
```
