#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input-apk-directory> <output-apk-directory>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

for APK_FILE in "$INPUT_DIR"/*.apk; do
    if [ ! -f "$APK_FILE" ]; then
        echo "[!] No APK files found in $INPUT_DIR"
        exit 1
    fi

    BASENAME=$(basename "$APK_FILE" .apk)
    DECOMPILED_DIR="decompiled_$BASENAME"
    OUTPUT_APK="$OUTPUT_DIR/${BASENAME}_obfuscated.apk"

    echo "[*] Processing $APK_FILE ..."

    echo "    [*] Decompiling APK..."
    apktool d "$APK_FILE" -o "$DECOMPILED_DIR" --force > /dev/null 2>&1

    echo "    [*] Obfuscating Smali class names, method names, and field names..."
    python3 <<EOF
import os
import re
import random
import string

def random_name(length=8):
    """Generate a random alphanumeric name."""
    return ''.join(random.choices(string.ascii_letters, k=length))

def rename_smali(smali_folder):
    """Rename class names, methods, and fields in Smali files."""
    class_mapping = {}

    for root, _, files in os.walk(smali_folder):
        for file in files:
            if file.endswith(".smali"):
                smali_file = os.path.join(root, file)
                with open(smali_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Rename class names (excluding system classes)
                class_matches = re.findall(r'\.class.*? L([\w/$]+);', content)
                for class_name in class_matches:
                    if not class_name.startswith("android") and class_name not in class_mapping:
                        new_name = "L" + random_name(10)
                        class_mapping[class_name] = new_name
                        content = content.replace(class_name, new_name)

                # Rename method names
                content = re.sub(r'(\.method.*? )(\w+)(\()', lambda m: m.group(1) + random_name(8) + m.group(3), content)

                # Rename field (variable) names
                content = re.sub(r'(\.field.*? )(\w+)(:)', lambda m: m.group(1) + random_name(8) + m.group(3), content)

                with open(smali_file, "w", encoding="utf-8") as f:
                    f.write(content)

rename_smali("$DECOMPILED_DIR/smali")
EOF

    echo "    [*] Recompiling APK..."
    apktool b "$DECOMPILED_DIR" -o "$OUTPUT_APK" > /dev/null 2>&1

    echo "[+] Process complete! Obfuscated APK saved at: $OUTPUT_APK"

    rm -rf "$DECOMPILED_DIR"
done

echo "[*] All APKs processed. Obfuscated APKs saved in $OUTPUT_DIR"
