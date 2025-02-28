import os
import re
import subprocess
import sys
import shutil
import time

# === CONFIG ===
APKTOOL_PATH = "apktool"  # Ensure apktool is in PATH

# === FUNCTION: Print Progress Bar ===
def print_progress(current, total, task_name="Processing"):
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r[{bar}] {percent:.2f}% - {task_name} ({current}/{total})", end="", flush=True)

# === FUNCTION: Decompile APK ===
def decompile_apk(apk_file, output_dir):
    apk_name = os.path.splitext(os.path.basename(apk_file))[0]
    decompiled_path = os.path.join(output_dir, f"{apk_name}_decompiled")
    
    print(f"\n[+] Decompiling {apk_file}...")
    subprocess.run([APKTOOL_PATH, "d", apk_file, "-o", decompiled_path, "--force"], check=True)
    
    return decompiled_path

# === FUNCTION: Modify Smali for Reflection ===
def obfuscate_smali(decompiled_path):
    print(f"[+] Applying reflection obfuscation in {decompiled_path}...")
    
    smali_dir = os.path.join(decompiled_path, "smali")
    modified = False

    for root, _, files in os.walk(smali_dir):
        for file in files:
            if file.endswith(".smali"):
                smali_file = os.path.join(root, file)
                
                with open(smali_file, "r", encoding="utf-8") as f:
                    smali_code = f.read()
                
                # Find direct method calls
                method_calls = re.findall(r"(invoke-(?:virtual|direct|static|interface) \{[^}]+\}, L([^;]+);->([^()]+)\(([^)]*)\)([VZBCSIJFD]))", smali_code)

                if not method_calls:
                    continue  # No method calls found, skip this file

                for full_match, class_path, method_name, args, return_type in method_calls:
                    # Convert class path format Lcom/example/MyClass; -> com.example.MyClass
                    class_name = class_path.replace("/", ".")
                    
                    # Reflection-based method call replacement
                    reflection_code = f"""
    const-string v1, "{class_name}"
    invoke-static {{v1}}, Ljava/lang/Class;->forName(Ljava/lang/String;)Ljava/lang/Class;

    move-result-object v2
    const-string v3, "{method_name}"
    invoke-virtual {{v2, v3}}, Ljava/lang/Class;->getMethod(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;

    move-result-object v4
    const/4 v5, 0x0
    new-array v5, v5, [Ljava/lang/Object;
    invoke-virtual {{v4, v2, v5}}, Ljava/lang/reflect/Method;->invoke(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;
"""
                    smali_code = smali_code.replace(full_match, reflection_code)
                    modified = True

                if modified:
                    with open(smali_file, "w", encoding="utf-8") as f:
                        f.write(smali_code)
                    print(f"[+] Modified: {smali_file}")

# === FUNCTION: Rebuild APK ===
def rebuild_apk(decompiled_path, output_dir):
    apk_name = os.path.basename(decompiled_path).replace("_decompiled", "")
    obfuscated_apk = os.path.join(output_dir, f"{apk_name}_obfuscated.apk")
    
    print(f"[+] Rebuilding obfuscated APK: {obfuscated_apk}...")
    subprocess.run([APKTOOL_PATH, "b", decompiled_path, "-o", obfuscated_apk], check=True)

    return obfuscated_apk

# === FUNCTION: Process All APKs in Input Directory ===
def process_apks(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    apk_files = [f for f in os.listdir(input_dir) if f.endswith(".apk")]
    total_apks = len(apk_files)
    
    if not apk_files:
        print("[!] No APK files found in the input directory.")
        return

    print(f"[+] Found {total_apks} APKs. Starting obfuscation...\n")

    for index, apk_file in enumerate(apk_files, start=1):
        apk_path = os.path.join(input_dir, apk_file)

        # Update progress
        print_progress(index - 1, total_apks, "Decompiling APKs")
        decompiled_path = decompile_apk(apk_path, output_dir)

        print_progress(index - 1, total_apks, "Applying Obfuscation")
        obfuscate_smali(decompiled_path)

        print_progress(index - 1, total_apks, "Rebuilding APKs")
        rebuild_apk(decompiled_path, output_dir)

        print_progress(index, total_apks, "Completed")
        print(f"\n[✔] Finished processing: {apk_file}\n")

    print(f"[✔] All APKs processed! Check output directory: {output_dir}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_reflect_obfuscator.py <input_apk_directory> <output_directory>")
        sys.exit(1)

    input_apk_dir = sys.argv[1]
    output_apk_dir = sys.argv[2]

    if not os.path.exists(input_apk_dir):
        print(f"[!] Input directory '{input_apk_dir}' does not exist.")
        sys.exit(1)

    process_apks(input_apk_dir, output_apk_dir)

    print(f"[✔] All APKs processed! Check output directory: {output_apk_dir}")
