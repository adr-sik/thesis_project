import os
import subprocess

def decompress_apks(input_directory, output_directory):
    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            if file_name.endswith(".apk"):
                print("Decompressing files")
                apk_path = os.path.join(root, file_name)
                decompress_path = os.path.join(output_directory, file_name.replace(".apk", "-decompressed"))
                
                try:
                    result = subprocess.run(
                        ["java", "-jar", "C:\\Windows\\baksmali.jar", "disassemble", apk_path, "-o", decompress_path],
                        capture_output=True,
                        text=True
                    )
                    print("Success")
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
                except subprocess.CalledProcessError as e:
                    with open("decompression_errors.log", "a") as log_file:
                        log_file.write(f"Error decompressing {apk_path}:\n{e.stderr}\n")
    return 0
    
def decompress_apk(apk_path, output_directory):
    if apk_path.endswith(".apk"):
        try:
            result = subprocess.run(
                ["java", "-jar", "C:\\Windows\\baksmali.jar", "disassemble", apk_path, "-o", output_directory],
                capture_output=True,
                text=True
            )
            print("Success")
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
        except subprocess.CalledProcessError as e:
            with open("decompression_errors.log", "a") as log_file:
                log_file.write(f"Error decompressing {apk_path}:\n{e.stderr}\n")
    
def main():
    input_directory = ""
    output_directory = ""
    decompress_apks(input_directory, output_directory)
    print("All files decompressed!")

if __name__ == "__main__":
    main()