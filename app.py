import os
import numpy as np
import tkinter as tk
import shutil
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model
from decompressor import decompress_apk
from feature_extractor import extract_features_test, get_ngrams, extract_opcodes
import threading

current_directory = os.getcwd()
temp_path = os.path.join(current_directory, "temp")
model = load_model(os.path.join(current_directory, "3_2048_model.keras"))
file_path = ""

def browse_file():
    label.config(text="")
    browse_button.config(state=tk.DISABLED)
    file_path = filedialog.askopenfilename(filetypes=[("APK files", "*.apk")])
    if file_path.endswith(".apk"):
        threading.Thread(target=process_file, args=(file_path,)).start()
    else:
        label.config(text="Invalid file type selected.", fg="red")
        browse_button.config(state=tk.NORMAL)

def process_file(file_path):
    try:
        decompress_apk(file_path, temp_path)
        data = np.array(extract_features_test(temp_path, 2048, 3))
        prediction = model.predict(data.reshape(1, -1))
        if prediction > 0.8:
            label.config(text=f"Confidence = {prediction}\n!!!MALWARE!!!", fg="red")
        else:
            label.config(text=f"Confidence = {prediction}\nBenign", fg="green")
    except Exception as e:
        label.config(text=f"Error: {str(e)}", fg="red")
    finally:
        shutil.rmtree(temp_path)
        browse_button.config(state=tk.NORMAL)

def main():
    print(current_directory)

    root = tk.Tk()
    root.title("Android Malware Classifier")

    bg_image = tk.PhotoImage(file=(os.path.join(current_directory, "icon.png")))

    canvas = tk.Canvas(root, width=bg_image.width(), height=bg_image.height())
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")

    center_x = bg_image.width() // 2
    canvas.create_text(center_x, 320, text="Provide path to your file.", fill="black", font=("Helvetica", 20), anchor="center")

    global browse_button
    browse_button = tk.Button(root, text="Click me", command=browse_file, width=20, height=2)
    browse_button.place(relx=0.5, rely=0.6, anchor="center")

    global label
    label = tk.Label(root, text="", font=("Helvetica", 22))
    label.place(relx=0.5, rely=0.92, anchor="center")

    root.mainloop()

if __name__ == "__main__":
    main()