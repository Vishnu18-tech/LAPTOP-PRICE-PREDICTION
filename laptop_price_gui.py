from joblib import load
model = load("laptop_price_model.pkl")

import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd

root = tk.Tk()
root.title("💻 Laptop Price Predictor")
root.geometry("700x750")
root.configure(bg="#1e1e2f")

title_label = tk.Label(root, text="Laptop Price Prediction",
                       font=("Helvetica", 24, "bold"),
                       bg="#1e1e2f", fg="#00ffcc")
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#2b2b40")
frame.pack(pady=20, padx=20)

companies = ['HP', 'Dell', 'Lenovo', 'Acer', 'Apple', 'Asus']
types = ['Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible']
cpus = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor']
gpus = ['Intel', 'Nvidia', 'AMD']
oss = ['Windows', 'Mac', 'Others/No OS/Linux']

inputs = {}

def create_dropdown(label, values, row):
    tk.Label(frame, text=label, font=("Arial", 12), bg="#2b2b40", fg="white").grid(row=row, column=0, pady=8, padx=5, sticky="e")
    combo = ttk.Combobox(frame, values=values, font=("Arial", 12), state="readonly", justify="center")
    combo.grid(row=row, column=1, pady=8, padx=5, sticky="ew")
    inputs[label] = combo

def create_slider(label, from_, to, step, row):
    tk.Label(frame, text=label, font=("Arial", 12), bg="#2b2b40", fg="white").grid(row=row, column=0, pady=8, padx=5, sticky="e")
    scale_var = tk.IntVar()
    scale = tk.Scale(frame, from_=from_, to=to, orient="horizontal",
                     resolution=step, variable=scale_var, length=300,
                     bg="#2b2b40", fg="white", troughcolor="#00ffcc",
                     highlightthickness=0)
    scale.grid(row=row, column=1, pady=8, padx=5, sticky="ew")
    inputs[label] = scale_var

def create_toggle(label, row):
    tk.Label(frame, text=label, font=("Arial", 12, "bold"), bg="#2b2b40", fg="white").grid(row=row, column=0, pady=8, padx=5, sticky="e")
    var = tk.IntVar(value=0)
    toggle_frame = tk.Frame(frame, bg="#3b3b55", bd=2, relief="groove")
    toggle_frame.grid(row=row, column=1, pady=8, padx=5, sticky="ew")
    yes_btn = tk.Radiobutton(toggle_frame, text="Yes", variable=var, value=1,
                             bg="#3b3b55", fg="#00ffcc", font=("Arial", 12, "bold"),
                             selectcolor="#1e1e2f", indicatoron=0, width=8)
    no_btn = tk.Radiobutton(toggle_frame, text="No", variable=var, value=0,
                            bg="#3b3b55", fg="#ff4d4d", font=("Arial", 12, "bold"),
                            selectcolor="#1e1e2f", indicatoron=0, width=8)
    yes_btn.pack(side="left", padx=5, pady=5)
    no_btn.pack(side="left", padx=5, pady=5)
    inputs[label] = var

frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

create_dropdown("Company", companies, 0)
create_dropdown("TypeName", types, 1)
create_dropdown("Cpu Brand", cpus, 2)
create_dropdown("Gpu Brand", gpus, 3)
create_dropdown("Os", oss, 4)

create_slider("Ram (GB)", 2, 64, 2, 5)
create_slider("HDD (GB)", 0, 2000, 250, 6)
create_slider("SSD (GB)", 0, 2000, 128, 7)

create_toggle("Touchscreen", 8)
create_toggle("Ips", 9)

result_label = tk.Label(root, text="Predicted Price will appear here",
                        font=("Helvetica", 16, "bold"),
                        bg="#1e1e2f", fg="#ffcc00")
result_label.pack(pady=20)

def predict_price():
    try:
        new_laptop = pd.DataFrame([{
            "Company": inputs["Company"].get(),
            "TypeName": inputs["TypeName"].get(),
            "Cpu Brand": inputs["Cpu Brand"].get(),
            "Gpu Brand": inputs["Gpu Brand"].get(),
            "Os": inputs["Os"].get(),
            "Ram": inputs["Ram (GB)"].get(),
            "HDD": inputs["HDD (GB)"].get(),
            "SSD": inputs["SSD (GB)"].get(),
            "Touchscreen": inputs["Touchscreen"].get(),
            "Ips": inputs["Ips"].get()
        }])
        
        log_price = model.predict(new_laptop)
        price = np.exp(log_price[0])

        result_label.config(text=f"💰 Estimated Price: ₹{int(price):,}", fg="#00ffcc")
    except Exception as e:
        result_label.config(text=f"⚠️ Error: {str(e)}", fg="red")

def reset_fields():
    for key, widget in inputs.items():
        if isinstance(widget, tk.IntVar):
            widget.set(0)
        elif isinstance(widget, ttk.Combobox):
            widget.set('')
    result_label.config(text="Predicted Price will appear here", fg="#ffcc00")

btn_frame = tk.Frame(root, bg="#1e1e2f")
btn_frame.pack(pady=20)

predict_btn = tk.Button(btn_frame, text="Predict Price", command=predict_price,
                        font=("Arial", 14, "bold"), bg="#00ffcc", fg="black", width=15)
predict_btn.grid(row=0, column=0, padx=10)

reset_btn = tk.Button(btn_frame, text="Reset", command=reset_fields,
                      font=("Arial", 14, "bold"), bg="#ffcc00", fg="black", width=15)
reset_btn.grid(row=0, column=1, padx=10)

quit_btn = tk.Button(btn_frame, text="Exit", command=root.destroy,
                     font=("Arial", 14, "bold"), bg="#ff4d4d", fg="white", width=15)
quit_btn.grid(row=0, column=2, padx=10)

btn_frame.grid_columnconfigure(0, weight=1)
btn_frame.grid_columnconfigure(1, weight=1)
btn_frame.grid_columnconfigure(2, weight=1)

root.mainloop()