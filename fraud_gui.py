import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load model, scaler, and encoders
try:
    model = joblib.load("fraud_Model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")  # Load saved encoders
    if model is None or scaler is None or label_encoders is None:
        raise ValueError("One or more required files are missing or corrupted.")
except (FileNotFoundError, ValueError):
    messagebox.showerror("Error", "Model, Scaler, or Encoder file not found! Ensure all files are present.")
    exit()

# Feature names
feature_names = ['cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender', 'street', 
                 'city', 'state', 'job', 'trans_num', 'year', 'month', 'day', 'hour', 'minute']

# Categorical features that require encoding
categorical_features = ['merchant', 'category', 'first', 'last', 'gender', 'street', 
                        'city', 'state', 'job', 'trans_num']

def predict_fraud():
    input_data = []
    
    try:
        # Collect numerical inputs
        input_data.append(float(cc_num_entry.get()))  # Credit card number
        input_data.append(float(amt_entry.get()))  # Transaction amount
        input_data.append(float(year_entry.get()))  
        input_data.append(float(month_entry.get()))  
        input_data.append(float(day_entry.get()))  
        input_data.append(float(hour_entry.get()))  
        input_data.append(float(minute_entry.get()))  
    except ValueError:
        messagebox.showerror("Input Error", "Invalid numerical input! Please enter valid numbers.")
        return

    # Collect categorical inputs and encode them safely
    try:
        for feature in categorical_features:
            entry_widget = globals().get(f"{feature}_entry")
            value = entry_widget.get().strip().lower()

            if feature in label_encoders:
                encoder = label_encoders[feature]
                if value in encoder.classes_:  
                    input_data.append(encoder.transform([value])[0])
                else:
                    input_data.append(0)  # Assign default label if unseen
            else:
                input_data.append(0)  # Assign default if no encoder exists
    except Exception as e:
        messagebox.showerror("Encoding Error", f"Error encoding categorical data: {str(e)}")
        return

    # Convert input data to 2D numpy array
    input_array = np.array(input_data).reshape(1, -1)

    # Scale input data
    try:
        input_scaled = scaler.transform(input_array)
    except Exception as e:
        messagebox.showerror("Scaling Error", f"Error scaling input data: {str(e)}")
        return

    # Make prediction
    try:
        prediction = model.predict(input_scaled)[0]  
        if (prediction == 0):
            result = "Non-Fraudulent Transaction"
        else:
            result = "Fraudulent Transaction"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error in model prediction: {str(e)}")

def clear_entries():
    for entry in entries:
        entry.delete(0, tk.END)

# GUI Setup
root = tk.Tk()
root.title("Credit Card Fraud Detection")
root.geometry("500x600")

# Scrollable Frame
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill="y")

# Entry fields
entries = []
tk.Label(scrollable_frame, text="Enter Transaction Details:", font=("Arial", 12, "bold")).pack(pady=10)

field_names = [
    ("Credit Card no", "cc_num_entry"),
    ("Merchant", "merchant_entry"),
    ("Category", "category_entry"),
    ("Amount", "amt_entry"),
    ("First Name", "first_entry"),
    ("Last Name", "last_entry"),
    ("Gender", "gender_entry"),
    ("Street", "street_entry"),
    ("City", "city_entry"),
    ("State", "state_entry"),
    ("Job", "job_entry"),
    ("Transaction no", "trans_num_entry"),
    ("Year", "year_entry"),
    ("Month", "month_entry"),
    ("Day", "day_entry"),
    ("Hour", "hour_entry"),
    ("Minute", "minute_entry"),
]

# Dynamically create label and entry fields
for label, var_name in field_names:
    tk.Label(scrollable_frame, text=label + ":").pack()
    globals()[var_name] = tk.Entry(scrollable_frame)
    globals()[var_name].pack()
    entries.append(globals()[var_name])

# Buttons
button_frame = tk.Frame(scrollable_frame)
button_frame.pack(pady=10)
tk.Button(button_frame, text="Check Transaction", command=predict_fraud, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Reset", command=clear_entries, bg="red", fg="white").pack(side=tk.RIGHT, padx=5)

# Run GUI
root.mainloop()
