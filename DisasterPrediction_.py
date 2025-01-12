import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Global model and column names for manual input
model = None
columns_to_exclude = ['title', 'date_time', 'net', 'magType', 'location', 'continent', 'country', 'alert']
feature_columns = []

def select_file():
    """Open a file dialog to select a file."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

def train_model():
    """Train the model and display results."""
    global model, feature_columns

    train_file = select_file()
    if not train_file:
        messagebox.showerror("Error", "No training file selected!")
        return

    try:
        # Load and preprocess data
        data = pd.read_csv(train_file)
        feature_columns = [col for col in data.columns if col not in columns_to_exclude + ['magnitude']]
        X = data[feature_columns]
        y = data['magnitude']

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate performance
        mse = mean_squared_error(y_test, predictions)
        messagebox.showinfo("Training Complete", f"Model trained successfully!\nMean Squared Error: {mse}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def predict_from_file():
    """Predict on new data from a file."""
    if model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return

    predict_file = select_file()
    if not predict_file:
        return

    try:
        test_data = pd.read_csv(predict_file)
        X_new = test_data[feature_columns]
        predictions = model.predict(X_new)

        # Display predictions
        test_data["Predicted Magnitude"] = predictions
        messagebox.showinfo("Predictions", f"Predictions:\n{test_data[['Predicted Magnitude']].to_string(index=False)}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def predict_from_input():
    """Predict using manually entered input values."""
    if model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return

    try:
        # Create a new window for manual input
        input_window = tk.Toplevel(root)
        input_window.title("Manual Input Prediction")

        input_values = []

        def submit_inputs():
            """Process the manually entered values and make predictions."""
            try:
                input_data = [[float(entry.get()) for entry in input_values]]
                prediction = model.predict(input_data)[0]
                messagebox.showinfo("Prediction", f"Predicted Magnitude: {prediction}")
                input_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid input! Please enter numeric values.")

        # Create input fields for each feature
        for i, feature in enumerate(feature_columns):
            tk.Label(input_window, text=f"{feature}:", font=("Arial", 10)).grid(row=i, column=0, padx=10, pady=5, sticky="e")
            entry = tk.Entry(input_window)
            entry.grid(row=i, column=1, padx=10, pady=5)
            input_values.append(entry)

        # Submit button
        tk.Button(input_window, text="Submit", command=submit_inputs, bg="#f57c00", fg="white", font=("Arial", 12)).grid(row=len(feature_columns), column=0, columnspan=2, pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main UI
root = tk.Tk()
root.title("Disaster Prediction")
root.geometry("800x600")
root.configure(bg="#f9f9f9")

# Header Section
header_frame = tk.Frame(root, bg="#0277bd")
header_frame.pack(fill="x")
header_label = tk.Label(header_frame, text="Disaster Prevention", font=("Arial", 24), bg="#0277bd", fg="white")
header_label.pack(pady=10)
sub_header_label = tk.Label(header_frame, text="Predict the probability of Disaster Occurrence", font=("Arial", 14), bg="#0277bd", fg="white")
sub_header_label.pack()

# Input Section
input_frame = tk.Frame(root, bg="#f9f9f9")
input_frame.pack(pady=20)

# Buttons
train_button = tk.Button(input_frame, text="Train Model", command=train_model, bg="#f57c00", fg="white", font=("Arial", 12))
train_button.grid(row=0, column=0, padx=20, pady=10)

file_predict_button = tk.Button(input_frame, text="Predict from File", command=predict_from_file, bg="#f57c00", fg="white", font=("Arial", 12))
file_predict_button.grid(row=0, column=1, padx=20, pady=10)

input_predict_button = tk.Button(input_frame, text="Predict from Input", command=predict_from_input, bg="#f57c00", fg="white", font=("Arial", 12))
input_predict_button.grid(row=0, column=2, padx=20, pady=10)

# Footer Section
footer_frame = tk.Frame(root, bg="#ffcc80")
footer_frame.pack(side="bottom", fill="x")
company_bio = tk.Label(footer_frame, text="Company Bio", bg="#ffcc80", font=("Arial", 12))
company_bio.pack(side="left", padx=20, pady=10)
settings = tk.Label(footer_frame, text="Settings", bg="#ffcc80", font=("Arial", 12))
settings.pack(side="left", padx=20, pady=10)
connect = tk.Label(footer_frame, text="Connect", bg="#ffcc80", font=("Arial", 12))
connect.pack(side="right", padx=20, pady=10)

root.mainloop()
