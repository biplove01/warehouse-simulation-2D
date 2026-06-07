import re
import tkinter as tk
from tkinter import ttk, messagebox
import requests

# --- API CONFIGURATION ---
BASE_URL = "http://localhost:8080"
SYSTEM_KEY = "SystemKey illsontpygamesystem"

# Global variable to store JWT token after login if needed for future authenticated requests
jwt_token = None

# --- CORE LOGIC FUNCTIONS ---

def validate_email(email):
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_regex, email) is not None

def fetch_unpacked_count():
    """Fetches the count of unpacked items from the Spring Boot API."""
    url = f"{BASE_URL}/gui/order-count"
    headers = {"SystemKey": SYSTEM_KEY}
    
    try:
        # Performing GET request with custom security header
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("Count", 0) # Key matches your response format precisely
        else:
            return f"Error ({response.status_code})"
            
    except requests.exceptions.RequestException:
        return "Offline"

def attempt_login(event=None):
    global jwt_token
    email = email_entry.get().strip()
    password = password_entry.get()

    if not email or not validate_email(email):
        messagebox.showerror("Error", "Please enter a valid email address.")
        return
    if not password:
        messagebox.showerror("Error", "Password field cannot be empty.")
        return

    url = f"{BASE_URL}/api/v1/auth/admin/login"
    login_dto = {
        "email": email,
        "password": password
    }

    try:
        # Performing POST request with JSON payload
        response = requests.post(url, json=login_dto, timeout=5)
        
        if response.status_code == 200 or response.status_code == 401:
            data = response.json()
            
            # Verify the inner business responseCode
            if data.get("responseCode") == 200:
                jwt_token = data.get("jwtToken") # Saved in case you need it later
                
                # Transition Views cleanly
                root.unbind('<Return>') 
                login_frame.destroy()   
                load_dashboard_view()   
            else:
                # Displays exact failure message returned from Spring Boot
                error_msg = data.get("message", "Login Failed")
                messagebox.showerror("Login Failed", error_msg)
        else:
            messagebox.showerror("Server Error", f"Unexpected response status: {response.status_code}")

    except requests.exceptions.RequestException as err:
        messagebox.showerror("Connection Error", f"Could not reach Spring Boot backend.\nEnsure server is running on port 8080.\n\nDetails: {err}")


# --- VIEW BUILDERS ---

def load_login_view():
    global login_frame, email_entry, password_entry
    
    login_frame = ttk.Frame(root)
    login_frame.grid(row=0, column=0, sticky="NSEW")
    login_frame.columnconfigure(0, weight=1)
    login_frame.rowconfigure(0, weight=1)

    card = ttk.Frame(login_frame, padding=30, style="Card.TFrame")
    card.grid(row=0, column=0)

    ttk.Label(card, text="Admin System Portal", font=("Helvetica", 18, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 20))

    ttk.Label(card, text="Email Address:", font=("Helvetica", 11)).grid(row=1, column=0, sticky="W", pady=5)
    email_entry = ttk.Entry(card, font=("Helvetica", 11), width=35)
    email_entry.grid(row=1, column=1, pady=5, padx=(10, 0))
    email_entry.focus()

    ttk.Label(card, text="Password:", font=("Helvetica", 11)).grid(row=2, column=0, sticky="W", pady=5)
    password_entry = ttk.Entry(card, show="*", font=("Helvetica", 11), width=35)
    password_entry.grid(row=2, column=1, pady=5, padx=(10, 0))

    login_btn = ttk.Button(card, text="Secure Admin Sign In", command=attempt_login)
    login_btn.grid(row=3, column=0, columnspan=2, pady=(25, 0), sticky="EW")

    # Re-enable Enter key activation
    root.bind('<Return>', attempt_login)


def load_dashboard_view():
    """Builds a clean warehouse statistics dashboard grid."""
    dashboard_frame = ttk.Frame(root, padding=40)
    dashboard_frame.grid(row=0, column=0, sticky="NSEW")
    
    dashboard_frame.columnconfigure(0, weight=1)
    dashboard_frame.rowconfigure(1, weight=1) 

    # 1. Header Banner
    header_label = ttk.Label(dashboard_frame, text="Fulfillment Dashboard", font=("Helvetica", 24, "bold"))
    header_label.grid(row=0, column=0, sticky="W", pady=(0, 20))

    # 2. KPI Section (Fetches live data over network HTTP)
    kpi_frame = ttk.Frame(dashboard_frame, padding=30)
    kpi_frame.grid(row=1, column=0, sticky="NSEW")
    
    count_value = fetch_unpacked_count()
    
    count_label = ttk.Label(kpi_frame, text=str(count_value), font=("Helvetica", 72, "bold"), foreground="#2b5797")
    count_label.pack()
    
    text_label = ttk.Label(kpi_frame, text="Items Left to be Packed", font=("Helvetica", 14))
    text_label.pack(pady=10)

    # Action for the refresh button
    def refresh_dashboard():
        updated_count = fetch_unpacked_count()
        count_label.config(text=str(updated_count))

    # 3. Control Panel Buttons (Bottom Row)
    control_frame = ttk.Frame(dashboard_frame, padding=20)
    control_frame.grid(row=2, column=0, sticky="EW")
    control_frame.columnconfigure((0, 1, 2), weight=1, uniform="equal")

    # Refresh Button
    refresh_btn = ttk.Button(control_frame, text="🔄 Refresh Data", command=refresh_dashboard)
    refresh_btn.grid(row=0, column=0, padx=10, ipady=10, sticky="EW")

    # Start Warehouse Button (No active function yet)
    start_btn = ttk.Button(control_frame, text="🚀 Start Warehouse")
    start_btn.grid(row=0, column=1, padx=10, ipady=10, sticky="EW")

    # Exit Button
    exit_btn = ttk.Button(control_frame, text="❌ Exit System", command=root.destroy)
    exit_btn.grid(row=0, column=2, padx=10, ipady=10, sticky="EW")


# --- APPLICATION INITIALIZATION ---

root = tk.Tk()
root.title("Enterprise Logistics Panel")
root.geometry("1200x720")
root.minsize(900, 550) 

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

style = ttk.Style()
style.theme_use('clam') 
style.configure("Card.TFrame", background="#f4f4f4", relief="solid", borderwidth=1)

# Start with the API Login Screen
load_login_view()

root.mainloop()