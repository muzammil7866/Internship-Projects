"""
Advanced Password Generator with GUI
Features: Multiple character sets, password strength indicator,
custom patterns, history, clipboard support
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import string
import json
import os
import pyperclip
from datetime import datetime


class PasswordStrength:
    @staticmethod
    def calculate_strength(password):
        """Calculate password strength (0-100)"""
        score = 0
        
        # Length bonus
        if len(password) >= 8:
            score += 10
        if len(password) >= 12:
            score += 10
        if len(password) >= 16:
            score += 10
        
        # Character variety
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 15
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in string.punctuation for c in password):
            score += 25
        
        return min(score, 100)

    @staticmethod
    def get_strength_label(score):
        """Get strength label based on score"""
        if score < 30:
            return ("Very Weak", "#d32f2f")
        elif score < 50:
            return ("Weak", "#f57c00")
        elif score < 70:
            return ("Fair", "#fbc02d")
        elif score < 85:
            return ("Strong", "#7cb342")
        else:
            return ("Very Strong", "#388e3c")


class PasswordGenerator:
    def __init__(self):
        self.history_file = "password_history.json"
        self.history = []
        self.load_history()

    def generate_password(self, length, use_uppercase, use_lowercase, 
                         use_digits, use_special, custom_chars=""):
        """Generate a password based on specified criteria"""
        characters = ""
        
        if use_lowercase:
            characters += string.ascii_lowercase
        if use_uppercase:
            characters += string.ascii_uppercase
        if use_digits:
            characters += string.digits
        if use_special:
            characters += string.punctuation
        if custom_chars:
            characters += custom_chars
        
        if not characters:
            raise ValueError("Please select at least one character type")
        
        password = ''.join(random.choice(characters) for _ in range(length))
        return password

    def generate_multiple(self, count, length, use_uppercase, use_lowercase,
                         use_digits, use_special, custom_chars=""):
        """Generate multiple passwords"""
        passwords = []
        try:
            for _ in range(count):
                pwd = self.generate_password(length, use_uppercase, use_lowercase,
                                            use_digits, use_special, custom_chars)
                passwords.append(pwd)
            return passwords
        except ValueError as e:
            raise e

    def add_to_history(self, password):
        """Add password to history"""
        entry = {
            "password": password,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strength": PasswordStrength.calculate_strength(password)
        }
        self.history.insert(0, entry)
        self.save_history()

    def clear_history(self):
        """Clear password history"""
        self.history = []
        self.save_history()

    def save_history(self):
        """Save history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history[:50], f, indent=4)  # Keep last 50

    def load_history(self):
        """Load history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)


class PasswordGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Password Generator")
        self.root.geometry("950x800")
        self.root.configure(bg="#f5f5f5")
        
        self.generator = PasswordGenerator()
        self.current_password = ""
        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#1a237e", height=80)
        header_frame.pack(fill=tk.X)
        
        title = tk.Label(header_frame, text="🔐 Advanced Password Generator", 
                        font=("Helvetica", 20, "bold"), fg="white", bg="#1a237e")
        title.pack(pady=20)

        # Main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Password Configuration", padding=15)
        config_frame.pack(fill=tk.X, pady=10)

        # Length settings
        length_frame = ttk.Frame(config_frame)
        length_frame.pack(fill=tk.X, pady=10)

        ttk.Label(length_frame, text="Password Length:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        self.length_var = tk.StringVar(value="16")
        length_spin = ttk.Spinbox(length_frame, from_=4, to=128, textvariable=self.length_var, width=8)
        length_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(length_frame, text="Number of Passwords:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=20)
        self.count_var = tk.StringVar(value="1")
        count_spin = ttk.Spinbox(length_frame, from_=1, to=10, textvariable=self.count_var, width=8)
        count_spin.pack(side=tk.LEFT, padx=5)

        # Character type checkboxes
        char_frame = ttk.LabelFrame(config_frame, text="Character Types", padding=10)
        char_frame.pack(fill=tk.X, pady=10)

        self.use_lowercase = tk.BooleanVar(value=True)
        self.use_uppercase = tk.BooleanVar(value=True)
        self.use_digits = tk.BooleanVar(value=True)
        self.use_special = tk.BooleanVar(value=True)

        ttk.Checkbutton(char_frame, text="a-z (Lowercase)", variable=self.use_lowercase).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(char_frame, text="A-Z (Uppercase)", variable=self.use_uppercase).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(char_frame, text="0-9 (Digits)", variable=self.use_digits).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(char_frame, text="!@#$%^&* (Special)", variable=self.use_special).pack(anchor=tk.W, pady=5)

        # Custom characters
        custom_frame = ttk.Frame(config_frame)
        custom_frame.pack(fill=tk.X, pady=10)
        ttk.Label(custom_frame, text="Custom Characters (optional):").pack(anchor=tk.W)
        self.custom_chars_entry = ttk.Entry(custom_frame, width=50)
        self.custom_chars_entry.pack(anchor=tk.W, pady=5)

        # Generate button
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        generate_btn = ttk.Button(button_frame, text="🔄 Generate Password(s)", 
                                  command=self.generate_password)
        generate_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        reset_btn = ttk.Button(button_frame, text="↺ Reset", command=self.reset_form)
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Password display section
        display_frame = ttk.LabelFrame(main_frame, text="Generated Password(s)", padding=15)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Password list
        scrollbar = ttk.Scrollbar(display_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.password_listbox = tk.Listbox(display_frame, yscrollcommand=scrollbar.set, 
                                           font=("Courier", 11), height=8)
        self.password_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.password_listbox.yview)

        self.password_listbox.bind('<<ListboxSelect>>', self.on_password_select)

        # Password details
        details_frame = ttk.LabelFrame(display_frame, text="Password Details", padding=10)
        details_frame.pack(fill=tk.X, pady=10)

        self.strength_label = ttk.Label(details_frame, text="Strength: N/A", font=("Helvetica", 10, "bold"))
        self.strength_label.pack(anchor=tk.W)

        self.strength_bar = ttk.Progressbar(details_frame, mode='determinate')
        self.strength_bar.pack(fill=tk.X, pady=5)

        self.details_text = ttk.Label(details_frame, text="", font=("Helvetica", 9))
        self.details_text.pack(anchor=tk.W)

        # Action buttons for password
        action_frame = ttk.Frame(display_frame)
        action_frame.pack(fill=tk.X, pady=10)

        copy_btn = ttk.Button(action_frame, text="📋 Copy to Clipboard", command=self.copy_to_clipboard)
        copy_btn.pack(side=tk.LEFT, padx=5)

        info_btn = ttk.Button(action_frame, text="ℹ️  Password Info", command=self.show_password_info)
        info_btn.pack(side=tk.LEFT, padx=5)

        # History section
        history_frame = ttk.LabelFrame(main_frame, text="Password History", padding=15)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        columns = ("Timestamp", "Password", "Strength")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, height=6, show="headings")
        
        self.history_tree.column("Timestamp", width=180)
        self.history_tree.column("Password", width=300)
        self.history_tree.column("Strength", width=100)
        
        self.history_tree.heading("Timestamp", text="Timestamp")
        self.history_tree.heading("Password", text="Password")
        self.history_tree.heading("Strength", text="Strength")
        
        scrollbar_history = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar_history.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_history.pack(side=tk.RIGHT, fill=tk.Y)

        # History buttons
        history_btn_frame = ttk.Frame(history_frame)
        history_btn_frame.pack(fill=tk.X, pady=10)

        refresh_history_btn = ttk.Button(history_btn_frame, text="🔄 Refresh History", 
                                         command=self.load_history_display)
        refresh_history_btn.pack(side=tk.LEFT, padx=5)

        clear_history_btn = ttk.Button(history_btn_frame, text="🗑️  Clear History", 
                                       command=self.clear_history)
        clear_history_btn.pack(side=tk.LEFT, padx=5)

        # Initial load
        self.load_history_display()

    def generate_password(self):
        try:
            length = int(self.length_var.get())
            count = int(self.count_var.get())
            
            if length < 4:
                messagebox.showerror("Error", "Password length must be at least 4")
                return
            
            passwords = self.generator.generate_multiple(
                count, length,
                self.use_uppercase.get(),
                self.use_lowercase.get(),
                self.use_digits.get(),
                self.use_special.get(),
                self.custom_chars_entry.get()
            )

            self.password_listbox.delete(0, tk.END)
            for pwd in passwords:
                self.password_listbox.insert(tk.END, pwd)
                self.generator.add_to_history(pwd)

            if passwords:
                self.password_listbox.select_set(0)
                self.on_password_select(None)
            
            self.load_history_display()
            messagebox.showinfo("Success", f"Generated {count} password(s) successfully!")

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def on_password_select(self, event):
        """Handle password selection"""
        selection = self.password_listbox.curselection()
        if selection:
            self.current_password = self.password_listbox.get(selection[0])
            self.update_password_details()

    def update_password_details(self):
        """Update password strength and details"""
        if not self.current_password:
            return
        
        strength_score = PasswordStrength.calculate_strength(self.current_password)
        strength_label, color = PasswordStrength.get_strength_label(strength_score)
        
        self.strength_label.config(text=f"Strength: {strength_label} ({strength_score}%)")
        self.strength_bar['value'] = strength_score
        
        details = f"Length: {len(self.current_password)} | "
        details += f"Uppercase: {sum(1 for c in self.current_password if c.isupper())} | "
        details += f"Lowercase: {sum(1 for c in self.current_password if c.islower())} | "
        details += f"Digits: {sum(1 for c in self.current_password if c.isdigit())} | "
        details += f"Special: {sum(1 for c in self.current_password if c in string.punctuation)}"
        
        self.details_text.config(text=details)

    def copy_to_clipboard(self):
        """Copy selected password to clipboard"""
        selection = self.password_listbox.curselection()
        if selection:
            password = self.password_listbox.get(selection[0])
            try:
                pyperclip.copy(password)
                messagebox.showinfo("Success", "Password copied to clipboard!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy: {str(e)}")

    def show_password_info(self):
        """Show detailed password information"""
        if not self.current_password:
            messagebox.showwarning("Warning", "No password selected")
            return
        
        strength_score = PasswordStrength.calculate_strength(self.current_password)
        strength_label, _ = PasswordStrength.get_strength_label(strength_score)
        
        info = f"""
Password Analysis:
═════════════════════════════════════

Password: {self.current_password}

Length: {len(self.current_password)} characters

Character Composition:
  • Uppercase: {sum(1 for c in self.current_password if c.isupper())}
  • Lowercase: {sum(1 for c in self.current_password if c.islower())}
  • Digits: {sum(1 for c in self.current_password if c.isdigit())}
  • Special: {sum(1 for c in self.current_password if c in string.punctuation)}

Strength: {strength_label} ({strength_score}%)

Recommendations:
  • Use at least 12 characters
  • Mix uppercase, lowercase, digits, and special characters
  • Avoid common words and patterns
        """
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Password Information")
        info_window.geometry("500x450")
        
        text_widget = tk.Text(info_window, font=("Courier", 10), border=0)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert("1.0", info)
        text_widget.config(state=tk.DISABLED)

    def load_history_display(self):
        """Load and display password history"""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        for entry in self.generator.history:
            strength_label, _ = PasswordStrength.get_strength_label(entry["strength"])
            self.history_tree.insert("", tk.END, values=(
                entry["timestamp"],
                entry["password"][:30] + "..." if len(entry["password"]) > 30 else entry["password"],
                strength_label
            ))

    def clear_history(self):
        """Clear password history with confirmation"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all history?"):
            self.generator.clear_history()
            self.load_history_display()
            messagebox.showinfo("Success", "History cleared successfully!")

    def reset_form(self):
        """Reset form to default values"""
        self.length_var.set("16")
        self.count_var.set("1")
        self.use_lowercase.set(True)
        self.use_uppercase.set(True)
        self.use_digits.set(True)
        self.use_special.set(True)
        self.custom_chars_entry.delete(0, tk.END)
        self.password_listbox.delete(0, tk.END)
        self.current_password = ""
        self.strength_label.config(text="Strength: N/A")
        self.strength_bar['value'] = 0
        self.details_text.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = PasswordGeneratorGUI(root)
    root.mainloop()
    
    


