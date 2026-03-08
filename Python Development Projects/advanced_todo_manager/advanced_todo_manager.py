"""
Advanced To-Do List Manager with GUI
Features: Add, delete, update, mark complete tasks
Also includes priority levels, categories, persistence, and datepicker
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime
from enum import Enum

try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Error", "Please install tkcalendar: pip install tkcalendar")
    exit()


class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TaskManager:
    def __init__(self):
        self.tasks = []
        self.task_id_counter = 1
        self.data_file = "tasks_data.json"
        self.load_tasks()

    def add_task(self, title, description, priority, category, due_date):
        task = {
            "id": self.task_id_counter,
            "title": title,
            "description": description,
            "priority": priority,
            "category": category,
            "due_date": due_date,
            "completed": False,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.tasks.append(task)
        self.task_id_counter += 1
        self.save_tasks()
        return task

    def delete_task(self, task_id):
        self.tasks = [task for task in self.tasks if task["id"] != task_id]
        self.save_tasks()

    def update_task(self, task_id, title, description, priority, category, due_date):
        for task in self.tasks:
            if task["id"] == task_id:
                task["title"] = title
                task["description"] = description
                task["priority"] = priority
                task["category"] = category
                task["due_date"] = due_date
                break
        self.save_tasks()

    def mark_complete(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = not task["completed"]
                break
        self.save_tasks()

    def get_statistics(self):
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t["completed"]])
        pending = total - completed
        return {"total": total, "completed": completed, "pending": pending}

    def save_tasks(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.tasks, f, indent=4)

    def load_tasks(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.tasks = data
                if self.tasks:
                    self.task_id_counter = max([t["id"] for t in self.tasks]) + 1


class TaskManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced To-Do List Manager")
        self.root.geometry("950x650") # Slightly wider to accommodate the new column
        self.root.configure(bg="#f0f0f0")
        
        self.manager = TaskManager()
        self.selected_task = None
        self.setup_ui()
        self.load_tasks_display()

    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(title_frame, text="📋 Advanced To-Do List Manager", 
                              font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=20)

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Task list
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        list_label = ttk.Label(left_frame, text="Tasks", font=("Helvetica", 12, "bold"))
        list_label.pack(pady=5)

        # Treeview for tasks (Added "Due Date" to columns)
        columns = ("ID", "Title", "Priority", "Category", "Due Date", "Status")
        self.tree = ttk.Treeview(left_frame, columns=columns, height=15, show="headings")
        
        # Adjust column widths so they fit nicely
        self.tree.column("ID", width=40, anchor=tk.CENTER)
        self.tree.column("Title", width=150)
        self.tree.column("Priority", width=80)
        self.tree.column("Category", width=90)
        self.tree.column("Due Date", width=100, anchor=tk.CENTER)
        self.tree.column("Status", width=80)

        for col in columns:
            self.tree.heading(col, text=col)
        
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_task_select)

        # Right panel - Actions and details
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Input fields
        input_frame = ttk.LabelFrame(right_frame, text="Task Details", padding=10)
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="Title:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.title_entry = ttk.Entry(input_frame, width=30)
        self.title_entry.grid(row=0, column=1, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.desc_text = tk.Text(input_frame, height=4, width=30)
        self.desc_text.grid(row=1, column=1, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Priority:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.priority_var = tk.StringVar(value=Priority.MEDIUM.value)
        priority_combo = ttk.Combobox(input_frame, textvariable=self.priority_var,
                                      values=[p.value for p in Priority], state="readonly", width=28)
        priority_combo.grid(row=2, column=1, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Category:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.category_var = tk.StringVar()
        category_combo = ttk.Combobox(input_frame, textvariable=self.category_var,
                                      values=["Work", "Personal", "Shopping", "Health", "Other"], state="readonly", width=28)
        category_combo.grid(row=3, column=1, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Due Date:").grid(row=4, column=0, sticky=tk.W, pady=5)
        
        self.due_date_entry = DateEntry(input_frame, width=28, background='darkblue',
                                        foreground='white', borderwidth=2,
                                        date_pattern='y-mm-dd', 
                                        mindate=datetime.now().date())
        self.due_date_entry.grid(row=4, column=1, pady=5, sticky=tk.W)

        # Buttons frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=10)

        add_btn = ttk.Button(button_frame, text="➕ Add Task", command=self.add_task)
        add_btn.pack(fill=tk.X, pady=5)

        update_btn = ttk.Button(button_frame, text="✏️  Update Task", command=self.update_task)
        update_btn.pack(fill=tk.X, pady=5)

        complete_btn = ttk.Button(button_frame, text="✅ Mark Complete", command=self.mark_complete)
        complete_btn.pack(fill=tk.X, pady=5)

        delete_btn = ttk.Button(button_frame, text="🗑️  Delete Task", command=self.delete_task)
        delete_btn.pack(fill=tk.X, pady=5)

        # Statistics frame
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)

        self.stats_label = ttk.Label(stats_frame, text="", font=("Helvetica", 10))
        self.stats_label.pack()

        # Bottom buttons
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.pack(fill=tk.X, pady=10)

        clear_btn = ttk.Button(bottom_frame, text="🔄 Refresh", command=self.load_tasks_display)
        clear_btn.pack(fill=tk.X, pady=5)

    def add_task(self):
        title = self.title_entry.get()
        description = self.desc_text.get("1.0", tk.END).strip()
        priority = self.priority_var.get()
        category = self.category_var.get()
        due_date = self.due_date_entry.get()

        if not title:
            messagebox.showerror("Error", "Please enter a task title")
            return

        self.manager.add_task(title, description, priority, category, due_date)
        messagebox.showinfo("Success", "Task added successfully!")
        self.clear_inputs()
        self.load_tasks_display()

    def update_task(self):
        if not self.selected_task:
            messagebox.showerror("Error", "Please select a task to update")
            return

        title = self.title_entry.get()
        description = self.desc_text.get("1.0", tk.END).strip()
        priority = self.priority_var.get()
        category = self.category_var.get()
        due_date = self.due_date_entry.get()

        if not title:
            messagebox.showerror("Error", "Please enter a task title")
            return

        self.manager.update_task(self.selected_task["id"], title, description, priority, category, due_date)
        messagebox.showinfo("Success", "Task updated successfully!")
        self.clear_inputs()
        self.load_tasks_display()

    def mark_complete(self):
        if not self.selected_task:
            messagebox.showerror("Error", "Please select a task")
            return

        self.manager.mark_complete(self.selected_task["id"])
        self.load_tasks_display()

    def delete_task(self):
        if not self.selected_task:
            messagebox.showerror("Error", "Please select a task to delete")
            return

        if messagebox.askyesno("Confirm", "Are you sure you want to delete this task?"):
            self.manager.delete_task(self.selected_task["id"])
            self.clear_inputs()
            self.load_tasks_display()

    def on_task_select(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            task_id = int(item["values"][0])
            self.selected_task = next((t for t in self.manager.tasks if t["id"] == task_id), None)
            
            if self.selected_task:
                self.title_entry.delete(0, tk.END)
                self.title_entry.insert(0, self.selected_task["title"])
                self.desc_text.delete("1.0", tk.END)
                self.desc_text.insert("1.0", self.selected_task["description"])
                self.priority_var.set(self.selected_task["priority"])
                self.category_var.set(self.selected_task["category"])
                
                try:
                    date_obj = datetime.strptime(self.selected_task["due_date"], "%Y-%m-%d").date()
                    self.due_date_entry.set_date(date_obj)
                except (ValueError, TypeError, KeyError):
                    self.due_date_entry.set_date(datetime.now().date())

    def load_tasks_display(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for task in self.manager.tasks:
            status = "✓ Done" if task.get("completed", False) else "⧗ Pending"
            # Extract due date safely just in case older JSON data doesn't have it
            due_date_str = task.get("due_date", "N/A") 
            
            self.tree.insert("", tk.END, values=(
                task["id"],
                task["title"][:15],
                task["priority"],
                task["category"],
                due_date_str, # Added to values tuple
                status
            ))

        stats = self.manager.get_statistics()
        self.stats_label.config(text=f"Total: {stats['total']} | Completed: {stats['completed']} | Pending: {stats['pending']}")

    def clear_inputs(self):
        self.title_entry.delete(0, tk.END)
        self.desc_text.delete("1.0", tk.END)
        self.priority_var.set(Priority.MEDIUM.value)
        self.category_var.set("")
        self.due_date_entry.set_date(datetime.now().date())
        self.selected_task = None


if __name__ == "__main__":
    root = tk.Tk()
    app = TaskManagerGUI(root)
    root.mainloop()