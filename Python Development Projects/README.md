# Python Development Projects

Three professional GUI applications evolved from basic internship tasks.

---

## 📋 Project 1: Advanced To-Do List Manager

**Location**: `advanced_todo_manager/`

Comprehensive task management with priorities, categories, and persistence.

**Key Features**:
- Add, update, delete, and mark tasks complete
- Priority levels (Low, Medium, High) and 5 categories
- Task descriptions and due dates
- Task statistics (Total, Completed, Pending)
- JSON persistence automatically saves data
- Professional dark-themed GUI

**Requirements**: `pip install tkcalender`

**Run**: `python advanced_todo_manager/advanced_todo_manager.py`

---

## 🎮 Project 2: Advanced Rock Paper Scissors Game

**Location**: `rock_paper_scissors_game/`

Interactive game with AI opponent and difficulty levels.

**Key Features**:
- Three difficulty modes (Easy, Medium, Hard)
- Play best-of-N rounds (configurable 1-20)
- Game history and round-by-round details
- Win rate calculation and all-time statistics tracking
- Visual result display with emoji indicators

**Run**: `python rock_paper_scissors_game/rock_paper_scissors_game.py`

---

## 🔐 Project 3: Advanced Password Generator

**Location**: `advanced_password_generator/`

Secure password generation with strength analysis and history.

**Key Features**:
- Customizable character sets (uppercase, lowercase, digits, special)
- Strength score (0-100) with visual indicator
- Generate multiple passwords at once
- Password history tracking (last 50 passwords)
- Copy to clipboard, detailed analysis, and security recommendations

**Requirements**: `pip install pyperclip`

**Run**: `python advanced_password_generator/advanced_password_generator.py`

---

## 📁 Directory Structure

```
Python Development Projects/
├── README.md
├── requirements.txt
├── advanced_todo_manager/
│   ├── advanced_todo_manager.py
│   └── tasks_data.json (auto-generated)
├── rock_paper_scissors_game/
│   ├── rock_paper_scissors_game.py
│   └── game_stats.json (auto-generated)
└── advanced_password_generator/
    ├── advanced_password_generator.py
    └── password_history.json (auto-generated)
```

---

## 🚀 Quick Start

**Requirements**: Python 3.6+

```bash
# Install dependencies 
pip install -r requirements.txt

# Run any application from the main directory
python advanced_todo_manager/advanced_todo_manager.py
python rock_paper_scissors_game/rock_paper_scissors_game.py
python advanced_password_generator/advanced_password_generator.py
```

Data files are created automatically on first run.

---

## 💾 Data Files

All applications save data automatically in JSON format:
- **To-Do Manager**: `tasks_data.json` - Tasks with priority, category, due dates
- **Rock Paper Scissors**: `game_stats.json` - Win/loss/tie statistics
- **Password Generator**: `password_history.json` - Generated passwords with timestamps

---

## 🎓 What Each Project Demonstrates

| Project | Key Concepts | Technologies |
|---------|-------------|--------------|
| To-Do Manager | OOP, Persistence, State Management | tkinter, JSON, Enums |
| Rock Paper Scissors | Game Logic, AI Strategy, Statistics | tkinter, Random, State Enums |
| Password Generator | Algorithm, Analysis, User Input | tkinter, String Operations, Strength Scoring |

---

## 📝 Technical Details

- **GUI Framework**: tkinter (built-in with Python)
- **Data Storage**: JSON files for persistence
- **Architecture**: Object-oriented with separation of concerns
- **Error Handling**: Input validation and user feedback
- **Status**: Active & maintained
 