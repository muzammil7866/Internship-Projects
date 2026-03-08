"""
Advanced Rock Paper Scissors Game with GUI
Features: Best of N rounds, difficulty levels, statistics, game history
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import json
import os
from datetime import datetime
from enum import Enum


class Difficulty(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class GameState(Enum):
    IDLE = "idle"
    PLAYING = "playing"
    ROUND_OVER = "round_over"
    GAME_OVER = "game_over"


class RockPaperScissorsGame:
    def __init__(self):
        self.choices = ['Rock', 'Paper', 'Scissors']
        self.user_score = 0
        self.computer_score = 0
        self.round_number = 0
        self.total_rounds = 1
        self.difficulty = Difficulty.MEDIUM
        self.game_history = []
        self.stats_file = "game_stats.json"
        self.load_stats()

    def get_computer_choice(self):
        if self.difficulty == Difficulty.EASY:
            return random.choice(self.choices)
        elif self.difficulty == Difficulty.MEDIUM:
            return random.choice(self.choices)
        else:  # HARD
            # Computer tries to win more often in hard mode
            return random.choices(self.choices, weights=[3, 3, 1])[0]

    def play_round(self, user_choice):
        computer_choice = self.get_computer_choice()
        result = self.determine_winner(user_choice, computer_choice)
        
        round_data = {
            "round": self.round_number,
            "user_choice": user_choice,
            "computer_choice": computer_choice,
            "result": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.game_history.append(round_data)
        
        return computer_choice, result

    def determine_winner(self, user, computer):
        if user == computer:
            return "tie"
        elif (user == 'Rock' and computer == 'Scissors') or \
             (user == 'Paper' and computer == 'Rock') or \
             (user == 'Scissors' and computer == 'Paper'):
            self.user_score += 1
            return "win"
        else:
            self.computer_score += 1
            return "loss"

    def reset_game(self):
        self.user_score = 0
        self.computer_score = 0
        self.round_number = 0
        self.game_history = []

    def is_game_over(self):
        return self.round_number >= self.total_rounds

    def get_statistics(self):
        return {
            "user_score": self.user_score,
            "computer_score": self.computer_score,
            "round": self.round_number,
            "total_rounds": self.total_rounds
        }

    def save_stats(self):
        stats = {
            "games_played": len([g for g in self.game_history if g.get("game_end")]),
            "win_rate": self.calculate_win_rate(),
            "total_wins": sum(1 for g in self.game_history if g.get("result") == "win"),
            "total_losses": sum(1 for g in self.game_history if g.get("result") == "loss"),
            "total_ties": sum(1 for g in self.game_history if g.get("result") == "tie")
        }
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {"all_time": {"wins": 0, "losses": 0, "ties": 0}}
        
        existing["all_time"]["wins"] += stats["total_wins"]
        existing["all_time"]["losses"] += stats["total_losses"]
        existing["all_time"]["ties"] += stats["total_ties"]
        
        with open(self.stats_file, 'w') as f:
            json.dump(existing, f, indent=4)

    def load_stats(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                self.all_time_stats = json.load(f).get("all_time", {})
        else:
            self.all_time_stats = {"wins": 0, "losses": 0, "ties": 0}

    def calculate_win_rate(self):
        total = len(self.game_history)
        if total == 0:
            return 0
        wins = sum(1 for g in self.game_history if g.get("result") == "win")
        return (wins / total) * 100


class RockPaperScissorsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Rock Paper Scissors Game")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a1a")
        
        self.game = RockPaperScissorsGame()
        self.game_state = GameState.IDLE
        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        
        title = tk.Label(header_frame, text="🎮 Advanced Rock Paper Scissors", 
                        font=("Helvetica", 20, "bold"), fg="white", bg="#2c3e50")
        title.pack(pady=15)

        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Setup frame
        setup_frame = ttk.LabelFrame(main_frame, text="Game Setup", padding=15)
        setup_frame.pack(fill=tk.X, pady=10)

        ttk.Label(setup_frame, text="Difficulty:").grid(row=0, column=0, padx=5)
        self.difficulty_var = tk.StringVar(value=Difficulty.MEDIUM.value)
        difficulty_combo = ttk.Combobox(setup_frame, textvariable=self.difficulty_var,
                                       values=[d.value for d in Difficulty], state="readonly", width=15)
        difficulty_combo.grid(row=0, column=1, padx=5)

        ttk.Label(setup_frame, text="Number of Rounds:").grid(row=0, column=2, padx=5)
        self.rounds_var = tk.StringVar(value="5")
        rounds_spin = ttk.Spinbox(setup_frame, from_=1, to=20, textvariable=self.rounds_var, width=15)
        rounds_spin.grid(row=0, column=3, padx=5)

        self.start_btn = ttk.Button(setup_frame, text="▶️  Start Game", command=self.start_game)
        self.start_btn.grid(row=0, column=4, padx=5)

        # Game area
        game_area = ttk.LabelFrame(main_frame, text="Game Area", padding=20)
        game_area.pack(fill=tk.BOTH, expand=True, pady=10)

        # Current round info
        info_frame = ttk.Frame(game_area)
        info_frame.pack(fill=tk.X, pady=10)
        self.round_label = ttk.Label(info_frame, text="", font=("Helvetica", 14, "bold"))
        self.round_label.pack()

        # Game choices
        choice_frame = ttk.Frame(game_area)
        choice_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        # User choice buttons
        user_frame = ttk.LabelFrame(choice_frame, text="Your Choice", padding=15)
        user_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.rock_btn = ttk.Button(user_frame, text="🪨 Rock", command=lambda: self.make_choice("Rock"))
        self.rock_btn.pack(fill=tk.BOTH, expand=True, pady=10)

        self.paper_btn = ttk.Button(user_frame, text="📄 Paper", command=lambda: self.make_choice("Paper"))
        self.paper_btn.pack(fill=tk.BOTH, expand=True, pady=10)

        self.scissors_btn = ttk.Button(user_frame, text="✂️  Scissors", command=lambda: self.make_choice("Scissors"))
        self.scissors_btn.pack(fill=tk.BOTH, expand=True, pady=10)

        # Result display
        result_frame = ttk.LabelFrame(choice_frame, text="Result", padding=15)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.result_text = tk.Text(result_frame, height=8, width=35, font=("Courier", 11))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Score display
        score_frame = ttk.LabelFrame(main_frame, text="Scores", padding=15)
        score_frame.pack(fill=tk.X, pady=10)

        self.score_label = ttk.Label(score_frame, text="", font=("Helvetica", 12, "bold"))
        self.score_label.pack()

        # Game history
        history_frame = ttk.LabelFrame(main_frame, text="Game History", padding=10)
        history_frame.pack(fill=tk.X, pady=10)

        columns = ("Round", "You", "Computer", "Result")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, height=5, show="headings")
        for col in columns:
            self.history_tree.column(col, width=100)
            self.history_tree.heading(col, text=col)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.end_game_btn = ttk.Button(control_frame, text="🛑 End Game", command=self.end_game, state=tk.DISABLED)
        self.end_game_btn.pack(side=tk.LEFT, padx=5)

        stats_btn = ttk.Button(control_frame, text="📊 Statistics", command=self.show_statistics)
        stats_btn.pack(side=tk.LEFT, padx=5)

        self.disable_game_buttons()

    def disable_game_buttons(self):
        self.rock_btn.config(state=tk.DISABLED)
        self.paper_btn.config(state=tk.DISABLED)
        self.scissors_btn.config(state=tk.DISABLED)
        self.end_game_btn.config(state=tk.DISABLED)

    def enable_game_buttons(self):
        self.rock_btn.config(state=tk.NORMAL)
        self.paper_btn.config(state=tk.NORMAL)
        self.scissors_btn.config(state=tk.NORMAL)
        self.end_game_btn.config(state=tk.NORMAL)

    def start_game(self):
        try:
            rounds = int(self.rounds_var.get())
            if rounds < 1:
                messagebox.showerror("Error", "Number of rounds must be at least 1")
                return
            
            difficulty_str = self.difficulty_var.get()
            self.game.difficulty = [d for d in Difficulty if d.value == difficulty_str][0]
            
            self.game.total_rounds = rounds
            self.game.reset_game()
            self.game_state = GameState.PLAYING
            self.result_text.delete("1.0", tk.END)
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            self.enable_game_buttons()
            self.update_round_label()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of rounds")

    def make_choice(self, choice):
        if self.game_state != GameState.PLAYING:
            return

        self.game.round_number += 1
        computer_choice, result = self.game.play_round(choice)
        
        self.display_round_result(choice, computer_choice, result)
        self.update_score_display()
        self.update_history_display()
        self.update_round_label()

        if self.game.is_game_over():
            self.game_state = GameState.GAME_OVER
            self.end_game()

    def display_round_result(self, user_choice, computer_choice, result):
        result_text = f"Your Choice: {user_choice}\n"
        result_text += f"Computer's Choice: {computer_choice}\n"
        result_text += "-" * 30 + "\n"
        
        if result == "win":
            result_text += "✅ YOU WIN THIS ROUND!\n"
        elif result == "loss":
            result_text += "❌ COMPUTER WINS THIS ROUND!\n"
        else:
            result_text += "🤝 IT'S A TIE!\n"
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", result_text)
        self.result_text.config(state=tk.DISABLED)

    def update_score_display(self):
        stats = self.game.get_statistics()
        score_text = f"🎯 You: {stats['user_score']} | 🤖 Computer: {stats['computer_score']} | Round: {stats['round']}/{stats['total_rounds']}"
        self.score_label.config(text=score_text)

    def update_round_label(self):
        self.round_label.config(text=f"Round {self.game.round_number} of {self.game.total_rounds}")

    def update_history_display(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        for game in self.game.game_history:
            result_display = "✅ Win" if game["result"] == "win" else ("❌ Loss" if game["result"] == "loss" else "🤝 Tie")
            self.history_tree.insert("", tk.END, values=(
                game["round"],
                game["user_choice"],
                game["computer_choice"],
                result_display
            ))

    def end_game(self):
        stats = self.game.get_statistics()
        self.game.save_stats()
        
        message = f"Game Over!\n"
        message += f"Final Score - You: {stats['user_score']} | Computer: {stats['computer_score']}\n\n"
        
        if stats['user_score'] > stats['computer_score']:
            message += "🎉 Congratulations! You Won!"
        elif stats['user_score'] < stats['computer_score']:
            message += "😔 Better luck next time!"
        else:
            message += "🤝 It was a perfect tie!"
        
        messagebox.showinfo("Game Over", message)
        self.disable_game_buttons()
        self.game_state = GameState.IDLE

    def show_statistics(self):
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Game Statistics")
        stats_window.geometry("400x300")
        
        stats_text = "📊 ALL-TIME STATISTICS\n"
        stats_text += "=" * 30 + "\n\n"
        stats_text += f"Total Wins: {self.game.all_time_stats.get('wins', 0)}\n"
        stats_text += f"Total Losses: {self.game.all_time_stats.get('losses', 0)}\n"
        stats_text += f"Total Ties: {self.game.all_time_stats.get('ties', 0)}\n\n"
        
        total = self.game.all_time_stats.get('wins', 0) + self.game.all_time_stats.get('losses', 0) + self.game.all_time_stats.get('ties', 0)
        if total > 0:
            win_rate = (self.game.all_time_stats.get('wins', 0) / total) * 100
            stats_text += f"Win Rate: {win_rate:.1f}%"
        
        text_widget = tk.Text(stats_window, font=("Courier", 12), border=0)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert("1.0", stats_text)
        text_widget.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = RockPaperScissorsGUI(root)
    root.mainloop()
                    
        
        
    

