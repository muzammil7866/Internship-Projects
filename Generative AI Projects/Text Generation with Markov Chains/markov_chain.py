import random
import sys

class MarkovChainGenerator:
    def __init__(self, order=1):
        """
        Initialize the Markov Chain Generator.
        :param order: The number of previous words to consider (the 'state').
        """
        self.order = order
        self.model = {}

    def train(self, text):
        """
        Builds the Markov chain model from the input text.
        :param text: The source text to train on.
        """
        words = text.split()
        if len(words) < self.order:
            return

        # Create n-grams
        for i in range(len(words) - self.order):
            state = tuple(words[i:i + self.order])
            next_word = words[i + self.order]

            if state not in self.model:
                self.model[state] = []
            self.model[state].append(next_word)

    def generate(self, length=50, start_prompt=None):
        """
        Generates text based on the learned probabilities.
        :param length: Approximate number of words to generate.
        :param start_prompt: Optional starting text.
        """
        if not self.model:
            return "Error: Model not trained."

        # Pick a random starting state
        current_state = random.choice(list(self.model.keys()))
        
        # If start_prompt is provided, try to find a matching state
        if start_prompt:
            prompt_words = start_prompt.split()
            if len(prompt_words) >= self.order:
                potential_state = tuple(prompt_words[-self.order:])
                if potential_state in self.model:
                    current_state = potential_state

        result = list(current_state)

        for _ in range(length):
            if current_state not in self.model:
                break
            
            next_word = random.choice(self.model[current_state])
            result.append(next_word)
            
            # Update state: drop the first word, append the new one
            current_state = tuple(result[-self.order:])

        # Join and return formatted text
        return ' '.join(result)

def main():
    # Configuration
    filename = "sample_data.txt"
    order = 2  # Higher order = more coherent but copies more verbatim
    length = 50

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please create it or adjust the path.")
        return

    print("Training Markov Chain Model...")
    generator = MarkovChainGenerator(order=order)
    generator.train(text)

    print(f"\n--- Generated Text (Order {order}) ---\n")
    generated_text = generator.generate(length=length)
    print(generated_text)
    print("\n-------------------------------------\n")

if __name__ == "__main__":
    main()
