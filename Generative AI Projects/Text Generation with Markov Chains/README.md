# Text Generation with Markov Chains

This project implements a simple yet effective text generation algorithm using **Markov Chains**. It demonstrates the fundamental concepts of statistical language modeling without using deep neural networks.

## Project Overview

A Markov chain is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. In the context of text generation:
- The **State** is the current sequence of $n$ words (an n-gram).
- The **Transition** is the probability of the next word appearing after that sequence.

By training on a source text, the model learns which words tend to follow others and uses this to generate plausible-sounding new text.

### How It Works
1.  **Ingestion**: The script reads a text file (`sample_data.txt`).
2.  **Training**: It builds a dictionary where keys are sequences of words (e.g., "The sky") and values are lists of words that followed them in the source text (e.g., ["was", "turned", "is"]).
3.  **Generation**: It picks a starting point and repeatedly rolls the dice to choose the next word based on the possibilities observed during training.

## Business Goals Achieved

While simple, this technology represents the foundational logic behind many business applications:

1.  **Chatbot Personality Design**:
    -   *Goal*: Create non-deterministic, varied responses for NPCs in games or simple chatbots.
    -   *Application*: Ensures a bot doesn't say the exact same phrase every time, adding "life" to interactions.

2.  **Data Augmentation**:
    -   *Goal*: Generate synthetic data for testing software.
    -   *Application*: Creating thousands of unique, valid-looking addresses, names, or sentence structures to stress-test UI layouts or database fields.

3.  **Creative Writing Assistants**:
    -   *Goal*: Help overcome writer's block.
    -   *Application*: Suggesting the next possible word or sentence completion based on the writer's own style history.

4.  **Procedural Content Generation**:
    -   *Goal*: Automatically generate descriptions, lore, or quests in video games.
    -   *Application*: "A [rusty/shiny] [sword/shield] found in the [cave/forest]."

## Usage

1.  Ensure you have Python installed.
2.  Place your source text in `sample_data.txt` (or use the provided sample).
3.  Run the script:
    ```bash
    python markov_chain.py
    ```
4.  Modify the `order` variable in the script to change coherence:
    -   **Order 1**: More chaotic, less coherent.
    -   **Order 2+**: More coherent, but may copy source text verbatim if the dataset is small.
