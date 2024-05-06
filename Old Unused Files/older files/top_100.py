import json
from collections import Counter
import matplotlib.pyplot as plt

def load_word_frequencies(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return Counter(data)

def get_top_words(word_counts, top_n=100):
    return word_counts.most_common(top_n)

def plot_word_distribution(top_words):
    words, counts = zip(*top_words)
    plt.figure(figsize=(20, 16))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 100 Most Common Words')
    plt.xticks(rotation=90)  # Rotate the x labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated x labels
    plt.show()

def main():
    json_path = 'word_frequencies.json'
    word_counts = load_word_frequencies(json_path)
    top_100_words = get_top_words(word_counts, 100)
    plot_word_distribution(top_100_words)

if __name__ == "__main__":
    main()
