import pandas as pd
import os
from collections import Counter
import json


def process_csvs(base_directory):
    folders = ['train', 'test', 'val']
    word_counts = Counter()

    for folder in folders:
        csv_path = os.path.join(base_directory, f'How2Sign/sentence_level/{folder}/text/how2sign_{folder}.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            continue

        # Load the CSV file
        data = pd.read_csv(csv_path, sep='\t')
        # Concatenate all sentences into a single string, then split into words
        sentences = ' '.join(data['SENTENCE'].dropna()).lower()  # Use lower to normalize the case
        words = sentences.split()
        word_counts.update(words)

    return word_counts

def get_top_words(word_counts, top_n=100):
    return word_counts.most_common(top_n)


def save_to_json(data, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data saved to {filepath}")


def main():
    base_directory = 'h2signdata'  # Modify this path to the base directory where your data folders are located
    word_counts = process_csvs(base_directory)
    wc = dict(word_counts)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    # Save results to JSON
    json_path = 'word_frequencies.json'
    save_to_json(dict(sorted_words), json_path)
    top_words = get_top_words(word_counts, top_n=500)
    sorted_top_words = sorted(top_words, key=lambda x: x[1], reverse=True)
    json_path2 = 'top_100word_frequencies.json'
    save_to_json(dict(sorted_top_words), json_path2)
    # Display the total number of unique words
    print(f"Total unique words: {len(word_counts)}")
    print(f"Top unique words: {len(top_words)}")


if __name__ == "__main__":
    main()
