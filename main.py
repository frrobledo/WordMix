import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('brown')

def load_common_nouns(min_freq=5):
    """Load a set of common English nouns based on frequency in the Brown Corpus."""
    # Get word frequencies from the Brown Corpus
    words = brown.words()
    freq_dist = nltk.FreqDist(w.lower() for w in words if w.isalpha())

    # Filter for nouns using WordNet and frequency threshold
    common_words = set()
    for word, freq in freq_dist.items():
        if freq >= min_freq:
            synsets = wn.synsets(word, pos=wn.NOUN)
            if synsets:
                common_words.add(word)
    return common_words

def load_glove_embeddings(glove_file_path, words_set):
    """Load GloVe embeddings for the specified set of words."""
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if word in words_set:
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
    return embeddings

def find_closest_word(target_vector, embeddings, all_embeddings, all_words, exclude_words):
    """Find the word closest to the target vector, excluding specified words."""
    similarities = cosine_similarity([target_vector], all_embeddings)[0]
    for word in exclude_words:
        if word in embeddings:
            idx = all_words.index(word)
            similarities[idx] = -np.inf
    idx = np.argmax(similarities)
    closest_word = all_words[idx]
    return closest_word

def get_word_input(prompt, words, embeddings):
    """Get valid word input from the user, either a number (1-10) or a custom word."""
    while True:
        user_input = input(prompt).strip()

        # Check if the user wants to quit
        if user_input.upper() == 'Q':
            return 'Q'

        # Check if the input is a number between 1 and 10
        if user_input.isdigit():
            num = int(user_input)
            if 1 <= num <= 10:
                return words[num - 1]  # Return the selected word
            else:
                print("Please enter a number between 1 and 10, or a valid word.")
        
        # Check if the input is a valid word in embeddings
        elif user_input.isalpha():
            if user_input in embeddings:
                return user_input  # Return the custom word
            else:
                print(f"'{user_input}' is not in the vocabulary. Try a different word.")
        else:
            print("Invalid input. Please enter a number (1-10), a word, or 'Q' to quit.")

def main():
    # Load common nouns and embeddings
    print("Loading common nouns and embeddings...")
    nouns = load_common_nouns(min_freq=5)
    embeddings = load_glove_embeddings('glove.6B.100d.txt', nouns)

    if not embeddings:
        print("No embeddings loaded. Please ensure 'glove.6B.100d.txt' is in the current directory.")
        return

    all_words = list(embeddings.keys())
    all_embeddings = np.array([embeddings[word] for word in all_words])

    print("Welcome to the Word Mixing Game!")
    previous_result = None  # Initialize previous result
    while True:
        # Select words for the current round
        if previous_result and previous_result in embeddings:
            # Exclude previous_result from random selection to avoid duplicates
            available_words = set(all_words) - {previous_result}
            words = np.random.choice(list(available_words), 9, replace=False)
            words = np.insert(words, 0, previous_result)
            # words = np.append(words, previous_result)  # Add previous_result to make 10 words
            # np.random.shuffle(words)  # Shuffle to randomize the position
        else:
            words = np.random.choice(all_words, 10, replace=False)

        print("\nHere are your words:")
        for idx, word in enumerate(words, 1):
            print(f"{idx}: {word}")

        # Get inputs from the user
        word1 = get_word_input("\nEnter the first word (number, word, or 'Q' to quit): ", words, embeddings)
        if word1 == 'Q':
            print("Thanks for playing!")
            break

        word2 = get_word_input("Enter the second word (number, word, or 'Q' to quit): ", words, embeddings)
        if word2 == 'Q':
            print("Thanks for playing!")
            break

        print(f"You selected: {word1} and {word2}")

        # Get embeddings for the selected words
        vec1 = embeddings[word1]
        vec2 = embeddings[word2]
        sum_vec = vec1 + vec2

        # Find the closest word, excluding the original words
        result_word = find_closest_word(
            sum_vec, embeddings, all_embeddings, all_words, exclude_words=[word1, word2]
        )
        print(f"The resulting word is: {result_word}")

        # Update previous_result for the next round
        previous_result = result_word

if __name__ == "__main__":
    main()
