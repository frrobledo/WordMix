import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_nouns():
    """Load a set of English nouns from WordNet."""
    nouns = set()
    for synset in wn.all_synsets('n'):
        for lemma in synset.lemmas():
            word = lemma.name()
            if word.isalpha():
                nouns.add(word.lower())
    return nouns

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

def main():
    # Load nouns and embeddings
    """
    Entry point for the Word Mixing Game.

    This script will load a set of nouns and their corresponding GloVe embeddings,
    then repeatedly prompt the user to select two numbers between 1 and 10 to mix.
    The script will then calculate the sum of the two corresponding word embeddings,
    and find the word in the original set that is closest to the resulting vector,
    excluding the original two words. The closest word is then printed to the user.

    The user may enter 'Q' to quit the game.
    """
    print("Loading nouns and embeddings...")
    nouns = load_nouns()

    try:
        embeddings = load_glove_embeddings('glove.6B.100d.txt', nouns)
    except FileNotFoundError:
        print("Glove embeddings file not found. Please ensure 'glove.6B.100d.txt' is in the current directory.")
        return

    if not embeddings:
        print("No embeddings loaded. Please ensure 'glove.6B.100d.txt' is in the current directory.")
        return

    all_words = list(embeddings.keys())
    all_embeddings = np.array([embeddings[word] for word in all_words])

    print("Welcome to the Word Mixing Game!")
    while True:
        # Select 10 random words
        words = np.random.choice(all_words, 10, replace=False)
        print("\nHere are your words:")
        for idx, word in enumerate(words, 1):
            print(f"{idx}): {word}")

        user_input = input("\nEnter two numbers between 1 and 10 to mix, or 'Q' to quit: ").strip()
        if user_input.upper() == 'Q':
            print("Thanks for playing!")
            break
        else:
            try:
                nums = user_input.replace(',', ' ').split()
                if len(nums) != 2:
                    raise ValueError("Please enter exactly two numbers.")
                num1, num2 = int(nums[0]), int(nums[1])
                if num1 < 1 or num1 > 10 or num2 < 1 or num2 > 10:
                    raise ValueError("Numbers must be between 1 and 10.")
                word1 = words[num1 - 1]
                word2 = words[num2 - 1]
                print(f"You selected: {word1} and {word2}")

                vec1 = embeddings[word1]
                vec2 = embeddings[word2]
                sum_vec = vec1 + vec2

                # Find the closest word, excluding the original words
                result_word = find_closest_word(
                    sum_vec, embeddings, all_embeddings, all_words, exclude_words=[word1, word2]
                )
                print(f"The resulting word is: {result_word}")

            except Exception as e:
                print(f"Error: {e}. Please try again.")

if __name__ == "__main__":
    main()
