from collections import Counter

def pre_process(text):
    text = text.lower().split()
    return text, Counter(text)

def create_ngrams(text, n):
    temp = zip(*[text[i:] for i in range(0, n)])
    return [" ".join(ngram) for ngram in temp]

def freq_ngram(text, n):
    tokens, count_tokens = pre_process(text)
    ngrams = create_ngrams(tokens, n)
    count_ngrams = Counter(ngrams)
    return count_tokens, count_ngrams

def calculate_probabilities(count_tokens, count_ngrams, n):
    probabilities = {}
    
    for ngram, count in count_ngrams.items():
        words = ngram.split()
        context = " ".join(words[:-1])
        next_word = words[-1]
        
        if context not in probabilities:
            probabilities[context] = {}
        
        # For unigrams, calculate based on total words count
        if n == 1:
            probabilities[context][next_word] = count / sum(count_tokens.values())
        else:
            # For n-grams, calculate based on context count
            context_count = count_tokens[context] if context in count_tokens else sum(count_tokens.values())
            probabilities[context][next_word] = count / context_count
    
    return probabilities

def make_prediction(probabilities, context):
    context_words = context.split()
    
    # Try to find the exact context in probabilities
    if context in probabilities:
        next_word_probs = probabilities[context]
        return max(next_word_probs, key=next_word_probs.get)
    
    # If exact match not found, try to find the highest probability next word
    if len(context_words) > 1:
        for i in range(len(context_words) - 1, 0, -1):
            short_context = " ".join(context_words[:i])
            if short_context in probabilities:
                next_word_probs = probabilities[short_context]
                return max(next_word_probs, key=next_word_probs.get)
    
    return None

# Example usage
sample = "I love watching cricket I love playing cricket"
count_tokens, count_bigrams = freq_ngram(sample, 2)
bigram_probabilities = calculate_probabilities(count_tokens, count_bigrams, 2)

print("Words Count: ", count_tokens)
print("\nBigrams Count: ", count_bigrams)
print("\nBigram Probabilities: ", bigram_probabilities)

# Predicting next word for different contexts
contexts = ["i", "love","watching cricket"]

for context in contexts:
    predicted_word = make_prediction(bigram_probabilities, context)
    print(f"\nNext word prediction for '{context}': {predicted_word}")
