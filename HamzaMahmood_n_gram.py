from nltk import word_tokenize

def find1(s,dct1):
    try:
        return dct1[s]
    except:
        return 0
def print_probability_table(distinct_tokens,dct,dct1):
    n=len(distinct_tokens)
    l=[[]*n for i in range(n)]
    for i in range(n):
        denominator = dct[distinct_tokens[i]]
        for j in range(n):
            numerator = find1(distinct_tokens[i]+" "+distinct_tokens[j],dct1)
            l[i].append(float("{:.3f}".format(numerator/denominator)))
    return l



def generate_n_gram_frequency(ngram):
    dict1 = {}
    for item in ngram:
        st = " ".join(item)
        dict1[st] = 0
    for item in ngram:
        st = " ".join(item)
        dict1[st] += 1
    return dict1
    


def generate_n_grams(tokens,k):
    n_gram_list = []
    i = 0
    while(i < len(tokens)):
        n_gram_list.append(tokens[i:i+k])
        i+=1
    return n_gram_list


def generate_token_frequency(tokens):
    dict1 = {}
    for item in tokens:
        dict1[item] = 0
    for item in tokens:
        dict1[item] += 1
    return dict1

def tokenize_data(data):
    data = word_tokenize(data)
    return data
def pre_process(data):
    data = data.lower()
    data = "eos " + data
    data = data.replace("."," eos")
    return data




# MAIN CODE
data = input("Enter the corpus:")
data = pre_process(data)
tokens = tokenize_data(data)
distinctive_tokens = list(set(sorted(tokens)))

print("List of sorted and distinctive tokens are:", distinctive_tokens)

token_frequency = generate_token_frequency(tokens)
print("Frequency of each token is equal to :")
for i in token_frequency.items():
    print(i[0] ,"\t",i[1])
bigram = generate_n_grams(tokens,2)

print("N-grams created using these tokens by taking n as 2:")
for i in bigram:
    print(i)

bigram_frequency = generate_n_gram_frequency(bigram)
print("The frequency of the N-gram is given below:")
for i in bigram_frequency.items():
    print(i[0],"\t",i[1])

print("Probability table = \n")
probability_table=print_probability_table(distinctive_tokens,token_frequency,bigram_frequency)
n=len(distinctive_tokens)
print("\t", end="")
for i in range(n):
    print(distinctive_tokens[i],end="\t")
print("\n")
for i in range(n):
    print(distinctive_tokens[i],end="\t")
    for j in range(n):
        print(probability_table[i][j],end="\t")
    print("\n")