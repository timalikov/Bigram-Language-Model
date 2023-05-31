import torch
import torch.nn.functional as F

def read_words_from_file(filename):
    words = open(filename, "r").read().splitlines()
    return words



def compute_bigram_matrix(words):
    # Initialize the bigram matrix N with zeros
    N = torch.zeros((27, 27), dtype=torch.int32)

    # Get a sorted list of unique characters from all the words
    chars = sorted(list(set(''.join(words))))
    
    # Create a dictionary mapping each character to its integer representation (index)
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    
    # Add special symbol '^' with integer representation 0
    stoi['^'] = 0
    
    # Create the inverse mapping of stoi to convert integer representation back to character
    itos = {i: s for s, i in stoi.items()}
    
    # Iterate over each word in the list of words
    for w in words:
        # Add special symbols '^' at the beginning and end of each word
        chs = ["^"] + list(w) + ["^"]
        
        # Iterate over each pair of characters in the word
        for ch1, ch2 in zip(chs, chs[1:]):
            # Get the integer representations of the characters
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            
            # Increment the count in the corresponding cell of the bigram matrix
            N[ix1, ix2] += 1
    
    return N, stoi, itos



def generate_names(N, stoi, itos):
    # Initialize the bigram probability matrix P by adding 1 to N and converting to float
    P = (N + 1).float()
    
    # Normalize the rows of P to get probabilities
    P /= P.sum(1, keepdims=True)
    
    # Set the seed for the random generator
    g = torch.Generator().manual_seed(2147483647)
    
    out = []
    
    # Generate 5 names
    for i in range(5):
        generated_name = []
        ix = 0
        
        # Generate a name until the end symbol '^' is encountered
        while True:
            # Get the probability distribution for the next character based on current character
            p = P[ix]
            
            # Sample a character index based on the probability distribution
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            
            # Convert the character index back to the actual character and append to the generated name
            generated_name.append(itos[ix])
            
            # Check if the end symbol '^' is encountered
            if ix == 0:
                break
        
        # Add the generated name to the output list
        out.append(''.join(generated_name))
    
    return out



def compute_likelihood(words, stoi, P):
    log_likelihood = 0.0
    n = 0
    
    # Compute the log likelihood and count of bigrams
    for w in words:
        chs = ['^'] + list(w) + ['^']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            
            # Get the probability of the bigram
            prob = P[ix1, ix2]
            
            # Compute the logarithm of the probability
            logprob = torch.log(prob)
            
            # Accumulate the log likelihood
            log_likelihood += logprob
            
            # Increment the count of bigrams
            n += 1
    
    # Calculate the negative log likelihood and average negative log likelihood
    nll = -log_likelihood
    avg_nll = nll / n
    
    return log_likelihood.item(), nll.item(), avg_nll.item()


#####################################################
#                   Neural Network                  #
#####################################################


def create_bigram_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['^'] + list(w) + ['^']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    
    return xs, ys



def perform_gradient_descent(xs, ys):
    num_classes = 27
    num = xs.nelement()
    # print('number of examples:', num)

    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((num_classes, num_classes), generator=g, requires_grad=True)

    for k in range(1):
        xenc = F.one_hot(xs, num_classes=num_classes).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
        # print(loss.item())

        W.grad = None
        loss.backward()
        W.data += -50 * W.grad

    generated_names = []
    g = torch.Generator().manual_seed(2147483647)
    for i in range(5):
        generated_name = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=num_classes).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            generated_name.append(itos[ix])
            if ix == 0:
                break
        generated_names.append(''.join(generated_name))
    
    return generated_names

# Read the words from the file and split them into a list of lines
words = read_words_from_file("names.txt")

# Compute the bigram matrix
N, stoi, itos = compute_bigram_matrix(words)

# Generate names using the bigram matrix
generated_names = generate_names(N, stoi, itos)

# Compute likelihood of the data
log_likelihood, nll, avg_nll = compute_likelihood(words, stoi, (N+1).float() / (N+1).sum(1, keepdims=True))

# Create the bigram dataset for the neural network
xs, ys = create_bigram_dataset(words, stoi)

# Perform gradient descent
generated_names_nn = perform_gradient_descent(xs, ys)

# Print the generated names from the bigram model
print("Generated Names (Bigram Model):")
for name in generated_names:
    print(name)

# Print the generated names from the neural network model
print("Generated Names (Neural Network Model):")
for name in generated_names_nn:
    print(name)

# Print the log likelihood, negative log likelihood, and average negative log likelihood
print("Log Likelihood:", log_likelihood)
print("Negative Log Likelihood:", nll)
print("Average Negative Log Likelihood:", avg_nll)

