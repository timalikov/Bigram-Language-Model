import torch

# Read the words from the file and split them into a list of lines
words = open("names.txt", "r").read().splitlines()

N = torch.zeros((27,27), dtype=torch.int32)     #2D array to store bigrams: rows are for first character, columns are for second

chars = sorted(list(set(''.join(words))))       #set of all characters that appear in names
stoi = {s:i+1 for i,s in enumerate(chars)}      #creating integer representation of each character
stoi['^'] = 0                                   #special symbols for starting and ending points
itos = {i:s for s,i in stoi.items()}            #Create the inverse mapping of stoi to convert integer representation back to character

for w in words:
    chs = ["^"] + list(w) + ["^"]       # Add special symbols at the beginning and end of each word
    for ch1, ch2 in zip(chs, chs[1:]): 
        # Get the integer representation of the characters
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1                # Increment the count in the corresponding cell of the bigram matrix




######Generator
P = (N+1).float()               # Initialize the bigram probability matrix P
P /= P.sum(1, keepdims=True)

g = torch.Generator().manual_seed(2147483647)       # Set the seed for the random generator

out = []
ix = 0
for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
        break
  print(''.join(out))


###############################################################################
# Quality Check. GOAL: Maximize likelihood of the data (statistical modeling) #
# Maximize log likelihood (log is monotonic)                                  #
# Minimize negative log likelihood (equivalent to maximizing log likelihood)  #
# Minimize average negative log likelihood                                    #
# Property: log(a*b*c) = log(a) + log(b) + log(c)                             #
###############################################################################


log_likelihood = 0.0  
n = 0  # the count of bigrams

for w in words:
    chs = ['^'] + list(w) + ['^']
    
    for ch1, ch2 in zip(chs, chs[1:]):
        # Get the integer representation of the characters
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
        

# Print the log likelihood, negative log likelihood, and average negative log likelihood
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')




##########################################################################################################################
######################################                                           #########################################
######################################             Neural Network                #########################################
######################################                                           #########################################
##########################################################################################################################

#####################################################
# Creating the bigram dataset for the Neural Network#
#####################################################
# xs, ys = [], []

# for w in words[:1]:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         # print(ch1, ch2)
#         xs.append(ix1)
#         ys.append(ix2)
    
# xs = torch.tensor(xs)
# ys = torch.tensor(ys)
