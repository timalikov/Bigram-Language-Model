from bigram import *
import matplotlib.pyplot as plt
import textwrap


plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

# Set the colormap to 'Blues' for better contrast
plt.imshow(N, cmap='Blues')

# Iterate through the rows and columns of N
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        wrapped_text = textwrap.fill(chstr, 5)  # Wrap the text within 5 characters
        # Use smaller font size for better readability
        plt.text(j, i, chstr, ha="center", va="bottom", fontsize=6, color='black')  
        plt.text(j, i, N[i, j].item(), ha="center", va="top", fontsize=4, color='black')

        

plt.axis('off')
plt.tight_layout()  # Adjust the spacing between subplots

plt.show()
