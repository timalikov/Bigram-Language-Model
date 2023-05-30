words = open("names.txt", "r").read().splitlines()

b = {}
for w in words[]:
    chs = ["^"] + list(w) + ["$"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        print(ch1, ch2)

