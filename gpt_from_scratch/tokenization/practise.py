with open("gpt_from_scratch/tokenization/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print(len(raw_text))

import re
# x = re.split(r'([,.-?:;()"\'_=+[]{}<>`~]|--|\s)', raw_text)
x = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
x_new = [item for item in x if item.split()]
# print(x_new[:30])

all_words = sorted(list(set(x_new)))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
print(vocab)
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i>30:
#         break

