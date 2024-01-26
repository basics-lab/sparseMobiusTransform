import mobiusmodule
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
from smt.utils import dec_to_bin
from tqdm import tqdm

# sentence = "I have never forgotten this movie."
sentence = "He never fails to disappoint."
# sentence = "The only good element is that it doesn’t take itself seriously."

words = np.array(sentence.split(" "))
n = len(words)

model_path = "JiaqiLee/imdb-finetuned-bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


def sampling_function(k):
    partial_sentence = " ".join(words[k == 1])
    output = pipeline(partial_sentence)[0]
    # print(partial_sentence, output)
    if output["label"] == "positive":
        return output["score"]
    else:
        return 1 - output["score"]


x = np.zeros(2 ** n)
x[0] = 0.5
for k_dec in tqdm(range(1, 2 ** n)):
    k = dec_to_bin(k_dec, n)
    x[k_dec] = sampling_function(k)

x = 2 * (x - 0.5)

mobiusmodule.mobius(x)

x_m = x.copy()

print(x_m)

sorted_indices = x_m.argsort()

x_m_degree = [{} for _ in range(n+1)]


for k_dec in range(2 ** n):
    k = dec_to_bin(k_dec, n)
    x_m_degree[int(np.sum(k == 1))][k_dec] = x_m[k_dec]

for i in range(n+1):
    x_m_degree[i] = dict(sorted(x_m_degree[i].items(), key=lambda item: item[1]))
    print("-------")
    print(f"---- Degree {i} Coefficients ---")
    for item in x_m_degree[i].items():
        print(dec_to_bin(item[0], n), ", ", item[1])

exit()

for i in range(10):
    print(dec_to_bin(sorted_indices[i], n), ", ", mobius_score[sorted_indices[i]])

print("-------------")

for i in range(1, 11):
    print(dec_to_bin(sorted_indices[-i], n), ", ", mobius_score[sorted_indices[-i]])

print("-------------")
#
# print(pipeline(""))
# print(pipeline("have"))
# print(pipeline("never"))
# print(pipeline("have never"))
#
# print("-------------")
#
# for i in tqdm(range(2 ** n)):
#     print(dec_to_bin(i, n), ", ", x[i])
