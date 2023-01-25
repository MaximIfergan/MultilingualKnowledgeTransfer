import json
import pandas as pd
import spacy
import numpy as np

def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)

# print_title("Load models:")

nlp = spacy.load('en_core_web_sm')
# for name in nlp.pipe_names:
#     if name != 'ner':
#         nlp.remove_pipe(name)

# nlp = spacy.blank("en")
# nlp = spacy.blank("fr")
# nlp.add_pipe('dbpedia_spotlight')
# nlp.add_pipe('opentapioca')
# ner_pipe = nlp.create_pipe("ner")
# nlp.add_pipe("ner")
# doc = nlp('Le président des États-Unis appelle Boris Johnson pour décider quoi faire à propos du coronavirus')
# sent = 'The president of USA is calling Boris Johnson to decide what to do about coronavirus'
sent = "How long did it take the twin towers to be built?"
doc = nlp(sent)
# how long did it take the twin towers to be built?
# doc = nlp("when was the last time the lakers made the playoffs?")
# doc = nlp("Los Angeles Lakers")
# doc = nlp('putin was born in russia')
# doc = nlp('Chris Young is a famous singer')
# doc = nlp('twin towers')
# print('Entities', [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])
# for ent in doc.ents:
#     print(sent[:ent.start_char] + "[START] " + sent[ent.start_char:ent.end_char] + " [END]" + sent[ent.end_char:])

for token in doc:
    print(token.text, token.pos_, token.dep_)

# for token in doc:
#     print(token.text, token.pos_, token.tag_)
# print(doc.ents[0]._.dbpedia_raw_result)


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR)
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR).eval()
#
# sentences = []
# # sentences.append("[START] Einstein [END] era un fisico tedesco.")
# # sentences.append("[START] The [END] rain is fine")
# # sentences.append("how long did it take the [START] twin towers [END] to be built?")
# # sentences.append("who sings love you like there's no tomorrow? [START] Chris Young [END]")
# sentences.append("how long did it take [START] the twin towers [END] to be built?")
# outputs = model.generate(
#     **tokenizer(sentences, return_tensors="pt"),
#     num_beams=2,
#     num_return_sequences=1,
#     # OPTIONAL: use constrained beam search
#     # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
# )
#
# res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(res)

data = [json.loads(line) for line in open("Data/Datasets/NQ/nq_train_entities.jsonl", 'r', encoding='utf8')]
count = 0
for ex in data:
    if len(ex["q_entities"]) == 0 and len(ex["a_entities"]) == 0:
        count += 1

print(count)
print(count / 87925)
# 87925