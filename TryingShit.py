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
nlp.add_pipe('opentapioca')
# ner_pipe = nlp.create_pipe("ner")
# nlp.add_pipe("ner")
# doc = nlp('Le président des États-Unis appelle Boris Johnson pour décider quoi faire à propos du coronavirus')
# doc = nlp('The president of USA is calling Boris Johnson to decide what to do about coronavirus')
# doc = nlp("when was the last time the lakers made the playoffs?")
doc = nlp("Los Angeles Lakers")
# doc = nlp('putin was born in russia')
# doc = nlp('Chris Young is a famous singer')
# doc = nlp('twin towers')
print('Entities', [(ent.text, ent.label_, ent.kb_id_, ent.start, ent.end) for ent in doc.ents])
for token in doc:
    print(token.text, token.pos_, token.tag_)
# print(doc.ents[0]._.dbpedia_raw_result)


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import pickle
#
# # # OPTIONAL: load the prefix tree (trie), you need to additionally download
# # # https://huggingface.co/facebook/mgenre-wiki/blob/main/trie.py and
# # # https://huggingface.co/facebook/mgenre-wiki/blob/main/titles_lang_all105_trie_with_redirect.pkl
# # # that is fast but memory inefficient prefix tree (trie) -- it is implemented with nested python `dict`
# # # NOTE: loading this map may take up to 10 minutes and occupy a lot of RAM!
# # # import pickle
# # # from trie import Trie
# # # with open("titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
# # #     trie = Trie.load_from_dict(pickle.load(f))
# #
# # or a memory efficient but a bit slower prefix tree (trie) -- it is implemented with `marisa_trie` from
# # https://huggingface.co/facebook/mgenre-wiki/blob/main/titles_lang_all105_marisa_trie_with_redirect.pkl
# # from genre.trie import MarisaTrie
#
# # with open("../GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
# #     trie = pickle.load(f)
#
# tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval()
#
# # sentences = ["[START] Einstein [END] era un fisico tedesco."] # Italian for "[START] Einstein [END] was a German physicist."
# # sentences = ["how long did it take the [START] twin towers [END] to be built?"]
# # sentences = ["[START] who sings love you like there's no tomorrow?, Chris Young [END]"]
# # sentences = ["joe biden and putin are both president. And [START] he [END] is very tall"]
# sentences = ["when was the last time the [START] lakers [END] made the playoffs?"]
# # sentences = ["[START] Kodiak Valley Ski Resort [END]"]
#
# outputs = model.generate(
#     **tokenizer(sentences, return_tensors="pt"),
#     num_beams=5,
#     num_return_sequences=5,
#     # OPTIONAL: use constrained beam search
#     # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
# )
#
# res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(res)