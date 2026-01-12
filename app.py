import os
import gensim
from nltk import sent_tokenize
from gensim.utils import simple_preprocess

# -----------------------------
# Path to GOT text files folder
# -----------------------------
DATA_PATH = r"C:\Users\guddu\OneDrive\Desktop\gameofthrons\gameofthrons"

story = []

# -----------------------------
# Read and preprocess text
# -----------------------------
for filename in os.listdir(DATA_PATH):
    file_path = os.path.join(DATA_PATH, filename)

    if filename.endswith(".txt"):
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            corpus = f.read()

        sentences = sent_tokenize(corpus)

        for sent in sentences:
            story.append(simple_preprocess(sent))

print("Total sentences:", len(story))

# -----------------------------
# Train Word2Vec model
# -----------------------------
model = gensim.models.Word2Vec(
    sentences=story,
    vector_size=100,
    window=10,
    min_count=2,
    workers=4,
    epochs=10
)

# -----------------------------
# Example outputs
# -----------------------------
print("\nOdd one out:")
print(model.wv.doesnt_match(['jon', 'rikon', 'arya', 'sansa', 'bran']))

print("\nSimilar to 'daenerys':")
print(model.wv.most_similar('daenerys', topn=5))





# Gensim via Colab...

# !pip install gensim nltk


# import numpy as np 
# import pandas as pd 

# import gensim 
# import os 
# import nltk

# from nltk import sent_tokenize 
# from gensim.utils import simple_preprocess 

# nltk.download('punkt')
# nltk.download('punkt_tab')

# story = []
# for filename in os.listdir('/content/sample_data/gameofthrons'):
#     f = open(os.path.join("/content/sample_data/gameofthrons" , filename)) 
#     corpus = f.read() 
#     raw_sent = sent_tokenize(corpus) 
#     for sent in raw_sent :
#         story.append(simple_preprocess(sent)) 
        

# print(len(story)) 

# ### story 
# model = gensim.models.Word2Vec(
#     window = 10 ,
#     min_count = 2
# )

# model.build_vocab(story) 

# print(model.train(story, total_examples = model.corpus_count , epochs = model.epochs)) 

# print(model.wv.most_similar('daenerys'))

# print(model.wv.doesnt_match(['jon' , 'rikon' , 'arya' , 'sansa' , 'bran']) )


# model.wv.doesnt_match(['cersei' , 'jaime' , 'bronn' , 'tyrion']) 

# print(model.wv['jon']) 