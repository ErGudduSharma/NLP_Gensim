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
