# NLP Gensim - Word2Vec on Game of Thrones

This project demonstrates how to train a Continuous Bag-of-Words (CBOW) / Skip-gram Word2Vec model using the `gensim` library. By leveraging `nltk` for sentence tokenization and `gensim` for preprocessing and embedding generation, the code trains a model to understand word relationships based on a custom text corpus—specifically, *Game of Thrones* text files.

## 🚀 Features

- **Text Preprocessing**: Reads raw text files, tokenizes them into sentences using `nltk`, and preprocesses each sentence into lowercase words using `gensim`'s `simple_preprocess`.
- **Custom Word Embeddings**: Trains a custom Word2Vec model to generate word vectors that capture semantic meanings.
- **Word Similarity Analysis**: Can perform tests to check the most similar words (e.g., related to 'daenerys') or find the odd-one-out in a list of characters.

## 📁 File Structure

```
NLP_Gensim/
│
├── gameofthrons/        # Directory containing your training text files (.txt)
├── app.py               # Main Python script for training and evaluating the Word2Vec model
├── main.py              # Secondary script (if applicable)
├── requirements.txt     # Python dependencies for the project
└── README.md            # Project documentation (this file)
```

## 🛠️ Prerequisites

Ensure you have Python installed on your system. You can install the required dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- `gensim`
- `nltk`
- `numpy`
- `pandas`

*Note: The script may require downloading NLTK tokenizers on the first run. You can add the following to your script if it throws an NLTK error:*
```python
import nltk
nltk.download('punkt')
```

## ⚙️ Usage

1. **Prepare your Data**: Place your text files inside a folder (e.g., `gameofthrons/`).
2. **Update Path**: If necessary, update the `DATA_PATH` variable in `app.py` to point to your directory of text files.
   ```python
   DATA_PATH = r"C:\Users\guddu\OneDrive\Desktop\gameofthrons\gameofthrons"
   # Or you can point it to the local 'gameofthrons' folder.
   ```
3. **Run the Script**:
   ```bash
   python app.py
   ```

## 📊 Example Output

When you run the script, it will parse the text, train the model, and then provide a few sample semantic relationship tests:

```text
Total sentences: 35000  # (Example number based on corpus)

Odd one out:
sansa

Similar to 'daenerys':
[('targaryen', 0.8923), ('stormborn', 0.8101), ('dragons', 0.7932), ('queen', 0.7410), ('viserys', 0.7100)]
```

## 📝 Notes

- The model uses `vector_size=100`, `window=10`, `min_count=2`, and `epochs=10` as default hyperparameters. Adjust these in `app.py` depending on the size of your corpus for better results.
- The commented code in `app.py` provides an easy-to-use template for running the exact same model inside **Google Colab**.
