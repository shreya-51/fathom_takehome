import numpy as np

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return lines

def get_embeddings(texts, model):
    return model.encode(texts, show_progress_bar=False)

def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename + '.npy')