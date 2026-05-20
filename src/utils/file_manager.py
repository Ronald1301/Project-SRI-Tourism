"""
Utility helpers for saving and loading index artifacts.

Allowed dependencies: os, json, pickle, numpy.
"""

import json
import os
import pickle

import numpy as np
from pathlib import Path


def _ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_pickle(obj, path):
    _ensure_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(array, path):
    _ensure_parent_dir(path)
    np.save(path, array)


def load_numpy(path):
    return np.load(path, allow_pickle=False)

def save_documents_to_jsonl(documents: list[dict[str,object]], output_file: Path) -> None:
    """Guarda los resultados en formato JSONL (una línea JSON por resultado)"""
    with open(output_file, 'a', encoding='utf-8') as f:
        for document in documents:
            f.write(json.dumps(document, ensure_ascii=False) + '\n')

def load_visited_urls(path: Path) -> set[str]:
    urls: set[str] = set()
    try:
        if not path.exists():
            return urls
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                url = line.strip()
                if url:
                    urls.add(url)
    except OSError:
        return urls
    return urls
