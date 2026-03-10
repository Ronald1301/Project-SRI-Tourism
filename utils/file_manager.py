"""
Utility helpers for saving and loading index artifacts.

Allowed dependencies: os, json, pickle, numpy.
"""

import json
import os
import pickle

import numpy as np


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
