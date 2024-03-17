import pandas as pd

def split_sequece(sequence, length):
    parts = []
    for i in range(0, len(sequence), length):
        parts.append(sequence[i:i+length])
    return parts
