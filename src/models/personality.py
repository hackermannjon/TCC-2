"""
Calcula traços Big-5 simples a partir de texto usando contagem de palavras-chave.
Retorna vetor (5,) normalizado por nº de tokens.
"""
import numpy as np
import re

# mini-lexicon (exemplo)  —  você pode substituir por um dicionário maior
LEXICON = {
    "openness"      : ["creative", "imaginative", "curious", "novel", "art", "music"],
    "conscientious" : ["organized", "punctual", "hardworking", "careful", "plan"],
    "extraversion"  : ["outgoing", "party", "friends", "energy", "talk"],
    "agreeable"     : ["kind", "sympathetic", "warm", "caring", "help"],
    "neurotic"      : ["anxious", "stress", "worried", "nervous", "depressed"]
}

DIM_ORDER = ["openness","conscientious","extraversion","agreeable","neurotic"]

token_re = re.compile(r"[A-Za-z]+")

def big5_from_text(text: str) -> np.ndarray:
    tokens = [t.lower() for t in token_re.findall(text)]
    if not tokens:
        return np.zeros(5, dtype="float32")
    counts = []
    for dim in DIM_ORDER:
        hits = sum(1 for t in tokens if t in LEXICON[dim])
        counts.append(hits / len(tokens))
    return np.array(counts, dtype="float32")
