"""Shared helpers for training and inference.

The model reads a word image from left to right and outputs a sequence of
"visual tokens". A visual token is either one of the 84 BanglaLekha-Isolated
character classes or a vowel sign (kar). The tokens are then rebuilt into a
proper Unicode Bangla word, so the app always shows the joined word.
"""

import json
import unicodedata

import cv2
import numpy as np

IMG_H = 64          # model input height
IMG_W = 256         # model input width
MAX_TOKENS = 12     # longest token sequence used in training
BLANK = 0           # CTC blank index


def nfc(text):
    return unicodedata.normalize("NFC", text)


# BanglaLekha-Isolated class folders 1..84, in folder order.
# 1-11 vowels, 12-50 consonants and signs, 51-60 digits, 61-84 compounds.
BL_CLASSES = [nfc(c) for c in [
    "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
    "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ",
    "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯",
    "ক্ষ", "ব্দ", "ঙ্গ", "স্ক", "স্ফ", "স্থ", "চ্ছ", "ক্ত", "স্ন", "ষ্ণ",
    "ম্প", "হ্ম", "প্ত", "ম্ব", "ন্ড", "দ্ভ", "ত্থ", "ষ্ঠ", "ল্প", "ষ্প",
    "ন্দ", "ন্ধ", "ম্ম", "ন্ঠ",
]]
assert len(BL_CLASSES) == 84

# Vowel signs (kar). They are not in BanglaLekha-Isolated, so train.py
# synthesises them. "pre" signs are written before the consonant,
# "post" after it, "below" under it. ো and ৌ are split into two parts.
MARKS = {
    "া": "post",   # া
    "ি": "pre",    # ি
    "ী": "post",   # ী
    "ু": "below",  # ু
    "ূ": "below",  # ূ
    "ৃ": "below",  # ৃ
    "ে": "pre",    # ে
    "ৈ": "pre",    # ৈ
    "ৗ": "post",   # ৗ (right part of ৌ)
}
O_KAR, OU_KAR = "ো", "ৌ"     # ো, ৌ
E_KAR, AA_KAR, AU_MARK = "ে", "া", "ৗ"

TOKENS = ["<blank>"] + BL_CLASSES + list(MARKS.keys())
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
NUM_CLASSES = len(TOKENS)
_BASES_BY_LEN = sorted(BL_CLASSES, key=len, reverse=True)


def tokenize_word(word):
    """Unicode word -> list of visual tokens (left-to-right drawing order).

    Returns None if the word uses something the model cannot write
    (for example a conjunct that is not one of the 24 dataset compounds).
    """
    w = nfc(word.strip())
    i, out = 0, []
    while i < len(w):
        base = next((b for b in _BASES_BY_LEN if w.startswith(b, i)), None)
        if base is None:
            return None
        i += len(base)
        pre, below, post = [], [], []
        while i < len(w) and ("া" <= w[i] <= "ৌ" or w[i] == AU_MARK):
            m = w[i]
            if m == O_KAR:
                pre.append(E_KAR); post.append(AA_KAR)
            elif m == OU_KAR:
                pre.append(E_KAR); post.append(AU_MARK)
            elif m in MARKS:
                {"pre": pre, "below": below, "post": post}[MARKS[m]].append(m)
            else:
                return None
            i += 1
        out += pre + [base] + below + post
    return out or None


def detokenize(tokens):
    """Visual tokens -> Unicode word (vowel signs moved to logical order)."""
    out, pending = [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        kind = MARKS.get(t)
        if kind == "pre":
            pending.append(t)
        elif kind is None:  # a base character
            out.append(t)
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if E_KAR in pending and nxt in (AA_KAR, AU_MARK):
                pending.remove(E_KAR)
                out.append(O_KAR if nxt == AA_KAR else OU_KAR)
                i += 1
            out += pending
            pending = []
        else:
            out.append(t)
        i += 1
    out += pending
    return nfc("".join(out))


def encode_word(word):
    toks = tokenize_word(word)
    return None if toks is None else [TOKEN_TO_ID[t] for t in toks]


def ctc_greedy(logits):
    """logits: (T, C) -> list of token ids (repeats and blanks removed)."""
    best = np.argmax(logits, axis=-1)
    ids, prev = [], -1
    for k in best:
        if k != prev and k != BLANK:
            ids.append(int(k))
        prev = k
    return ids


def ids_to_word(ids):
    return detokenize([TOKENS[i] for i in ids])


def lexicon_rank(logits, words, top_k=5):
    """Score every dictionary word against the model output with CTC.

    Returns [(word, probability), ...] for the best top_k words. The
    probabilities are normalised over the dictionary only.
    """
    import tensorflow as tf

    enc = [(w, encode_word(w)) for w in words]
    enc = [(w, ids) for w, ids in enc if ids and len(ids) <= logits.shape[0] // 2]
    labels = np.zeros((len(enc), max(len(ids) for _, ids in enc)), np.int32)
    for i, (_, ids) in enumerate(enc):
        labels[i, :len(ids)] = ids
    lengths = np.array([len(ids) for _, ids in enc], np.int32)
    lg = np.repeat(logits[None].astype(np.float32), len(enc), axis=0)
    loss = tf.nn.ctc_loss(labels, lg, lengths, np.full(len(enc), logits.shape[0], np.int32),
                          logits_time_major=False, blank_index=BLANK).numpy()
    score = -loss
    prob = np.exp(score - score.max())
    prob /= prob.sum()
    order = np.argsort(-score)[:top_k]
    return [(enc[i][0], float(prob[i])) for i in order]


# ----------------------------------------------------------------- images
def to_ink(gray):
    """Any grayscale image -> uint8, white ink (255) on black (0)."""
    g = gray.astype(np.uint8)
    if g.mean() > 127:
        g = 255 - g
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g


def crop_ink(ink, pad=0):
    ys, xs = np.nonzero(ink > 0)
    if len(xs) == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), ys.max() + 1 + pad
    x0, x1 = max(xs.min() - pad, 0), xs.max() + 1 + pad
    return ink[y0:y1, x0:x1]


def fit_to_canvas(ink, target_h=52, x_off=4, y_off=6):
    """Cropped ink image -> float32 (IMG_H, IMG_W, 1) model input."""
    h, w = ink.shape
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    max_w = IMG_W - x_off - 4
    new_w = min(new_w, max_w)          # very long words get squeezed
    target_h = min(target_h, IMG_H - y_off)
    img = cv2.resize(ink, (new_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_H, IMG_W), np.float32)
    canvas[y_off:y_off + target_h, x_off:x_off + new_w] = img / 255.0
    return canvas[..., None]


def preprocess_drawing(gray, stroke_fix=True):
    """Drawing from the app (any colours) -> model input, or None if empty."""
    ink = to_ink(gray)
    ink = crop_ink(ink)
    if ink is None or ink.shape[0] < 5:
        return None
    if stroke_fix:
        # keep stroke thickness close to the training images after resizing
        h = ink.shape[0]
        k = max(1, int(round(h / 52 * 1.5)))
        ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    return fit_to_canvas(ink)


def load_labels(path="labels.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
