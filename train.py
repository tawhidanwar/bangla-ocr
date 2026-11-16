"""Train a Bangla handwritten WORD recogniser and log the run to MLflow.

Approach (see README for details):
1. Load the 84 isolated character classes of BanglaLekha-Isolated.
2. Build synthetic handwritten WORD images on the fly by joining real
   handwritten character samples side by side (with a shared headline).
   Vowel signs (kar) are not in the dataset, so they are synthesised from
   a Bangla font with heavy distortion and placed relative to the letter.
3. Train a CRNN (CNN + BiLSTM) with CTC loss that reads the whole word
   image and outputs the token sequence, which is rebuilt into Unicode.

Usage:
    python train.py --data-dir data/BanglaLekha-Isolated --run-name baseline
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils import (BL_CLASSES, IMG_H, IMG_W, MARKS, MAX_TOKENS, NUM_CLASSES,
                       TOKEN_TO_ID, TOKENS, crop_ink, ctc_greedy, encode_word,
                       fit_to_canvas, ids_to_word, nfc, to_ink, tokenize_word)

CONSONANTS = BL_CLASSES[11:43] + BL_CLASSES[43:46]          # ক..হ, ড় ঢ় য়
VOWELS = BL_CLASSES[:11]
DIGITS = BL_CLASSES[50:60]
COMPOUNDS = BL_CLASSES[60:]
SMALL_SIGNS = {"ং", "ঃ", "ঁ"}
KAR_CHOICES = ["া", "ি", "ী", "ু", "ূ", "ৃ",
               "ে", "ৈ", "ো", "ৌ"]
CHAR_H = 48   # stored height of every character sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/BanglaLekha-Isolated",
                   help="folder that contains the 84 class folders (searched recursively)")
    p.add_argument("--words", default="words.txt", help="dictionary of words, one per line")
    p.add_argument("--font", default="fonts/NotoSansBengali-Regular.ttf",
                   help="font used only to synthesise vowel signs (kar)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=300, help="batches per epoch")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lstm-units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--max-per-class", type=int, default=1000,
                   help="max images loaded per class (dataset has about 2000)")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="fraction of each class kept for validation words")
    p.add_argument("--lexicon-ratio", type=float, default=0.6,
                   help="share of training words taken from the dictionary (rest random)")
    p.add_argument("--val-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", default="bangla-word-recognition")
    p.add_argument("--run-name", default=None)
    p.add_argument("--tracking-uri", default=os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow/mlflow.db"))
    p.add_argument("--model-out", default="models/model.keras")
    p.add_argument("--labels-out", default="labels.json")
    return p.parse_args()


# ------------------------------------------------------------------ data
def find_class_root(data_dir):
    for root, dirs, _ in os.walk(data_dir):
        if all(str(i) in dirs for i in (1, 2, 50, 84)):
            return root
    raise FileNotFoundError(
        f"Could not find the 84 class folders (1..84) under '{data_dir}'. "
        "Download BanglaLekha-Isolated and extract it there.")


def load_char(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    ink = crop_ink(to_ink(g))
    if ink is None or ink.shape[0] < 8 or ink.shape[1] < 3 or ink.mean() < 3:
        return None   # blank or broken sample
    h, w = ink.shape
    new_w = int(np.clip(round(w * CHAR_H / h), 4, 2 * CHAR_H))
    return cv2.resize(ink, (new_w, CHAR_H), interpolation=cv2.INTER_AREA)


def load_banks(data_dir, max_per_class, val_frac, rng):
    root = find_class_root(data_dir)
    print(f"Loading characters from {root}")
    train_bank, val_bank = {}, {}
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for idx, char in enumerate(BL_CLASSES, start=1):
            folder = os.path.join(root, str(idx))
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
            rng.shuffle(files)
            files = files[:max_per_class]
            imgs = [im for im in pool.map(load_char, [os.path.join(folder, f) for f in files])
                    if im is not None]
            n_val = max(1, int(len(imgs) * val_frac))
            val_bank[char], train_bank[char] = imgs[:n_val], imgs[n_val:]
            if idx % 12 == 0 or idx == 84:
                print(f"  classes loaded: {idx}/84")
    return train_bank, val_bank


def build_mark_bank(font_path, rng, n_variants=40):
    """Render each vowel sign next to a dotted circle so we learn where it sits
    relative to a letter, then keep many distorted copies of it."""
    font = ImageFont.truetype(font_path, 120)

    def render(text):
        im = Image.new("L", (420, 300), 0)
        ImageDraw.Draw(im).text((150, 60), text, fill=255, font=font)
        return np.array(im)

    circle = render("◌")
    ys, xs = np.nonzero(circle)
    cx0, cx1 = xs.min(), xs.max()
    # a reference letter (ক) gives the real letter height, headline to baseline
    ys_k, _ = np.nonzero(render("ক"))
    ky0, kh = ys_k.min(), ys_k.max() - ys_k.min()
    circle_big = cv2.dilate(circle, np.ones((9, 9), np.uint8))
    bank = {}
    for mark in MARKS:
        combo = render("◌" + mark)
        m = np.where(circle_big > 0, 0, combo).astype(np.uint8)
        ys, xs = np.nonzero(m > 128)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = m[y0:y1, x0:x1]
        # positions and sizes in units of the letter height
        geom = dict(left=(x0 - cx0) / kh, right=(x0 - cx1) / kh,
                    center=(x0 - (cx0 + cx1) / 2) / kh, top=(y0 - ky0) / kh,
                    w=(x1 - x0) / kh, h=(y1 - y0) / kh)
        variants = []
        for _ in range(n_variants):
            variants.append(distort(crop, rng, strong=True))
        bank[mark] = (variants, geom)
    return bank


def distort(ink, rng, strong=False):
    """Small random affine warp plus stroke thickness change."""
    h, w = ink.shape
    pad = max(h, w) // 3 + 4
    img = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = img.shape
    ang = rng.uniform(-10, 10) if strong else rng.uniform(-4, 4)
    sh = rng.uniform(-0.25, 0.25) if strong else rng.uniform(-0.1, 0.1)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    M[0, 1] += sh
    M[0, 2] -= sh * H / 2
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    if strong:  # wobble the font shape so it looks less printed
        dx = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), 8) * 12
        dy = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), 8) * 12
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        img = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR)
        k = int(rng.integers(0, 3))
        if k:
            img = cv2.erode(img, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))
    img = (img > 100).astype(np.uint8) * 255
    out = crop_ink(img)
    return out if out is not None else ink


class WordSynth:
    def __init__(self, char_bank, mark_bank, lexicon_ids, lexicon_ratio, rng):
        self.chars, self.marks = char_bank, mark_bank
        self.lexicon_ids, self.lexicon_ratio, self.rng = lexicon_ids, lexicon_ratio, rng

    def random_tokens(self):
        rng = self.rng
        if rng.random() < 0.07:
            word = "".join(rng.choice(DIGITS, size=int(rng.integers(1, 5))))
        else:
            word = ""
            for i in range(int(rng.integers(1, 5))):
                r = rng.random()
                if r < 0.68:
                    base = rng.choice(CONSONANTS)
                elif r < 0.83:
                    base = rng.choice(COMPOUNDS)
                else:
                    base = rng.choice(VOWELS)
                word += base
                if base not in VOWELS and rng.random() < 0.55:
                    word += rng.choice(KAR_CHOICES)
            if rng.random() < 0.08:
                word += rng.choice(["ং", "ঃ", "ৎ"])
        toks = tokenize_word(word)
        if toks is None or len(toks) > MAX_TOKENS:
            return self.random_tokens()
        return [TOKEN_TO_ID[t] for t in toks]

    def sample_ids(self):
        if self.lexicon_ids and self.rng.random() < self.lexicon_ratio:
            return self.lexicon_ids[int(self.rng.integers(len(self.lexicon_ids)))]
        return self.random_tokens()

    def _place_mark(self, canvas, mark, box, scale):
        variants, g = self.marks[mark]
        img = variants[int(self.rng.integers(len(variants)))]
        bx, by, bw, bh = box
        s = bh * scale
        mw, mh = max(2, int(g["w"] * s)), max(2, int(g["h"] * s))
        kind = MARKS[mark]
        if kind == "pre":
            mx = bx + g["left"] * s
        elif kind == "post":
            mx = bx + bw + g["right"] * s
        else:
            mx = bx + bw * self.rng.uniform(0.45, 0.65) + g["center"] * s
        my = by + g["top"] * s + self.rng.uniform(-2, 2)
        return img, int(mx), int(my), mw, mh

    @staticmethod
    def _paste(canvas, img, x, y, w=None, h=None):
        if w is not None:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        h, w = img.shape
        x, y = max(x, 0), max(y, 0)
        region = canvas[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = np.maximum(region, img[:region.shape[0], :region.shape[1]])

    def render(self, ids):
        rng = self.rng
        tokens = [TOKENS[i] for i in ids]
        canvas = np.zeros((220, 90 * len(tokens) + 200), np.uint8)
        word_h = int(rng.integers(40, 53))
        top = 70
        x, last_box, pending = 40, None, []
        scale = rng.uniform(0.85, 1.15)
        lefts, rights, tops = [], [], []
        for t in tokens:
            kind = MARKS.get(t)
            if kind == "pre":
                pending.append(t)
                continue
            if kind is None:
                pool = self.chars[t]
                ch = pool[int(rng.integers(len(pool)))]
                h = word_h * rng.uniform(0.9, 1.05)
                if t in SMALL_SIGNS:
                    h *= 0.6
                h = int(h)
                w = max(3, int(ch.shape[1] * h / ch.shape[0] * rng.uniform(0.85, 1.15)))
                by = top + int(rng.integers(-2, 3)) + (int(word_h * 0.3) if t in {"ং", "ঃ"} else 0)
                placed = [self._place_mark(canvas, m, (x, by, w, h), scale) for m in pending]
                shift = max([0] + [x - p[1] for p in placed])
                if placed:
                    shift += int(rng.integers(0, 4))
                bx = x + shift
                # ে and ৈ sit fully left of the letter; ি is allowed to overlap
                for p, m in zip(placed, pending):
                    if m != "ি":
                        bx = max(bx, p[1] + shift + int(p[3] * rng.uniform(0.7, 1.0)))
                self._paste(canvas, ch, bx, by, w, h)
                ends = [bx + w]
                for img, mx, my, mw, mh in placed:
                    self._paste(canvas, img, mx + shift, my, mw, mh)
                    ends.append(mx + shift + mw)
                last_box = (bx, by, w, h)
                lefts.append(bx); tops.append(by)
                pending = []
                x = max(ends) + int(rng.integers(-4, 7) * h / 48)
            else:
                box = last_box or (x, top, 30, word_h)
                img, mx, my, mw, mh = self._place_mark(canvas, t, box, scale)
                self._paste(canvas, img, mx, my, mw, mh)
                if kind == "post":
                    x = max(x, mx + mw + int(rng.integers(-2, 6)))
            rights.append(x)
        # many writers draw one continuous headline (matra) over the word
        if lefts and rng.random() < 0.6:
            y = int(np.median(tops)) + int(rng.integers(0, 4))
            cv2.line(canvas, (min(lefts), y), (max(rights) - 2, y + int(rng.integers(-2, 3))),
                     255, int(rng.integers(2, 5)))
        ink = crop_ink(canvas)
        ink = distort(ink, rng)
        k = int(rng.integers(-1, 3))
        if k > 0:
            ink = cv2.dilate(ink, np.ones((k + 1, k + 1), np.uint8))
        elif k < 0:
            ink = cv2.erode(ink, np.ones((2, 2), np.uint8))
        ink = crop_ink(ink) if crop_ink(ink) is not None else canvas
        th = int(rng.integers(44, 59))
        x_off = int(rng.integers(2, 12))
        y_off = int(rng.integers(2, max(3, IMG_H - th)))
        img = fit_to_canvas(ink, target_h=th, x_off=x_off, y_off=y_off)
        if rng.random() < 0.3:
            img = np.clip(img + rng.normal(0, 0.05, img.shape), 0, 1).astype(np.float32)
        return img

    def batch(self, n):
        x = np.zeros((n, IMG_H, IMG_W, 1), np.float32)
        y = np.zeros((n, MAX_TOKENS), np.int32)
        for i in range(n):
            ids = self.sample_ids()
            x[i] = self.render(ids)
            y[i, :len(ids)] = ids
        return x, y


# ----------------------------------------------------------------- model
def build_model(lstm_units, dropout):
    import keras
    from keras import layers

    inp = keras.Input((IMG_H, IMG_W, 1), name="image")
    x = inp
    for filters, pool in [(32, (2, 2)), (64, (2, 2)), (128, None), (128, (2, 1)),
                          (256, None), (256, (2, 1))]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if pool:
            x = layers.MaxPooling2D(pool)(x)
    # (4, 64, 256) -> 64 time steps along the width
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((IMG_W // 4, (IMG_H // 16) * 256))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout))(x)
    out = layers.Dense(NUM_CLASSES, name="logits")(x)
    return keras.Model(inp, out, name="bangla_crnn")


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def save_samples(x, y, font_path, path, n=12):
    font = ImageFont.truetype(font_path, 22)
    sheet = Image.new("L", (IMG_W * 2 + 20, n * (IMG_H + 6)), 255)
    draw = ImageDraw.Draw(sheet)
    for i in range(min(n, len(x))):
        im = Image.fromarray((255 - x[i, :, :, 0] * 255).astype(np.uint8))
        sheet.paste(im, (0, i * (IMG_H + 6)))
        ids = [int(t) for t in y[i] if t > 0]
        draw.text((IMG_W + 15, i * (IMG_H + 6) + 15), ids_to_word(ids), fill=0, font=font)
    sheet.save(path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    import mlflow
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    words = []
    with open(args.words, encoding="utf-8") as f:
        for line in f:
            w = nfc(line.strip())
            ids = encode_word(w) if w else None
            if ids and len(ids) <= MAX_TOKENS and w not in words:
                words.append(w)
    lexicon_ids = [encode_word(w) for w in words]
    print(f"Dictionary: {len(words)} usable words")

    train_bank, val_bank = load_banks(args.data_dir, args.max_per_class, args.val_frac, rng)
    mark_bank = build_mark_bank(args.font, rng)
    train_gen = WordSynth(train_bank, mark_bank, lexicon_ids, args.lexicon_ratio, rng)
    val_gen = WordSynth(val_bank, mark_bank, lexicon_ids, 0.5, np.random.default_rng(args.seed + 1))
    x_val, y_val = val_gen.batch(args.val_size)
    val_words = [ids_to_word([int(t) for t in row if t > 0]) for row in y_val]

    labels = {
        "tokens": TOKENS, "blank_index": 0, "img_h": IMG_H, "img_w": IMG_W,
        "max_tokens": MAX_TOKENS,
        "banglalekha_classes": {str(i + 1): c for i, c in enumerate(BL_CLASSES)},
        "vowel_signs": MARKS, "lexicon": words,
    }
    with open(args.labels_out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)

    model = build_model(args.lstm_units, args.dropout)
    optimizer = tf.keras.optimizers.Adam(args.lr, clipnorm=5.0)

    def ctc_loss(y, logits):
        y = tf.cast(y, tf.int32)
        label_len = tf.reduce_sum(tf.cast(y > 0, tf.int32), axis=1)
        logit_len = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        return tf.reduce_mean(tf.nn.ctc_loss(y, logits, label_len, logit_len,
                                             logits_time_major=False, blank_index=0))

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = ctc_loss(y, model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_logits(x):
        return model(x, training=False)

    def gen():
        while True:
            yield train_gen.batch(args.batch_size)

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((None, IMG_H, IMG_W, 1), tf.float32),
        tf.TensorSpec((None, MAX_TOKENS), tf.int32))).prefetch(4)
    it = iter(ds)

    def evaluate():
        losses, preds = [], []
        for i in range(0, len(x_val), 128):
            lg = eval_logits(x_val[i:i + 128])
            losses.append(float(ctc_loss(y_val[i:i + 128], lg)) * len(lg))
            preds += [ids_to_word(ctc_greedy(l)) for l in lg.numpy()]
        acc = np.mean([p == t for p, t in zip(preds, val_words)])
        cer = sum(edit_distance(p, t) for p, t in zip(preds, val_words)) / sum(len(t) for t in val_words)
        return sum(losses) / len(x_val), acc, cer

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    os.makedirs("artifacts/mlflow", exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=os.path.abspath(
            "artifacts/mlflow/mlartifacts"))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        params = {k: v for k, v in vars(args).items() if k not in ("tracking_uri",)}
        params.update(num_classes=NUM_CLASSES, dictionary_words=len(words),
                      train_char_images=sum(len(v) for v in train_bank.values()),
                      val_char_images=sum(len(v) for v in val_bank.values()),
                      model_params=model.count_params())
        mlflow.log_params(params)
        mlflow.set_tag("approach", "synthetic words from BanglaLekha chars + CRNN-CTC")

        xs, ys = train_gen.batch(16)
        save_samples(xs, ys, args.font, "artifacts/training_samples.png")
        mlflow.log_artifact("artifacts/training_samples.png")

        best, bad_epochs, lr = float("inf"), 0, args.lr
        for epoch in range(1, args.epochs + 1):
            t0, total = time.time(), 0.0
            for step in range(args.steps):
                x, y = next(it)
                total += float(train_step(x, y))
            train_loss = total / args.steps
            val_loss, val_acc, val_cer = evaluate()
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_word_acc": val_acc, "val_cer": val_cer, "lr": lr}, step=epoch)
            print(f"epoch {epoch:2d}  train_loss {train_loss:.3f}  val_loss {val_loss:.3f}  "
                  f"val_word_acc {val_acc:.3f}  val_cer {val_cer:.3f}  ({time.time() - t0:.0f}s)")
            if val_loss < best:
                best, bad_epochs = val_loss, 0
                model.save(args.model_out)
            elif epoch > 5:  # CTC validation loss is noisy at the start
                bad_epochs += 1
                if bad_epochs >= 3:
                    lr *= 0.5
                    optimizer.learning_rate.assign(lr)
                    bad_epochs = 0
                    print(f"  learning rate -> {lr:.2e}")

        # final evaluation with the best model, including dictionary decoding
        import keras
        best_model = keras.models.load_model(args.model_out)
        from ocr_utils import lexicon_rank
        lex_set = set(words)
        idx = [i for i, w in enumerate(val_words) if w in lex_set][:200]
        logits = best_model.predict(x_val[idx], verbose=0)
        hits = sum(lexicon_rank(l, words)[0][0] == val_words[i] for l, i in zip(logits, idx))
        lex_acc = hits / max(1, len(idx))
        mlflow.log_metrics({"best_val_loss": best, "val_dictionary_acc": lex_acc})
        print(f"best val_loss {best:.3f}   dictionary-decoding accuracy {lex_acc:.3f}")

        mlflow.log_artifact(args.labels_out)
        mlflow.log_artifact(args.words)
        mlflow.log_artifact(args.model_out, artifact_path="model")
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
