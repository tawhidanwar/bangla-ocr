"""Create the source-code-only zip for submission.

Leaves out the dataset, trained model, MLflow runs, caches and environments.
Usage: python make_zip.py
"""

import os
import zipfile

KEEP = ["train.py", "app.py", "ocr_utils.py", "make_zip.py", "requirements.txt", "Dockerfile",
        ".dockerignore", ".gitignore", "README.md", "labels.json", "words.txt",
        "github_link.txt", "fonts/NotoSansBengali-Regular.ttf", "fonts/OFL.txt"]

out = "bangla-ocr-assignment-source.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for path in KEEP:
        if os.path.exists(path):
            z.write(path, os.path.join("bangla-ocr-assignment", path))
        else:
            print("missing (skipped):", path)
print("wrote", out, f"({os.path.getsize(out) / 1024:.0f} KB)")
