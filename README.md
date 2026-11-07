# Image-classification-on-flower-dataset

Implementation of CNN on flower dataset in python using keras

Refer following link to download dataset.
https://www.kaggle.com/alxmamaev/flowers-recognition
http://blog.yhat.com/posts/currency-portfolio-optimization-using-scienceops.html

The Kaggle page is the flower-image source. The yhat link is the second link from the original README and is not read by the code.

## Setup

The checked Python is 3.12. Create a virtual environment and install the training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

`requirements.txt` lists numpy, pandas, Pillow, scikit-learn, matplotlib, and tensorflow. `requirements-dev.txt` lists pytest and ruff.

This tree was checked with numpy 2.5.4, pandas 2.3.3, Pillow 11.3.0, scikit-learn 1.9.1, matplotlib 3.11.2, pytest 8.4.2, ruff 0.17.4, tensorflow 2.21.0, and Keras 3.15.1.

## Data the script actually reads

Flower JPEGs are not in this repository. `python flower_data.py` still looks in the original location:

```text
D:/Image_data/flowers/daisy/*.jpg
D:/Image_data/flowers/dandelion/*.jpg
D:/Image_data/flowers/rose/*.jpg
D:/Image_data/flowers/sunflower/*.jpg
D:/Image_data/flowers/tulip/*.jpg
```

Download [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) and keep those five folder names. The Kaggle description says chamomile; the folder name used by this script and by the dataset is `daisy`. Only lowercase `.jpg` names are read. Kaggle lists the dataset license as Unknown. The images were scraped from Flickr, Google Images, and Yandex, so this repo does not redistribute them.

Override the location without editing the file:

```bash
python flower_data.py --data-dir /path/to/flowers
FLOWER_DATA_DIR=/path/to/flowers python flower_data.py
```

Labels stay daisy 0, dandelion 1, rose 2, sunflower 3, tulip 4. Each image is resized to 100×100 with Pillow's default resample filter (bicubic for RGB and grayscale) and flattened. Rows are shuffled with no seed, then split with `test_size=0.20` and `random_state=42`, then divided by 255.

## Train

```bash
python flower_data.py
```

Importing `flower_data` does not train. The conv and dense stack is the original one: three conv/pool blocks (64, 128, and 150 filters), dense layers of 500 and 600, a 5-way softmax, categorical crossentropy, Adam, batch size 16, 30 epochs, and TensorFlow seed 2. Current Keras wants the `(32, 32, 3)` input declared as its own `Input` layer, so the summary shows that layer before the first convolution.

On the real 100×100 RGB vectors this command stops before `fit`. Those vectors have length 30000. The network reshape is still `(32, 32, 3)`, which is 3072 values. That mismatch is the original script, and this cleanup does not pick a replacement size because that would change the network. The command now raises `ValueError` with those two sizes before it imports Keras.

If a class folder has no `.jpg` files, the command raises `FileNotFoundError` naming the missing classes.

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs those three commands on Python 3.12. It installs `requirements.txt` except tensorflow. The default 30-epoch command and the Kaggle download are outside CI.

One test fits a single epoch on synthetic vectors that already have length 3072, and only when tensorflow is installed. That covers the model body. It does not cover a successful run on the flower JPEGs, because those vectors still do not match the CNN input.

## Large files in the repository

These files are Windows software, not flower photos. `flower_data.py` does not open them. They are still in git. Nothing here deletes them, moves them, or points the code at a Google Drive folder.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `7z1900-x64.exe` | 1,447,178 | `0f5d4dbbe5e55b7aa31b91e5925ed901fdf46a367491d81381846f05ad54c45e` |
| `Tesseract-OCR.7z.001` | 10,485,760 | `18df81b93a42ab542ad88af1451b6a4d243f60d2ce4921e80cd945fc2cffd8f8` |
| `Tesseract-OCR.7z.002` | 10,485,760 | `ff75f248a5f8b86727ed67463a8147bf2662e5e31d6bf03ade4c3a4bd142c87d` |
| `Tesseract-OCR.7z.003` | 10,485,760 | `ed6cd64dd0fc76fda134ab95e39c7746b3ed8c88907cf957f278a961dacf3b4f` |
| `Tesseract-OCR.7z.004` | 10,485,760 | `232426eeff5344c3e79886cd7da3c1999aa47813b4ee4047c3db7dc4a7060f71` |
| `Tesseract-OCR.7z.005` | 4,745,476 | `a444fe2f500015a4a21271a7328b3d8b9f30077c264827bcf29155525ff2ec87` |

The five Tesseract volumes total 46,688,516 bytes. Concatenated in name order, their SHA-256 is `a8c3552f0245c90f3a0727d3100bc6ed3886dfb7d7cd35d9f472a216ea2d005a`.

`7z1900-x64.exe` is the official 7-Zip 19.00 64-bit Windows installer. Its SHA-256 is the publisher hash for <https://7-zip.org/a/7z1900-x64.exe>. 7-Zip is GNU LGPL software with the upstream unRAR restriction. The executable's version resource is 19.00, copyright 1999–2018 Igor Pavlov.

`Tesseract-OCR.7z.001` is the first volume of a split 7z archive (7-Zip 23 lists 5 volumes, uncompressed 241,352,329 bytes, 123 files and 6 folders). The payload is a Windows portable tree `Tesseract-OCR/` built as `tesseract v5.0.0-alpha.20200328` (`libtesseract-5.dll`, x86_64 mingw), including `tesseract.exe`, `eng.traineddata`, and `osd.traineddata`. `Tesseract-OCR/doc/LICENSE` is the Apache License 2.0. The same tree also vendors other libraries (Leptonica, ICU, and others) under their own licenses. There are no flower images inside.

The `tesseract.exe` deleted in commit `dfc968b` (975,648 bytes, git blob `54fbd56ebf515703b8d08e379259da342e7dac2a`) is byte-identical to `Tesseract-OCR/tesseract.exe` in that archive. SHA-256 `e81768a8ec8130d02b4e8c1815da305fe420d4d780f86d8a5c1b516e95d06c4b`.

## Later backup, not part of this change

Keep these originals until a backup has been checked against the SHA-256 values above.

- Flower images, once a Drive folder exists: point `--data-dir` or `FLOWER_DATA_DIR` at the synced folder. Add the Drive URL only after that folder exists. A fresh copy can also be downloaded from the Kaggle dataset; its license is Unknown, so do not assume the images can be republished.
- 7-Zip: download <https://7-zip.org/a/7z1900-x64.exe> again and check the SHA-256. The classifier does not use 7-Zip, so the code should not grow a Drive link for it. A Drive copy is only a personal backup of the installer.
- Tesseract: install a current build from <https://github.com/tesseract-ocr/tesseract/releases> (Apache-2.0) if OCR is ever needed. The checked-in archive is an unused 2020 alpha. A Drive copy of the five volumes is optional backup; verify each SHA-256 before any later commit removes them from git.
