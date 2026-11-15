# ============================================================================
# RUN THIS ON KAGGLE (not on your laptop).
#   1. kaggle.com -> Create -> New Notebook
#   2. Right panel -> Add Input -> Datasets -> "ArtiFact: Real and Fake Image Dataset" (by Awsaf)
#   3. Paste PART 1 into a cell and run it. It builds /kaggle/working/data.zip
#   4. Paste PART 2 into a new cell and run it. It builds /kaggle/working/data2.zip
#   5. Download both zips (right panel -> Output -> /kaggle/working -> three dots -> Download)
#   6. On your laptop, extract BOTH zips into a folder named  data  next to the notebook:
#        data.zip  -> data\trainval and data\test_unseen
#        data2.zip -> data\test_unseen2
# (The Kaggle API on a laptop gets rate-limited, so we copy inside Kaggle instead.)
# ============================================================================

# %% ---------------------------- PART 1: train/val + GANformer ----------------------------
import os, shutil, pandas as pd
from tqdm import tqdm

ROOT = "/kaggle/input/datasets/awsaf49/artifact-dataset"
OUT = "/kaggle/working/data"
shutil.rmtree(OUT, ignore_errors=True)

def sample(src, cat, n):
    df = pd.read_csv(f"{ROOT}/{src}/metadata.csv")
    if cat: df = df[df["category"] == cat]
    return df.sample(n=n, random_state=42)["image_path"].tolist()

jobs = []
def add(src, paths, split, label):
    for i, p in enumerate(paths):
        jobs.append((f"{ROOT}/{src}/{p}", f"{OUT}/{split}/{label}/{src}_{i:05d}.jpg"))

real = sample("ffhq", None, 2000)
add("ffhq", real[:1500], "trainval", "real")          # 1500 real faces for training/validation
add("ffhq", real[1500:], "test_unseen", "real")       # 500 other real faces for the GANformer test
add("stylegan2", sample("stylegan2", "ffhq", 500), "trainval", "fake")
add("stylegan1", sample("stylegan1", None, 500), "trainval", "fake")
add("projected_gan", sample("projected_gan", "ffhq", 500), "trainval", "fake")
add("gansformer", sample("gansformer", "ffhq_images", 500), "test_unseen", "fake")   # unseen generator 1

missing = 0
for src, dst in tqdm(jobs):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src): shutil.copy(src, dst)
    else: missing += 1

print("Missing source files:", missing)
for d in sorted([f"{OUT}/{s}/{l}" for s in ("trainval", "test_unseen") for l in ("real", "fake")]):
    print(d, len(os.listdir(d)))          # expected 1500, 1500, 500, 500
shutil.make_archive("/kaggle/working/data", "zip", OUT)
print("zip size (MB):", round(os.path.getsize("/kaggle/working/data.zip") / 1e6, 1))

# %% ---------------------------- PART 2: second unseen generator (CIPS) ----------------------------
import os, shutil, pandas as pd
from tqdm import tqdm

ROOT = "/kaggle/input/datasets/awsaf49/artifact-dataset"
OUT = "/kaggle/working/data2"
shutil.rmtree(OUT, ignore_errors=True)

def meta(src, cat=None):
    df = pd.read_csv(f"{ROOT}/{src}/metadata.csv")
    return df[df["category"] == cat] if cat else df

used = set(meta("ffhq").sample(n=2000, random_state=42)["image_path"])      # same sample as Part 1
pool = meta("ffhq"); pool = pool[~pool["image_path"].isin(used)]            # real faces not used before
real_p = pool.sample(n=500, random_state=7)["image_path"].tolist()
fake_p = meta("cips", "cips-ffhq").sample(n=500, random_state=42)["image_path"].tolist()

jobs = [(f"{ROOT}/ffhq/{p}", f"{OUT}/test_unseen2/real/ffhq_{i:05d}.jpg") for i, p in enumerate(real_p)]
jobs += [(f"{ROOT}/cips/{p}", f"{OUT}/test_unseen2/fake/cips_{i:05d}.jpg") for i, p in enumerate(fake_p)]

missing = 0
for src, dst in tqdm(jobs):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src): shutil.copy(src, dst)
    else: missing += 1

print("Missing:", missing)
for l in ("real", "fake"):
    print(l, len(os.listdir(f"{OUT}/test_unseen2/{l}")))     # expected 500, 500
shutil.make_archive("/kaggle/working/data2", "zip", OUT)
print("zip MB:", round(os.path.getsize("/kaggle/working/data2.zip") / 1e6, 1))
