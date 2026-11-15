# AI-Generated Face Detection Using Spatial and Frequency Domain Features

**Late-fusion detector evaluated on unseen generators** (ML subject project, B.Tech Data Science, 5th semester)
Team: Lokesh Kumar, Dhruv Khurana, Sameer Yadav | Guide: Dr. Sudhir Dawra
SRM Institute of Science and Technology, Delhi-NCR, Ghaziabad

## What this project does
It decides whether a face image is **real** or **AI-generated**. Two models look at the same image in different ways:

- **Spatial branch**: ResNet-18 (pretrained on ImageNet, fine-tuned) looks at pixels, texture and edges.
- **Frequency branch**: grayscale -> 2D FFT -> log magnitude -> per-frequency standardization -> small CNN (about 0.1 M parameters).
- **Late fusion (proposed)**: the two models are trained separately; the final fake-probability is `0.5 x spatial + 0.5 x frequency`. The weight 0.5 was fixed in advance and not tuned.
- **Joint fusion (baseline, worse)**: both branches trained together in one network.

**Main question:** does combining the two views help on generators that were never seen during training?

## Data (ArtiFact, Kaggle, faces only, 200x200 JPEG)
| Split | Real | Fake | Fake source |
|---|---|---|---|
| Train | 1,200 | 1,200 | StyleGAN2, StyleGAN1, ProjectedGAN |
| Validation (seen) | 300 | 300 | StyleGAN2, StyleGAN1, ProjectedGAN |
| Unseen test 1 | 500 | 500 | GANformer |
| Unseen test 2 | 500 | 500 | CIPS |

Real faces come from FFHQ. Faces only, so the model cannot cheat by telling scene types apart.

## Results (mean ± std over 3 seeds; accuracy % / ROC-AUC)
| Model | Seen validation | Unseen: GANformer | Unseen: CIPS |
|---|---|---|---|
| Spatial | 71.9±1.5 / 0.791 | 68.6±1.2 / 0.753 | 83.0±0.8 / 0.931 |
| Frequency | 61.4±0.8 / 0.655 | 69.7±1.9 / 0.790 | 77.7±4.3 / 0.921 |
| **Late fusion** | 72.2±2.0 / 0.792 | **71.4±0.4 / 0.804** | **84.7±1.4 / 0.958** |

Late fusion has the best accuracy and AUC on both unseen generators. Joint fusion (single runs) did worse on the unseen generator (about 61% to 62% accuracy on GANformer).

**Honest caveats:** the gain over the frequency branch on GANformer is within seed noise; only two unseen generators, both GANs; faces only; small compressed images; small training set (2,400 images); about a quarter of real faces are flagged as fake; the seen numbers are slightly optimistic because validation picks the best epoch.

## What is in this folder
| File | Purpose |
|---|---|
| `kaggle_build_dataset.py` | Run on Kaggle. Builds the dataset subset (`data.zip`, `data2.zip`). |
| `AI_Face_Detection_Main_Notebook.ipynb` | Everything else: loaders, models, training (3 seeds), results, figures, demo images. |
| `app.py` | Streamlit demo app (upload up to 4 images, Check button, 2x2 results, session history, About page). |
| `requirements.txt` | Python packages. |

This bundle contains **code only**. It does not include the data, the trained weights (`*.pt`) or the paper/report.

## How to reproduce (in order)
1. **Environment (Windows, Command Prompt)**
   ```
   py -3.12 -m venv aiimg
   aiimg\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   python -m ipykernel install --user --name=aiimg --display-name "AI Image Detection"
   ```
2. **Data**: follow the instructions at the top of `kaggle_build_dataset.py` (run Part 1 and Part 2 on Kaggle, download both zips, extract both into a folder called `data` next to the notebook).
3. **Notebook**: `jupyter notebook`, open `AI_Face_Detection_Main_Notebook.ipynb` with the "AI Image Detection" kernel and run the cells from top to bottom. The 3-seed training takes about 17 minutes on an RTX 3050 (run it only once).
4. **Outputs created in the same folder:** `spatial_s1-3.pt`, `frequency_s1-3.pt`, `seed_probs.pkl`, `cips_probs.pkl`, `results_all.csv`, `norm_stats.npz`, `fft_real_vs_fake.png`, `fig_roc.png`, `fig_confusion_unseen.png`, `fig_accuracy_all.png`, `demo_images/`.
5. **Demo app** (new Command Prompt, same folder):
   ```
   aiimg\Scripts\activate
   streamlit run app.py
   ```
   Needs `spatial_s1.pt`, `frequency_s1.pt` and `norm_stats.npz` next to `app.py`.

## Notes and troubleshooting
- **Python version:** use 3.12. Python 3.14 had no PyTorch CUDA wheels when this was built.
- **Kaggle API 429 "Too Many Requests":** downloading thousands of single files from a laptop gets rate-limited, which is why the dataset is copied inside a Kaggle notebook instead.
- **Jupyter `.py` instead of notebook:** always save with File -> Save Notebook as `.ipynb`.
- **Joint-fusion numbers in the paper** come from single development runs (an earlier pipeline version used per-image spectrum normalization). Notebook section 12 gives a fresh single run, so numbers can differ a little.
- The demo is a research prototype: roughly 70% to 85% accurate on unseen generators, so errors are normal.
