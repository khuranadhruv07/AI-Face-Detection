import io, os, time, base64
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision, streamlit as st
import matplotlib
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="AI Face Detector", layout="wide", initial_sidebar_state="expanded")
dev = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMGS = 4


# ---------------- models (same as the ones evaluated in the project) ----------------
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class FreqNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(s, x):
        return s.f(x)


class Spatial(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = torchvision.models.resnet18(weights=None)
        s.net.fc = nn.Linear(512, 2)

    def forward(s, im):
        return s.net(im)


class Freq(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = FreqNet()
        s.fc = nn.Linear(128, 2)

    def forward(s, sp):
        return s.fc(s.f(sp))


@st.cache_resource
def load():
    a = Spatial()
    a.load_state_dict(torch.load("spatial_s1.pt", map_location=dev))
    a.to(dev).eval()
    b = Freq()
    b.load_state_dict(torch.load("frequency_s1.pt", map_location=dev))
    b.to(dev).eval()
    stats = np.load("norm_stats.npz")
    return a, b, stats["MU"], stats["SD"]


spatial, freq, MU, SD = load()
norm = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
VIRIDIS = matplotlib.colormaps["viridis"]


def predict(img):
    im = img.convert("RGB").resize((200, 200))
    g = np.array(im.convert("L"), dtype=np.float32)
    raw = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    sp = torch.from_numpy((raw - MU) / SD).unsqueeze(0).unsqueeze(0).float().to(dev)
    x = norm(im).unsqueeze(0).to(dev)
    with torch.no_grad():
        ps = torch.softmax(spatial(x), 1)[0, 1].item()
        pf = torch.softmax(freq(sp), 1)[0, 1].item()
    r = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    fft_img = (VIRIDIS(r)[:, :, :3] * 255).astype(np.uint8)
    return ps, pf, 0.5 * ps + 0.5 * pf, fft_img


def thumb_uri(img):
    t = img.convert("RGB").resize((64, 64))
    buf = io.BytesIO()
    t.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def verdict(p):
    fake = p > 0.5
    return fake, (p if fake else 1 - p)


def warnings_for(img):
    w, h = img.size
    out = []
    if min(w, h) < 100:
        out.append("Very small image: the model was trained on 200x200 faces.")
    if max(w, h) / max(1, min(w, h)) > 1.4:
        out.append("Not square: it will be squeezed to 200x200.")
    return out


# ---------------- session state (lives only while this browser session / app run lasts) ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = []

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("Controls")
    view = st.radio("Compare view",
                    ["Late fusion (final)", "Spatial only", "Frequency only", "Show all three"],
                    help="Choose which model decides the label on each card.")
    show_fft = st.checkbox("Show FFT spectrum", value=True)
    show_bars = st.checkbox("Show branch probability bars", value=True)
    st.divider()
    if st.button("Clear session history", key="clear_side"):
        st.session_state.history = []
        st.session_state.last = []
        st.rerun()
    st.caption("History is kept only for this running session. It is cleared when you refresh the page "
               "or restart the app.")
    st.caption("Research prototype. Expected accuracy on unseen generators is roughly 70% to 85%, "
               "so errors are normal.")

st.title("AI-Generated Face Detector")
st.caption("Late fusion of a spatial branch (ResNet-18) and a frequency branch (FFT + small CNN)")

tab_det, tab_hist, tab_about = st.tabs(["Detector", "Session history", "About the project"])

# ---------------- detector tab ----------------
with tab_det:
    source = st.radio("Image source", ["Upload my images", "Pick demo images"], horizontal=True)
    imgs = []
    if source == "Upload my images":
        files = st.file_uploader(f"Upload up to {MAX_IMGS} face images (JPG or PNG)",
                                 type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if len(files) > MAX_IMGS:
            st.warning(f"Only the first {MAX_IMGS} images will be checked.")
        for f in files[:MAX_IMGS]:
            imgs.append((f.name, Image.open(io.BytesIO(f.getvalue()))))
    else:
        if os.path.isdir("demo_images"):
            names = sorted(os.listdir("demo_images"))
            pick = st.multiselect(f"Choose up to {MAX_IMGS} demo images", names)
            if len(pick) > MAX_IMGS:
                st.warning(f"Only the first {MAX_IMGS} images will be checked.")
            for n in pick[:MAX_IMGS]:
                imgs.append((n, Image.open(os.path.join("demo_images", n))))
        else:
            st.info("No demo_images folder found next to app.py.")

    if imgs:
        pc = st.columns(len(imgs))
        for c, (n, im) in zip(pc, imgs):
            c.image(im, caption=n, width=110)

    if st.button("Check", type="primary", disabled=(len(imgs) == 0)):
        results = []
        with st.spinner("Checking..."):
            for name, im in imgs:
                ps, pf, p, fft_img = predict(im)
                results.append(dict(name=name, img=im.convert("RGB"), ps=ps, pf=pf, p=p,
                                    fft=fft_img, warn=warnings_for(im)))
        st.session_state.last = results
        t = time.strftime("%H:%M:%S")
        for r in results:
            st.session_state.history.append({
                "Image": thumb_uri(r["img"]), "Time": t, "File": r["name"],
                "Verdict": "Fake" if r["p"] > 0.5 else "Real",
                "Late fusion % fake": round(r["p"] * 100, 1),
                "Spatial % fake": round(r["ps"] * 100, 1),
                "Frequency % fake": round(r["pf"] * 100, 1)})

    res = st.session_state.last
    if res:
        key = {"Late fusion (final)": "p", "Spatial only": "ps",
               "Frequency only": "pf", "Show all three": "p"}[view]
        fakes = [verdict(r[key])[0] for r in res]
        confs = [verdict(r[key])[1] for r in res]
        st.subheader(f"Results: {sum(fakes)} fake, {len(res) - sum(fakes)} real "
                     f"(average confidence {np.mean(confs) * 100:.1f}%)")
        st.caption(f"View: {view}. These are the results of the last Check click.")
        for i in range(0, len(res), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(res):
                    continue
                r = res[i + j]
                with cols[j]:
                    with st.container(border=True):
                        st.markdown(f"**{r['name']}**")
                        if show_fft:
                            a, b = st.columns(2)
                            a.image(r["img"], use_container_width=True)
                            b.image(r["fft"], caption="FFT spectrum", use_container_width=True)
                        else:
                            st.image(r["img"], use_container_width=True)
                        fake, conf = verdict(r[key])
                        if fake:
                            st.error(f"AI-GENERATED (Fake) | confidence {conf * 100:.1f}%")
                        else:
                            st.success(f"REAL | confidence {conf * 100:.1f}%")
                        if view == "Show all three":
                            for nm, k in [("Late fusion", "p"), ("Spatial branch", "ps"), ("Frequency branch", "pf")]:
                                f2, _ = verdict(r[k])
                                st.write(f"**{nm}:** {'Fake' if f2 else 'Real'} ({r[k] * 100:.1f}% fake)")
                        if show_bars:
                            for nm, k in [("Late fusion", "p"), ("Spatial", "ps"), ("Frequency", "pf")]:
                                st.progress(float(r[k]), text=f"{nm}: {r[k] * 100:.1f}% fake")
                        for w in r["warn"]:
                            st.caption(w)
        with st.expander("How to read this"):
            st.markdown(
                "- **Confidence** is how sure the chosen model is about its label.\n"
                "- **Spatial** looks at the image itself (texture, edges). **Frequency** looks at the "
                "Fourier spectrum. **Late fusion** is the average of the two, and it is the final answer.\n"
                "- The model was trained on 200x200 face images, so photos that are not faces can give "
                "meaningless scores.\n"
                "- The score is a probability, not proof.")

# ---------------- history tab ----------------
with tab_hist:
    st.subheader("Checks in this session")
    hist = st.session_state.history
    if hist:
        df = pd.DataFrame(hist)
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"Image": st.column_config.ImageColumn("Image", width="small")})
        st.caption("History always stores the late-fusion verdict plus both branch scores.")
        st.download_button("Download history as CSV", df.drop(columns=["Image"]).to_csv(index=False),
                           file_name="session_history.csv", mime="text/csv")
        if st.button("Clear history", key="clear_tab"):
            st.session_state.history = []
            st.session_state.last = []
            st.rerun()
    else:
        st.info("Nothing checked yet in this session.")


# ---------------- about tab ----------------
def show_fig(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


with tab_about:
    st.header("AI-Generated Face Detection Using Spatial and Frequency Domain Features")
    st.caption("ML subject project | B.Tech Data Science | SRM Institute of Science and Technology, Delhi-NCR, Ghaziabad")

    with st.expander("1. Overview", expanded=True):
        st.markdown(
            "This project decides whether a face image is **real** or **AI-generated**. It combines two "
            "views of an image: the normal picture (spatial view) and its Fourier spectrum (frequency view). "
            "The two models are trained separately and their probabilities are averaged (late fusion).\n\n"
            "The main question: does combining the two views help on AI generators that the model never saw "
            "during training?")

    with st.expander("2. Problem and motivation"):
        st.markdown(
            "**Why it matters.** GANs and diffusion models can make faces that people cannot tell from "
            "photographs. This enables misinformation, fake evidence and identity misuse.\n\n"
            "**The hard part.** A detector trained on one generator often fails on a new one. Testing on "
            "unseen generators is therefore more important than testing on the training generators.")

    with st.expander("3. How it works"):
        st.markdown(
            "**3.1 Spatial branch.** A ResNet-18 pretrained on ImageNet, fine-tuned on our faces. It looks at "
            "pixels, texture and edges.\n\n"
            "**3.2 Frequency branch.** The image is turned to grayscale, a 2D FFT is applied, the log of the "
            "magnitude is taken and each frequency is standardized. A small CNN (about 0.1 million parameters) "
            "reads this spectrum. Generators leave faint repeating traces there.\n\n"
            "**3.3 Late fusion.** Final score = 0.5 x spatial probability + 0.5 x frequency probability. "
            "The weight 0.5 was fixed before looking at any unseen result.\n\n"
            "**3.4 Joint fusion (tried, not used).** Training both branches together in one network "
            "generalized worse on unseen generators.")

    with st.expander("4. Dataset and split"):
        st.markdown(
            "**Dataset:** ArtiFact (Kaggle), faces only, 200x200 JPEG images. Real faces: FFHQ.")
        st.table(pd.DataFrame([
            ["Train", "1,200", "1,200", "StyleGAN2, StyleGAN1, ProjectedGAN"],
            ["Validation (seen)", "300", "300", "StyleGAN2, StyleGAN1, ProjectedGAN"],
            ["Unseen test 1", "500", "500", "GANformer"],
            ["Unseen test 2", "500", "500", "CIPS"]],
            columns=["Split", "Real", "Fake", "Fake source"]))

    with st.expander("5. Training setup"):
        st.markdown(
            "- Spatial: 5 epochs, AdamW, learning rate 1e-4, batch size 32.\n"
            "- Frequency: 15 epochs, AdamW, learning rate 1e-3, batch size 32.\n"
            "- Horizontal flip augmentation, mixed precision, best epoch chosen on validation accuracy.\n"
            "- Each model was trained 3 times (seeds 1, 2, 3). This demo uses the weights of seed 1.\n"
            "- Hardware: laptop with an NVIDIA RTX 3050 GPU.")

    with st.expander("6. Results"):
        st.markdown("Mean ± standard deviation over 3 seeds (accuracy % / ROC-AUC).")
        st.table(pd.DataFrame([
            ["Spatial", "71.9±1.5 / 0.791", "68.6±1.2 / 0.753", "83.0±0.8 / 0.931"],
            ["Frequency", "61.4±0.8 / 0.655", "69.7±1.9 / 0.790", "77.7±4.3 / 0.921"],
            ["Late fusion", "72.2±2.0 / 0.792", "71.4±0.4 / 0.804", "84.7±1.4 / 0.958"]],
            columns=["Model", "Seen (validation)", "Unseen: GANformer", "Unseen: CIPS"]))
        st.markdown(
            "**Main findings**\n"
            "- Late fusion has the best accuracy and AUC on both unseen generators.\n"
            "- Neither single branch is best everywhere, so the two views look complementary.\n"
            "- Joint end-to-end fusion did worse on unseen generators than late fusion.\n"
            "- The gain of late fusion over the frequency branch on GANformer is within seed noise.")
        show_fig("fft_real_vs_fake.png", "Average FFT spectrum: real, fake (ProjectedGAN) and their difference")
        show_fig("fig_roc.png", "ROC curves (mean of 3 seeds)")
        show_fig("fig_confusion_unseen.png", "Confusion matrices on the unseen generators")
        show_fig("fig_accuracy_all.png", "Accuracy on seen and unseen data")

    with st.expander("7. Limitations"):
        st.markdown(
            "- Faces and GAN generators only. Diffusion models were not tested.\n"
            "- Only two unseen generators and three seeds, so conclusions are preliminary.\n"
            "- Small, JPEG-compressed 200x200 images and a small training set (2,400 images).\n"
            "- About a quarter of real faces are flagged as fake on the unseen sets.\n"
            "- Robustness to resizing, compression and deliberate attacks was not tested.")

    with st.expander("8. Future scope"):
        st.markdown(
            "- More generators, especially diffusion models.\n"
            "- Robustness tests (resizing, compression).\n"
            "- Explanations such as Grad-CAM.\n"
            "- Wavelet or noise-residual features, more training data, calibrated probabilities.")

    with st.expander("9. Tools used"):
        st.markdown(
            "Python, PyTorch and torchvision, NumPy (FFT), Pillow, scikit-learn, matplotlib, pandas, "
            "Jupyter, Kaggle notebooks (to build the data subset), Streamlit (this app).")

    with st.expander("10. Team and guide"):
        st.markdown(
            "**Team:** Lokesh Kumar, Dhruv Khurana, Sameer Yadav  \n"
            "**Guide:** Dr. Sudhir Dawra, Department of Computer Science and Engineering, SRM Institute of "
            "Science and Technology, Delhi-NCR, Ghaziabad")
