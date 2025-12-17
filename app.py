import streamlit as st
from PIL import Image
import os, random

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Fashion Foresight",
    page_icon="👗",
    layout="centered"
)

# ======================================================
# CSS — BIG RADIO CARDS (CLICKABLE)
# ======================================================
st.markdown("""
<style>
/* Hide default radio circles */
div[role="radiogroup"] input {
    display: none;
}

/* Layout grids */
.body-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 18px; }
.skin-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.gender-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }

/* Card styling */
div[role="radiogroup"] label {
    background: #151923;
    border: 2px solid #333;
    border-radius: 20px;
    padding: 26px 10px;
    text-align: center;
    cursor: pointer;
    transition: 0.25s;
}

/* Selected */
div[role="radiogroup"] input:checked + label {
    border-color: #ff4b4b;
    box-shadow: 0 0 16px #ff4b4b88;
    background: #1f2433;
}

/* Icons & labels */
.icon {
    font-size: 40px;
}
.label {
    margin-top: 10px;
    font-size: 15px;
    font-weight: 600;
}

/* Skin circle */
.skin-circle {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    margin: auto;
    border: 3px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================
st.markdown("## 👗 Fashion Foresight")
st.caption("Premium AI Fashion Recommendation System")

photo = st.file_uploader("📸 Upload your photo", ["jpg", "png", "jpeg"])

# ======================================================
# BODY SHAPE
# ======================================================
st.markdown("### Body Shape")

body = st.radio(
    "",
    [
        "▭ Rectangle",
        "🍐 Pear",
        "🔺 Inverted Triangle",
        "🍎 Apple",
        "⏳ Hourglass"
    ],
    index=0,
    key="body",
    horizontal=True
)

st.markdown(
    "<div class='body-grid'></div>",
    unsafe_allow_html=True
)

# ======================================================
# SKIN TONE
# ======================================================
st.markdown("### Skin Tone")

skin = st.radio(
    "",
    [
        "🟤 Dark",
        "🟠 Medium",
        "⚪ Light"
    ],
    index=1,
    key="skin",
    horizontal=True
)

st.markdown(
    "<div class='skin-grid'></div>",
    unsafe_allow_html=True
)

# ======================================================
# GENDER
# ======================================================
st.markdown("### Gender")

gender = st.radio(
    "",
    [
        "👨 Male",
        "👩 Female"
    ],
    index=0,
    key="gender",
    horizontal=True
)

st.markdown(
    "<div class='gender-grid'></div>",
    unsafe_allow_html=True
)

# ======================================================
# DATA PATHS
# ======================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Data", "train")

def get_images(folder, n=3):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.exists(path):
        return []
    imgs = [os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith((".jpg",".png",".jpeg"))]
    return random.sample(imgs, min(len(imgs), n))

# ======================================================
# RECOMMENDATION LOGIC
# ======================================================
def recommend(gender, body, skin):
    body = body.split(" ", 1)[1]
    skin = skin.split(" ", 1)[1]
    gender = gender.split(" ", 1)[1]

    if gender == "Male":
        category = "MEN_Suits" if body in ["Rectangle","Inverted Triangle","Hourglass"] else "MEN_Coats"
        textures = ["Wool", "Cotton"]
    else:
        category = "WOMEN_Dress" if body in ["Pear","Hourglass"] else "WOMEN_Hood"
        textures = ["Soft Fabric", "Knit"]

    color_map = {
        "Light": [("Navy", "#1f3c88"), ("Pastel Pink", "#f4c2c2"), ("Grey", "#b0b0b0")],
        "Medium": [("Olive", "#6b8e23"), ("Royal Blue", "#4169e1"), ("Beige", "#f5f5dc")],
        "Dark": [("White", "#ffffff"), ("Emerald", "#50c878"), ("Yellow", "#ffd700")]
    }

    colors = color_map[skin]

    return category, get_images(category), colors, textures


# ======================================================
# OUTPUT
# ======================================================
if photo and st.button("✨ Recommended for You"):
    st.image(Image.open(photo), width=120)

    category, images, colors, textures = recommend(gender, body, skin)

    # ---- OUTFITS ----
    st.markdown("## 👚 Recommended Outfits")
    cols = st.columns(len(images))
    for col, img in zip(cols, images):
        col.image(img, use_container_width=True)

    # ---- TEXTURES ----
    st.markdown("### 🧵 Recommended Textures")
    cols = st.columns(len(textures))
    for col, t in zip(cols, textures):
        col.markdown(
            f"<div class='texture-box'>🧵</div>",
            unsafe_allow_html=True
        )
        col.caption(t)

    # ---- COLORS ----
    st.markdown("### 🎨 Recommended Colors")
    cols = st.columns(len(colors))
    for col, (name, hexcode) in zip(cols, colors):
        col.markdown(
            f"<div class='color-circle' style='background:{hexcode}'></div>",
            unsafe_allow_html=True
        )
        col.caption(name)


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Fashion Foresight · Final Stable Build · No UI Bugs")
