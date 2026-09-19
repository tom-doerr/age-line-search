import pandas as pd
import streamlit as st
from PIL import Image

from age_search import FORMATS, MODELS, build_prompt, line_search, load_scorer, make_score_fn

CUSTOM = "custom…"

st.set_page_config(page_title="Age line search", layout="wide")
st.title("Age line search")


@st.cache_resource(max_entries=1, show_spinner="Loading model…")
def get_scorer(family: str, model_id: str, device: str):
    return load_scorer(family, model_id, device)


with st.sidebar:
    family = st.radio("Model", list(MODELS), horizontal=True)
    model_id = st.selectbox("Checkpoint", MODELS[family])
    device = st.selectbox("Device", ["cuda", "cpu"])

    format_name = st.selectbox("Age text format", list(FORMATS) + [CUSTOM])
    if format_name == CUSTOM:
        template = st.text_input("Template (use {age})", "a photo of a {age} year old person")
    else:
        template = FORMATS[format_name]
    as_words = st.checkbox("Spell the number as words (e.g. thirty-four)")

    left, right = st.columns(2)
    min_age = int(left.number_input("Min age (years)", min_value=0, max_value=150, value=1, step=1))
    max_age = int(right.number_input("Max age (years)", min_value=0, max_value=150, value=90, step=1))

    method = st.radio(
        "Search", ["scan", "golden"],
        format_func={"scan": "Full scan", "golden": "Golden-section"}.get,
        help="Full scan evaluates every step. Golden-section needs few evaluations "
             "but assumes the similarity curve has a single peak.",
    )
    step = int(st.number_input("Scan step (years)", min_value=1, max_value=50, value=1, disabled=method != "scan"))

upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])

if "{age}" not in template:
    st.error("The template must contain the placeholder {age}.")
    st.stop()
if min_age > max_age:
    st.error("Min age must not be larger than max age.")
    st.stop()
st.caption(f"Example prompt: “{build_prompt(template, min_age, as_words)}”")

if upload is None:
    st.info("Upload an image to start.")
    st.stop()

image = Image.open(upload).convert("RGB")
image_col, result_col = st.columns([1, 2])
image_col.image(image, width="stretch")

scorer = get_scorer(family, model_id, device)
with st.spinner("Searching…"):
    result = line_search(
        make_score_fn(scorer, image), template, min_age, max_age,
        method=method, step=step, as_words=as_words, logit_scale=scorer.logit_scale,
    )

with result_col:
    a, b, c = st.columns(3)
    a.metric("Best age", f"{result.best_age} years")
    b.metric("Softmax-weighted age", f"{result.expected_age:.1f} years")
    c.metric("Prompts evaluated", result.evaluations)
    table = pd.DataFrame({"age": result.ages, "similarity": result.scores, "prompt": result.prompts})
    st.line_chart(table, x="age", y="similarity")
    st.dataframe(table.sort_values("similarity", ascending=False), hide_index=True, width="stretch")
