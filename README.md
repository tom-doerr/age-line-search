# age-line-search

Streamlit app: upload an image, estimate the age by line search over text
prompts with a contrastive image-text model (CLIP or TIPSv2).

    ./run.sh            # http://localhost:8540  (PORT=... to change)
    python3 -m pytest -q

The image is embedded once; every candidate age becomes a prompt such as
"a photo of a 34 year old person" and is scored by cosine similarity.

- **Model**: CLIP (`openai/clip-vit-*`) or TIPS (`google/tipsv2-*`), device cuda/cpu
- **Age text format**: preset templates, a custom `{age}` template, digits or spelled-out words
- **Min / max age** in years
- **Search**: full scan (every `step` years; the step may be fractional, e.g. 0.5) or golden-section (few evaluations, assumes one peak)

Output: best age, softmax-weighted age (using the model's own logit scale),
the similarity curve and the ranked prompt table.
