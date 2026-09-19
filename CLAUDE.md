# age-line-search

Private repo github.com/tom-doerr/age-line-search. Streamlit age estimation via
line search over age prompts with CLIP or TIPSv2.

- `age_search.py` = all logic (formats, number words, `line_search`, `ClipScorer`,
  `TipsScorer`); no streamlit import, torch imported lazily inside the scorers so
  the tests need no GPU. `app.py` = UI only. `tests/` = 16 tests with a fake score_fn.
- Run: `./run.sh` (port 8540, system python3 — it carries the GB10 cu130 torch;
  do not create a sealed venv). Tests: `python3 -m pytest -q`.
- Models used from the HF cache: `openai/clip-vit-large-patch14` (+ -336, base-patch32),
  `google/tipsv2-l14` (`google/tipsv2-b14` would download). `HF_HUB_OFFLINE=1` works.
- TIPSv2: `trust_remote_code`, input = raw [0,1] pixels squashed to 448x448 (no mean/std),
  image embedding = `encode_image(...).cls_token[:, 0]`, tokenisation happens inside
  `encode_text`, logit scale = 1/`config.temperature` (~209; CLIP = 100).
- CLIP embeddings are computed as `visual_projection(vision_model(...).pooler_output)`
  (and the text twin) instead of `get_image_features`, whose return type changed in
  transformers 5. The "UNEXPECTED position_ids" load report for CLIP is benign.
- Golden-section assumes a unimodal similarity curve; real curves are jagged
  (e.g. TIPS on the test image: scan says 21, golden 12) — full scan is the default
  and costs ~1.5 s for 90 prompts, so golden is for demonstration, not speed.
- `st.cache_resource(max_entries=1)` keeps only ONE model resident (UMA memory).
  Device is an explicit select (cuda/cpu) — no silent CPU fallback if CUDA fails.
- Headless UI test trick: `AppTest.from_string` with `st.file_uploader` patched to
  return an open image file, `PYTHONPATH=.`.
- Ages are FLOATS on a grid built by index (`age_grid`: min + i*step, rounded to 6
  places, max appended if the steps miss it); step is a float input (min 0.01), so
  prompts can read "34.5" / "thirty-four point five" (`format_age`, `%g` drops ".0").
  Both scan and golden-section search the grid INDICES, so golden works at any step.
  Full scan refuses > 2000 prompts (UI error); text encoding is chunked by 256.
- **Run it as a systemd unit, not as a Claude background task** (the harness killed the
  task under memory pressure, Sep 19 2026): `systemd-run --user --unit=age-line-search
  --collect -p MemoryMax=12G -p WorkingDirectory=$PWD --setenv=HF_HUB_OFFLINE=1
  --setenv=PATH="$PATH" $PWD/run.sh`; stop `systemctl --user stop age-line-search`,
  logs `journalctl --user -u age-line-search`. Transient = gone after a reboot.
- Streamlit re-runs `app.py` on edit but may keep the OLD imported `age_search` module:
  a traceback pointing at a harmless line (e.g. the signature) = stale module → restart.
