"""Age estimation by line search over text prompts with a contrastive
image-text model (CLIP or TIPS). No streamlit imports here so it stays testable."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

FORMATS: Dict[str, str] = {
    "photo of a N year old person": "a photo of a {age} year old person",
    "N year old": "a {age} year old",
    "N-year-old": "a {age}-year-old",
    "N years old": "{age} years old",
    "age N": "age {age}",
    "person aged N": "a person aged {age}",
    "face of a N year old": "the face of a {age} year old",
}

MODELS: Dict[str, List[str]] = {
    "CLIP": [
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-base-patch32",
    ],
    "TIPS": [
        "google/tipsv2-l14",
        "google/tipsv2-b14",
    ],
}

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_to_words(n: int) -> str:
    if not 0 <= n < 1000:
        raise ValueError(f"number_to_words supports 0..999, got {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(n, 100)
    return f"{_ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")


def format_age(age: float, as_words: bool = False) -> str:
    """34.0 -> "34", 34.5 -> "34.5"; as words: "thirty-four point five"."""
    text = f"{round(age, 6):g}"
    if not as_words:
        return text
    whole, _, fraction = text.partition(".")
    words = number_to_words(int(whole))
    if fraction:
        words += " point " + " ".join(_ONES[int(d)] for d in fraction)
    return words


def build_prompt(template: str, age: float, as_words: bool = False) -> str:
    if "{age}" not in template:
        raise ValueError("template must contain the placeholder {age}")
    return template.replace("{age}", format_age(age, as_words))


def age_grid(min_age: float, max_age: float, step: float) -> List[float]:
    """min, min+step, ... plus max itself if the steps do not land on it."""
    if min_age > max_age:
        raise ValueError(f"min age {min_age} is larger than max age {max_age}")
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    count = int((max_age - min_age) / step + 1e-9) + 1
    grid = [round(min_age + i * step, 6) for i in range(count)]
    if grid[-1] != round(max_age, 6):
        grid.append(round(max_age, 6))
    return grid


@dataclass
class SearchResult:
    ages: List[float]          # every age that was evaluated, ascending
    prompts: List[str]
    scores: List[float]        # cosine similarity per evaluated age
    best_age: float
    expected_age: float        # softmax(scale * score)-weighted mean age
    evaluations: int


def _finish(grid: List[float], cache: Dict[int, float], prompts: Dict[int, str], logit_scale: float) -> SearchResult:
    indices = sorted(cache)
    ages = [grid[i] for i in indices]
    scores = np.array([cache[i] for i in indices])
    logits = logit_scale * scores
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return SearchResult(
        ages=ages,
        prompts=[prompts[i] for i in indices],
        scores=scores.tolist(),
        best_age=ages[int(scores.argmax())],
        expected_age=float((probs * np.array(ages)).sum()),
        evaluations=len(ages),
    )


def line_search(
    score_fn: Callable[[Sequence[str]], Sequence[float]],
    template: str,
    min_age: float,
    max_age: float,
    method: str = "scan",
    step: float = 1,
    as_words: bool = False,
    logit_scale: float = 100.0,
) -> SearchResult:
    """score_fn maps a list of prompts to image-text similarities.

    Both methods work on the grid min, min+step, ..., max (step may be < 1).
    method "scan": evaluate every grid point (exhaustive).
    method "golden": golden-section search over the grid indices; needs far
    fewer evaluations but assumes the similarity curve has a single peak.
    """
    grid = age_grid(min_age, max_age, step)
    cache: Dict[int, float] = {}
    prompts: Dict[int, str] = {}

    def evaluate(indices: Sequence[int]) -> None:
        new = [i for i in dict.fromkeys(indices) if i not in cache]
        if not new:
            return
        for i in new:
            prompts[i] = build_prompt(template, grid[i], as_words)
        values = score_fn([prompts[i] for i in new])
        if len(values) != len(new):
            raise RuntimeError(f"score_fn returned {len(values)} scores for {len(new)} prompts")
        cache.update({i: float(v) for i, v in zip(new, values)})

    if method == "scan":
        evaluate(range(len(grid)))
    elif method == "golden":
        inv_phi = (5 ** 0.5 - 1) / 2
        lo, hi = 0, len(grid) - 1
        while hi - lo > 2:
            c = round(hi - inv_phi * (hi - lo))
            d = round(lo + inv_phi * (hi - lo))
            if c >= d:  # rounding collapsed the probes; keep them distinct
                c, d = d - 1, d
            evaluate([c, d])
            if cache[c] > cache[d]:
                hi = d
            else:
                lo = c
        evaluate(range(lo, hi + 1))
    else:
        raise ValueError(f"unknown method {method!r}")
    return _finish(grid, cache, prompts, logit_scale)


class ClipScorer:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.logit_scale = float(self.model.logit_scale.exp())

    def embed_image(self, image):
        torch = self.torch
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            emb = self.model.visual_projection(self.model.vision_model(pixel_values=pixels).pooler_output)
        return torch.nn.functional.normalize(emb.float(), dim=-1)[0]

    def embed_texts(self, texts: Sequence[str]):
        torch = self.torch
        tokens = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            pooled = self.model.text_model(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            ).pooler_output
            emb = self.model.text_projection(pooled)
        return torch.nn.functional.normalize(emb.float(), dim=-1)


class TipsScorer:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        self.image_size = self.model.config.img_size
        self.logit_scale = 1.0 / float(self.model.config.temperature)

    def embed_image(self, image):
        torch = self.torch
        # TIPS expects plain [0, 1] pixels, no mean/std normalisation
        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        emb = self.model.encode_image(pixels.to(self.device)).cls_token[:, 0]
        return torch.nn.functional.normalize(emb.float(), dim=-1)[0]

    def embed_texts(self, texts: Sequence[str]):
        emb = self.model.encode_text(list(texts))
        return self.torch.nn.functional.normalize(emb.float(), dim=-1)


def load_scorer(family: str, model_id: str, device: str):
    if family == "CLIP":
        return ClipScorer(model_id, device)
    if family == "TIPS":
        return TipsScorer(model_id, device)
    raise ValueError(f"unknown model family {family!r}")


def make_score_fn(scorer, image) -> Callable[[Sequence[str]], List[float]]:
    """Embed the image once; every later call only runs the text encoder."""
    image_emb = scorer.embed_image(image)

    def score(texts: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(texts), 256):  # bound memory for fine steps
            scores += (scorer.embed_texts(texts[start:start + 256]) @ image_emb).tolist()
        return scores

    return score
