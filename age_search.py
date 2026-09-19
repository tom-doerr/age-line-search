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


def build_prompt(template: str, age: int, as_words: bool = False) -> str:
    if "{age}" not in template:
        raise ValueError("template must contain the placeholder {age}")
    return template.replace("{age}", number_to_words(age) if as_words else str(age))


@dataclass
class SearchResult:
    ages: List[int]            # every age that was evaluated, ascending
    prompts: List[str]
    scores: List[float]        # cosine similarity per evaluated age
    best_age: int
    expected_age: float        # softmax(scale * score)-weighted mean age
    evaluations: int


def _finish(cache: Dict[int, float], prompts: Dict[int, str], logit_scale: float) -> SearchResult:
    ages = sorted(cache)
    scores = np.array([cache[a] for a in ages])
    logits = logit_scale * scores
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return SearchResult(
        ages=ages,
        prompts=[prompts[a] for a in ages],
        scores=scores.tolist(),
        best_age=ages[int(scores.argmax())],
        expected_age=float((probs * np.array(ages)).sum()),
        evaluations=len(ages),
    )


def line_search(
    score_fn: Callable[[Sequence[str]], Sequence[float]],
    template: str,
    min_age: int,
    max_age: int,
    method: str = "scan",
    step: int = 1,
    as_words: bool = False,
    logit_scale: float = 100.0,
) -> SearchResult:
    """score_fn maps a list of prompts to image-text similarities.

    method "scan": evaluate every `step` years between min and max (exhaustive).
    method "golden": golden-section search on the integer age axis; needs far
    fewer evaluations but assumes the similarity curve has a single peak.
    """
    if min_age > max_age:
        raise ValueError(f"min age {min_age} is larger than max age {max_age}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    cache: Dict[int, float] = {}
    prompts: Dict[int, str] = {}

    def evaluate(ages: Sequence[int]) -> None:
        new = [a for a in dict.fromkeys(ages) if a not in cache]
        if not new:
            return
        for a in new:
            prompts[a] = build_prompt(template, a, as_words)
        values = score_fn([prompts[a] for a in new])
        if len(values) != len(new):
            raise RuntimeError(f"score_fn returned {len(values)} scores for {len(new)} prompts")
        cache.update({a: float(v) for a, v in zip(new, values)})

    if method == "scan":
        ages = list(range(min_age, max_age + 1, step))
        if ages[-1] != max_age:
            ages.append(max_age)
        evaluate(ages)
    elif method == "golden":
        inv_phi = (5 ** 0.5 - 1) / 2
        lo, hi = min_age, max_age
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
    return _finish(cache, prompts, logit_scale)


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
        return (scorer.embed_texts(texts) @ image_emb).tolist()

    return score
