import re

import pytest

from age_search import FORMATS, build_prompt, line_search, number_to_words


def peak_at(target):
    def score(prompts):
        return [-abs(int(re.search(r"\d+", p).group()) - target) / 100 for p in prompts]
    return score


def test_number_to_words():
    assert number_to_words(0) == "zero"
    assert number_to_words(17) == "seventeen"
    assert number_to_words(40) == "forty"
    assert number_to_words(34) == "thirty-four"
    assert number_to_words(101) == "one hundred one"


def test_build_prompt():
    assert build_prompt("age {age}", 7) == "age 7"
    assert build_prompt("age {age}", 21, as_words=True) == "age twenty-one"
    with pytest.raises(ValueError):
        build_prompt("no placeholder", 3)


def test_all_formats_have_placeholder():
    assert all("{age}" in t for t in FORMATS.values())


def test_scan_finds_peak_and_includes_max():
    result = line_search(peak_at(37), "age {age}", 0, 90)
    assert result.best_age == 37 and result.evaluations == 91
    stepped = line_search(peak_at(37), "age {age}", 0, 89, step=10)
    assert stepped.ages[-1] == 89 and stepped.best_age == 40


@pytest.mark.parametrize("target", [0, 1, 37, 62, 89, 90])
def test_golden_finds_peak_with_few_evaluations(target):
    result = line_search(peak_at(target), "age {age}", 0, 90, method="golden")
    assert result.best_age == target
    assert result.evaluations < 25


def test_expected_age_near_peak():
    result = line_search(peak_at(30), "age {age}", 0, 60, logit_scale=100)
    assert abs(result.expected_age - 30) < 0.5


def test_bad_arguments():
    with pytest.raises(ValueError):
        line_search(peak_at(5), "age {age}", 10, 5)
    with pytest.raises(ValueError):
        line_search(peak_at(5), "age {age}", 0, 10, method="newton")
    assert line_search(peak_at(5), "age {age}", 5, 5).best_age == 5
