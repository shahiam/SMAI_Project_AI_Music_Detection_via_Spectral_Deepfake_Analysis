import math

import pandas as pd

from music_detection.sonics.preprocessing.download_youtube import section_for
from music_detection.sonics.preprocessing.prepare_clips import (
    balance_groups,
    clip_region,
    parse_volumedetect,
    segment_starts,
)
from music_detection.sonics.preprocessing.utils import allocate, coarse_genre, normalise_split


def test_normalise_split_maps_sonics_names():
    assert normalise_split(" Valid ") == "val"
    assert normalise_split("train") == "train"


def test_allocate_is_proportional_and_exact():
    assert allocate(10, {"a": 50, "b": 30, "c": 20}) == {"a": 5, "b": 3, "c": 2}
    assert sum(allocate(7, {"a": 1, "b": 100}).values()) == 7
    capped = allocate(10, {"a": 50, "b": 50}, caps={"a": 2, "b": 50})
    assert capped["a"] == 2 and sum(capped.values()) == 10


def test_segment_starts_stay_inside_the_song():
    starts = segment_starts(120.0, skip=2.0, k=3, seg=10.0, frac_start=0.1, frac_end=0.9)
    assert len(starts) == 3
    assert starts[0] >= 12.0 - 1e-6 and starts[-1] + 10.0 <= 108.0 + 1e-6
    assert segment_starts(4.0, 0.0, 3, 10.0, 0.1, 0.9) == [0.0]


def test_section_for_clamps_to_the_song():
    assert section_for(176, 40, 0.25) == (44.0, 84.0)
    assert section_for(30, 40, 0.25) == (0.0, 30.0)


def test_coarse_genre_maps_prompt_styles_and_lyrics_site_tags():
    cases = {
        "Hip Hop": "hiphop", "trap": "hiphop", "rb": "rnb_soul", "R&B": "rnb_soul",
        "heavy metal": "metal", "indie rock": "rock", "synthwave": "electronic", "k-pop": "pop",
        "country": "country_folk", "classical": "classical", "misc": "other",
    }
    for raw, expected in cases.items():
        assert coarse_genre(raw) == expected, raw
    assert coarse_genre(float("nan")) == "unknown"
    assert coarse_genre("") == "unknown"


def test_clip_region_matches_human_sections():
    assert clip_region(180.0, 2.0, False, 0.25, 40.0) == (45.0, 85.0)  # AI song: 40 s from 25%
    assert clip_region(32.0, 1.0, False, 0.25, 40.0) == (0.0, 32.0)    # short AI song: whole song
    assert clip_region(40.0, 0.0, True, 0.25, 40.0) == (0.0, 40.0)     # downloaded human section


def test_parse_volumedetect():
    text = "[Parsed_volumedetect_0 @ 0x1] mean_volume: -18.3 dB\n[Parsed_volumedetect_0 @ 0x1] max_volume: -0.5 dB\n"
    assert parse_volumedetect(text) == (-18.3, -0.5)
    mean, peak = parse_volumedetect("mean_volume: -inf dB\nmax_volume: -inf dB")
    assert math.isinf(mean) and math.isinf(peak)
    assert parse_volumedetect("nothing here") == (None, None)


def test_balance_groups_equalises_every_split_and_genre():
    reals = pd.DataFrame({"split": ["train"] * 4, "genre": ["pop", "pop", "pop", "rock"],
                          "generator": "human", "sonics_label": "real"})
    fakes = pd.DataFrame({"split": ["train"] * 6, "genre": ["pop", "pop", "rock", "rock", "rock", "rock"],
                          "generator": ["a", "b", "a", "a", "b", "b"], "sonics_label": "full fake"})
    kept_real, kept_fake = balance_groups(reals, fakes, ["split", "genre"], seed=0)
    assert kept_real["genre"].value_counts().to_dict() == {"pop": 2, "rock": 1}
    assert kept_fake["genre"].value_counts().to_dict() == {"pop": 2, "rock": 1}
