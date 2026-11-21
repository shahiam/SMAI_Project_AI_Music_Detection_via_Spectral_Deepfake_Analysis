from music_detection.sonics.preprocessing.download_youtube import section_for
from music_detection.sonics.preprocessing.prepare_clips import segment_starts
from music_detection.sonics.preprocessing.utils import allocate, normalise_split


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
