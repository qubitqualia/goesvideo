from pathlib import Path
from PIL import Image
from goesvideo import utils
from importlib.resources import files as importfiles
import pytz
import sys
import pytest


def test_annotations():
    # Grab test image
    p = importfiles("goesvideo") / "tests" / "Test Images"

    fnames = list(p.glob("*.png"))
    baseimg = Image.open(str(fnames[0]))

    # Apply annotations
    fontpath = str(
        importfiles("goesvideo") / "tests" / "Fonts" / "StoryElementRegular-X3RWa.ttf"
    )
    exists = []

    # Timestamps
    kwargs = {
        "position": "upper-right",
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "tzinfo": (pytz.utc, "UTC"),
        "fontsize": 5,
        "opacity": 0.7,
    }
    img = utils.add_timestamps(baseimg, "2023-10-01 00:00:00", **kwargs)
    exists.append(img)

    # Text
    kwargs = {
        "label": "this is a test label",
        "position": (75, 75),
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "fontsize": 5,
        "opacity": 0.7,
    }
    img = utils.add_text(baseimg, **kwargs)
    exists.append(img)

    # Arrow
    startpos = (75, 75)
    endpos = (45, 45)
    kwargs = {
        "label": {
            "label": "this is an arrow",
            "padding": (3, 3),
            "fontpath": fontpath,
            "fontcolor": (255, 0, 0),
            "fontsize": 5,
            "opacity": 0.7,
        },
        "tiplength": 0.1,
        "width": 2,
        "color": (255, 0, 0),
        "opacity": 0.7,
    }
    img = utils.add_arrow(baseimg, startpos, endpos, **kwargs)
    exists.append(img)

    # Circle
    radius = 3
    centerpos = (50, 50)
    kwargs = {
        "label": {
            "label": "this is a circle",
            "padding": (3, 3),
            "fontpath": fontpath,
            "fontcolor": (255, 0, 0),
            "fontsize": 5,
            "opacity": 0.7,
        },
        "fill": (255, 0, 0),
        "outline": (0, 0, 0),
        "width": 1,
    }
    img = utils.add_circle(baseimg, centerpos, radius, **kwargs)
    exists.append(img)

    assert all(isinstance(im, Image.Image) for im in exists)

    for im in exists:
        im.show()

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
