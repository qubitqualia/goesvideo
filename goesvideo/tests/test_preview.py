import sys
import tempfile
from pathlib import Path
from importlib.resources import files as importfiles

from PIL import Image
import pytest
import pytz

from goesvideo import GoesAnimator


def test_preview():
    tmpfolder = tempfile.TemporaryDirectory()
    tmppath = Path(tmpfolder.name)

    ga = GoesAnimator("goes-east", "full", "ABI-L2-CMIP", base_dir=str(tmppath))

    imgpath = importfiles("goesvideo") / "tests" / "Test Images"
    imgfiles = list(imgpath.glob("*.png"))

    fontpath = str(
        importfiles("goesvideo") / "tests" / "Fonts" / "StoryElementRegular-X3RWa.ttf"
    )

    timestamps = {
        "label": "2023-01-01 00:00:00 UTC",
        "position": "upper-right",
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "tzinfo": (pytz.utc, "UTC"),
        "fontsize": 5,
        "opacity": 0.7,
    }

    text = {
        "label": "this is a test label",
        "position": (75, 75),
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "fontsize": 5,
        "opacity": 0.7,
    }

    arrow = {
        "label": {
            "label": "this is an arrow",
            "padding": (10, 10),
            "fontpath": fontpath,
            "fontcolor": (255, 0, 0),
            "fontsize": 5,
            "opacity": 0.7,
        },
        "tiplength": 0.1,
        "width": 2,
        "color": (255, 0, 0),
        "opacity": 0.7,
        "start_position": (45, 45),
        "end_position": (65, 65),
    }

    circle = {
        "label": {
            "label": "this is a circle",
            "padding": (10, 10),
            "fontpath": fontpath,
            "fontcolor": (255, 0, 0),
            "fontsize": 5,
            "opacity": 0.7,
        },
        "fill": (255, 0, 0),
        "outline": (0, 0, 0),
        "width": 1,
        "radius": 5,
        "centerpos": (100, 100),
    }

    kwargs = {}
    kwargs["timestamps"] = timestamps
    kwargs["text"] = text
    kwargs["arrow"] = arrow
    kwargs["circle"] = circle

    img = ga.preview(use_image_file=imgfiles[0], **kwargs, display=False)

    assert isinstance(img, Image.Image)

    img.show()
    return


if __name__ == "__main__":
    sys.exit(pytest.main())
