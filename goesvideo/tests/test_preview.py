import os
import shutil
import sys
import tempfile
from importlib.resources import files as importfiles
from pathlib import Path

import pytest
import pytz
from PIL import Image

from goesvideo import GoesAnimator
from goesvideo.tests import treegenerator

try:
    import cv2

    has_cv2 = True
except ModuleNotFoundError:
    has_cv2 = False


def test_preview():
    # Toggle display of image
    show_image = True

    # Generate expected folder tree
    tmpfolder = tempfile.TemporaryDirectory()
    base_dir = Path(tmpfolder.name)
    treegenerator.generate_tree(base_dir, "BigHorn", copy_ncfiles=True)

    ga = GoesAnimator("goes-east", "full", "ABI-L2-CMIP", base_dir=base_dir)

    imgpath = importfiles("goesvideo") / "tests" / "Test Images"
    imgfiles = list(imgpath.glob("*.png"))

    fontpath = str(importfiles("goesvideo") / "tests" / "Fonts" / "Roboto-Regular.ttf")

    timestamps = {
        "label": "2023-01-01 00:00:00",
        "position": "upper-left",
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "tzinfo": (pytz.utc, "UTC"),
        "fontsize": 20,
        "opacity": 0.7,
    }

    text = {
        "label": "this is a test label",
        "position": (600, 600),
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "fontsize": 20,
        "opacity": 0.7,
    }

    if has_cv2:
        arrow = {
            "label": {
                "label": "this is an arrow",
                "padding": (10, 10),
                "fontpath": fontpath,
                "fontcolor": (255, 0, 0),
                "fontsize": 20,
                "opacity": 0.7,
            },
            "tiplength": 0.1,
            "width": 2,
            "color": (255, 0, 0),
            "opacity": 0.7,
            "start_position": (200, 200),
            "end_position": (400, 400),
        }

    circle = {
        "label": {
            "label": "this is a circle",
            "padding": (10, 10),
            "fontpath": fontpath,
            "fontcolor": (255, 0, 0),
            "fontsize": 20,
            "opacity": 0.3,
        },
        "fill": (255, 0, 0),
        "outline": (0, 0, 0),
        "width": 1,
        "radius": 5,
        "centerpos": (500, 500),
    }

    kwargs = {}
    kwargs["timestamps"] = timestamps
    kwargs["text"] = text
    if has_cv2:
        kwargs["arrow"] = arrow
    kwargs["circle"] = circle

    testimg = Image.open(imgfiles[0])
    testimg = testimg.resize((1024, 768))
    tmpimgfile = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False)
    testimg.save(tmpimgfile.name)
    testimg.close()
    img = ga.preview(
        use_image_file=tmpimgfile.name, **kwargs, res=(1024, 768), display=False
    )

    assert isinstance(img, Image.Image)

    if show_image:
        img.show()

    img.close()

    # Cleanup
    try:
        tmpimgfile.close()
        os.unlink(tmpimgfile.name)
        shutil.rmtree(base_dir)
    except FileNotFoundError:
        pass
    except PermissionError:
        pass

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
