import sys
from importlib.resources import files as importfiles

import pytest
from PIL import Image
from moviepy.video.VideoClip import VideoClip

from goesvideo.utils import editortools


def test_video_edit():
    # Toggle display of output image
    show_image = True

    # Open test video
    clip = editortools.GoesClip(
        str(importfiles("goesvideo") / "tests" / "Test Videos" / "video.mp4")
    )

    fontpath = str(importfiles("goesvideo") / "tests" / "Fonts" / "Roboto-Regular.ttf")

    # Test video annotation
    txt = {
        "label": "this is a test label",
        "position": (75, 75),
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "fontsize": 5,
        "opacity": 0.7,
    }
    kwargs = {}
    kwargs["text"] = txt
    newclip = clip.annotate(0, 1, freeze=True, **kwargs)

    assert isinstance(newclip, VideoClip)

    # Test preview
    img = clip.preview(tmark=2, display=False, **kwargs)

    assert isinstance(img, Image.Image)
    if show_image:
        img.show()

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
