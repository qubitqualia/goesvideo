import sys
from importlib.resources import files as importfiles

from PIL import Image
import pytest
from moviepy.video.VideoClip import VideoClip

from goesvideo.utils import editortools


def test_video_edit():
    # Open test video
    clip = editortools.GoesClip(
        str(importfiles("goesvideo") / "tests" / "Test Videos" / "video.mp4")
    )

    fontpath = str(
        importfiles("goesvideo") / "tests" / "Fonts" / "StoryElementRegular-X3RWa.ttf"
    )

    # Test video annotation
    kwargs = {
        "label": "this is a test label",
        "position": (75, 75),
        "fontpath": fontpath,
        "fontcolor": (255, 0, 0),
        "fontsize": 5,
        "opacity": 0.7,
    }
    newclip = clip.annotate(0, 1, freeze=True, **kwargs)

    assert isinstance(newclip, VideoClip)

    # Test preview
    img = clip.preview(tmark=2, **kwargs)

    assert isinstance(img, Image.Image)
    img.show()

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
