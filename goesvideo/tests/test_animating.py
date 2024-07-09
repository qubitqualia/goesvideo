import os
import sys
import tempfile
from pathlib import Path
from importlib.resources import files as importfiles
import shutil

import pytest

from goesvideo import GoesAnimator


def test_animating():
    # Create temp folder
    tmpfolder = tempfile.TemporaryDirectory()
    tmppath = Path(tmpfolder.name)
    imgpath = tmppath / "Images"
    imgsubpath = imgpath / "subfolder"
    imgpath.mkdir(exist_ok=True)
    imgsubpath.mkdir(exist_ok=True)
    vidsubpath = tmppath / "Videos" / "subfolder"

    # Copy test images to tmpfolder
    imgpath = importfiles("goesvideo") / "tests" / "Test Images"

    files = imgpath.glob("*.png")
    for f in files:
        shutil.copy(str(f), str(imgsubpath / (f.stem + f.suffix)))
    shutil.copy(str(imgpath / "metadata.json"), str(imgsubpath / "metadata.json"))
    shutil.copy(str(imgpath / "timestamps.csv"), str(imgsubpath / "timestamps.csv"))

    # Create videos
    exists = []
    ga = GoesAnimator("goes-east", "full", "ABI-L2-CMIP", base_dir=str(tmppath))
    ga.create_video("C01", from_existing_imgs=True, fps=1, force=True)
    exists.append((vidsubpath / "video.mp4").exists())
    os.remove(str(vidsubpath / "video.mp4"))

    ga = GoesAnimator("goes-east", "full", "ABI-L2-CMIP", base_dir=str(tmppath))
    ga.create_video("C01", from_existing_imgs=True, fps=1, cmap="Spectral", force=True)
    exists.append((vidsubpath / "video.mp4").exists())
    os.remove(str(vidsubpath / "video.mp4"))

    ga = GoesAnimator("goes-east", "full", "ABI-L2-CMIP", base_dir=str(tmppath))
    ga.create_video("C01", from_existing_imgs=True, fps=1, force=True)
    exists.append((vidsubpath / "video.mp4").exists())
    os.remove(str(vidsubpath / "video.mp4"))

    assert all(exists)
    tmpfolder.cleanup()

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
