import sys
import tempfile
from pathlib import Path

import pytest

from goesvideo import GoesAnimator
from goesvideo.tests import treegenerator


def test_animating():
    # Create base dir tree
    tmppath = tempfile.TemporaryDirectory()
    base_dir = Path(tmppath.name)
    treegenerator.generate_tree(base_dir, "C02", copy_images=True)

    # Create videos
    exists = []
    ga = GoesAnimator("goes-east", "conus", "ABI-L2-CMIP", base_dir=str(base_dir))
    ga.create_video("C02", from_existing_imgs=True, fps=1, force=True)
    exists.append((base_dir / "Videos" / "C02").exists())

    ga = GoesAnimator("goes-east", "conus", "ABI-L2-CMIP", base_dir=str(base_dir))
    ga.create_video("C02", from_existing_imgs=True, fps=1, cmap="Spectral", force=True)
    exists.append((base_dir / "Videos" / "C02").exists())

    ga = GoesAnimator("goes-east", "conus", "ABI-L2-CMIP", base_dir=str(base_dir))
    ga.create_video("C02", from_existing_imgs=True, fps=1, force=True)
    exists.append((base_dir / "Videos" / "C02").exists())

    assert all(exists)

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
