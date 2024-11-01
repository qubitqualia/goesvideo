import sys
import tempfile
from importlib.resources import files as importfiles
from pathlib import Path

import pytest
from PIL import Image

from goesvideo import GoesCompositor


def test_compositing():
    # Toggle display of image
    show_image = True

    # Create temp folder
    tmpfolder = tempfile.TemporaryDirectory()
    tmppath = Path(tmpfolder.name)
    imgpath = tmppath / "Images"
    imgpath.mkdir()

    sat = "goes-east"
    region = "full"
    product = "ABI-L2-CMIP"

    # Create composite
    gc = GoesCompositor(sat, region, product, base_dir=tmpfolder.name)
    basepath = importfiles("goesvideo") / "tests" / "Test NC Files"
    scene_dict = {
        "C01": [
            str(
                basepath
                / "OR_ABI-L2-CMIPF-M6C01_G16_s20231411110208_e20231411119516_c20231411119596.nc"
            )
        ],
        "C02": [
            str(
                basepath
                / "OR_ABI-L2-CMIPF-M6C02_G16_s20231411110208_e20231411119516_c20231411119590.nc"
            )
        ],
        "C03": [
            str(
                basepath
                / "OR_ABI-L2-CMIPF-M6C03_G16_s20231411110208_e20231411119516_c20231411119596.nc"
            )
        ],
    }

    gc.composites_from_files("true_color", scene_dict, folder_name=str(imgpath))

    # Check file
    files = list(imgpath.glob("*.png"))

    assert len(files) > 0

    if show_image:
        img = Image.open(str(files[0]))

        img.show()

    tmpfolder.cleanup()
    return


if __name__ == "__main__":
    sys.exit(pytest.main())
