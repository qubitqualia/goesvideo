import os
import shutil
import tempfile
from importlib.resources import files as importfiles
from pathlib import Path

import matplotlib.cm as cm
import pytz
from PIL import Image

from goesvideo.addons.overlays import GenericOverlay


def test_generic_overlays():
    # Toggle display of output images
    # Note that this will open a large number of
    # images for viewing
    show_images = False

    # Create temp folders
    tmpfolder = tempfile.TemporaryDirectory()
    tmppath = Path(tmpfolder.name)
    subpath = tmppath / "Top_Level"
    basepath = tmppath / "Base_Images"
    overlaypath = tmppath / "Overlay_Images_1"
    outputpath = tmppath / "Final_Output"
    gtiffpath = tmppath / "Gtiff_Output"
    basepath.mkdir(exist_ok=True)
    overlaypath.mkdir(exist_ok=True)
    outputpath.mkdir(exist_ok=True)
    gtiffpath.mkdir(exist_ok=True)

    # Copy test images to tmpfolders
    subpath = importfiles("goesvideo") / "tests" / "Test Images"

    # Sort and move images to correct folders
    basefiles = subpath.glob("*.png")
    overlayfiles = subpath.glob("*.tif")
    for f in basefiles:
        shutil.copy(f, str(basepath))
    for f in overlayfiles:
        shutil.copy(f, str(overlaypath))

    # Create overlay objects
    ol_timezone = {"timezone": [pytz.utc], "filename format": ["YYYY-MM-DD hh_mm_ss"]}
    base_timezone = {"timezone": pytz.utc, "filename format": "YYYY-MM-DD hh_mm_ss"}
    ol1 = GenericOverlay(
        str(basepath),
        [str(overlaypath)],
        base_timezone=base_timezone,
        overlay_timezones=ol_timezone,
    )
    ol2 = GenericOverlay(
        str(basepath),
        [str(overlaypath)],
        start_time="2024-10-08 17:19:00",
        end_time="2024-10-08 19:50:51",
    )

    # Generate overlays
    # Test 1 - Change CRS
    out_crs = "EPSG:4326"
    ol1.create_overlays(str(outputpath / "test1a"), out_crs=out_crs)
    ol2.create_overlays(str(outputpath / "test1b"), out_crs=out_crs)

    # Test 2 - Fully overlapped bbox
    bbox = [-90, 27, -82, 38]
    ol1.create_overlays(str(outputpath / "test2a"), bbox=bbox)
    ol2.create_overlays(str(outputpath / "test2b"), bbox=bbox)

    # Test 3 - Partially overlapped bbox
    bbox = [-105, 21, -90, 40]
    ol1.create_overlays(str(outputpath / "test3a"), bbox=bbox)
    ol2.create_overlays(str(outputpath / "test3b"), bbox=bbox)

    # Test 4 - Non-overlapping bbox
    bbox = [-110, -60, -100, -40]
    ol1.create_overlays(str(outputpath / "test4a"), bbox=bbox)
    ol2.create_overlays(str(outputpath / "test4b"), bbox=bbox)

    # Test 5 - Set overlay opacity to 0.3
    opacities = [0.3]
    ol1.create_overlays(str(outputpath / "test5a"), overlay_opacities=opacities)
    ol2.create_overlays(str(outputpath / "test5b"), overlay_opacities=opacities)

    # Test 6 - Save overlay images as geotiffs
    ol1.create_overlays(
        str(outputpath / "test6a"), save_overlays_gtiff=str(gtiffpath / "test6a")
    )
    ol2.create_overlays(
        str(outputpath / "test6b"), save_overlays_gtiff=str(gtiffpath / "test6b")
    )

    # Test 7 - Cumulative overlay
    ol1.create_overlays(str(outputpath / "test7a"), cumulative=True)
    ol2.create_overlays(str(outputpath / "test7b"), cumulative=True)

    # Test 8 - Cumulative overlay with cumulative colormap
    cmap = cm.get_cmap("Spectral")
    ol1.create_overlays(
        str(outputpath / "test8a"), cumulative=True, cumulative_colormap=cmap
    )
    ol2.create_overlays(
        str(outputpath / "test8b"), cumulative=True, cumulative_colormap=cmap
    )

    # Assertions to check files exist in every subdirectory
    subpaths = [
        "test1a",
        "test1b",
        "test2a",
        "test2b",
        "test3a",
        "test3b",
        "test4a",
        "test4b",
        "test5a",
        "test5b",
        "test6a",
        "test6b",
        "test7a",
        "test7b",
        "test8a",
        "test8b",
    ]

    for sp in subpaths:
        assert any((outputpath / sp).iterdir())

    # Show images if toggled on
    if show_images:
        for sp in subpaths:
            pngfile = list((outputpath / sp).glob("*.png"))
            tiffile = list((outputpath / sp).glob("*.tif"))
            if pngfile:
                Image.open(pngfile[0]).show()
            if tiffile:
                os.replace(tiffile[0], str(tiffile[0]).split(".")[0] + ".tiff")
                tf = list((outputpath / sp).glob("*.tiff"))
                Image.open(tf[0]).show()
