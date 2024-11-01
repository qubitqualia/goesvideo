import json
import os
import shutil
import sys
import tempfile
import time
from importlib.resources import files as importfiles
from pathlib import Path

import pytest
import pytz
import rasterio.crs
from PIL import Image

from goesvideo.addons.sceneeditor import GoesSceneEditor
from goesvideo.tests import treegenerator
from goesvideo.utils import gistools


def check_metadata(key, metadata):
    crs = rasterio.crs.CRS.from_wkt(metadata["geodata"]["crs"])
    retval = (None, None)
    if key == "bbox":
        bbox = metadata["Crop_Box"]
        # if crs != "EPSG:4326":
        #    bbox = gistools.transform_bbox(bbox, crs, "EPSG:4326")
        retval = (bbox,)
    elif key == "size":
        width = metadata["geodata"]["width"]
        height = metadata["geodata"]["height"]
        retval = (width, height)
    elif key == "crs":
        retval = (crs,)

    return retval


def get_metadata(_dir):
    with open(str(_dir / "metadata.json"), "r") as f:
        metadata = json.load(f)

    return metadata


def test_scene_editor():
    """
    This test evaluates the functions in class sceneditor.GoesSceneEditor using the png and tif images located in
    tests/Test Images. The scene depicted in these images is GOES-East, Ch02 (png) and fire detection (tif) for
    Big Horn, WY on October 8, 2024, which is when a large wildfire began in that area. The test evaluates functions
    for resizing, recropping, reprojecting, adding timestamps and adding various annotations. At the end of these tests,
    a video is generated which shows the overlay of the fire detection product on the base GOES imagery of the area.
    """

    # Toggle display of output images; if true, this will only generate and display a single preview image for each
    # test function below; if false, all 8 png and 8 tif images in the tests/Test Images folder will be manipulated
    preview = False

    # Specify base dir if you wish to inspect the output of the test functions below
    base_dir = None

    # Set fontpath
    fontpath = str(importfiles("goesvideo") / "tests" / "Fonts" / "Roboto-Regular.ttf")

    # Grab test images - png base, tif overlay
    p = importfiles("goesvideo") / "tests" / "Test Images"

    png_fnames = list(p.glob("*.png"))
    png_count = len(png_fnames)
    tiff_fnames = list(p.glob("*.tif"))
    tiff_count = len(tiff_fnames)

    # Create standard folder structure using temp dir as base dir
    if base_dir:
        base_dir = base_dir
        delete_flag = False
    else:
        base_dir_tmp = tempfile.TemporaryDirectory()
        base_dir = Path(base_dir_tmp.name)
        delete_flag = True

    treegenerator.generate_tree(
        base_dir, "BigHorn", overlay_scene="BigHornFire", copy_images=True
    )
    scene_dir_base = Path(base_dir / "Scenes" / "BigHorn")
    scene_dir_overlay = Path(base_dir / "Scenes" / "BigHornFire")
    scene_edit_dir = Path(base_dir / "SceneEdits")
    scene_edit_dir_base = scene_edit_dir / "BigHorn"
    scene_edit_dir_overlay = scene_edit_dir / "BigHornFire"

    # Initialize scene editors
    editor = GoesSceneEditor(str(base_dir), "BigHorn", session_name="BigHorn_Base_Edit")
    oleditor = GoesSceneEditor(
        str(base_dir), "BigHornFire", session_name="BigHorn_Fire_Edit"
    )
    editor.set_font(fontpath=fontpath, fontsize=20, fontcolor=(255, 0, 0))

    metadata_base = get_metadata(scene_dir_base)
    metadata_overlay = get_metadata(scene_dir_overlay)

    assert (
        len(list((scene_edit_dir_base / "BigHorn_Base_Edit").glob("*.png")))
        == png_count
    )
    assert (
        len(list((scene_edit_dir_overlay / "BigHorn_Fire_Edit").glob("*.tif")))
        == tiff_count
    )

    # Recrop images
    # original bbox: [-108.09, 43.794, -106.43, 45.25]
    new_bbox = (-108.088, 43.796, -106.432, 45.23)
    editor.recrop(new_bbox, preview=preview)
    oleditor.recrop(new_bbox, preview=preview)
    if not preview:
        metadata_base = get_metadata(scene_edit_dir_base / "BigHorn_Base_Edit")
        metadata_overlay = get_metadata(scene_edit_dir_overlay / "BigHorn_Fire_Edit")
        assert (
            len(list((scene_edit_dir_base / "BigHorn_Base_Edit").glob("*.png")))
            == png_count
        )
        assert (
            len(list((scene_edit_dir_overlay / "BigHorn_Fire_Edit").glob("*.tif")))
            == tiff_count
        )
        assert gistools.check_bbox(
            new_bbox, check_metadata("bbox", metadata_base)[0], degtol=0.5
        ) == (True, True)
        assert gistools.check_bbox(
            new_bbox, check_metadata("bbox", metadata_overlay)[0], degtol=0.5
        ) == (True, True)

    # Resize images
    # original size: 322 x 207
    w, h = (1024, 768)
    editor.resize(w, h, preview=preview)
    oleditor.resize(w, h, preview=preview)
    if not preview:
        metadata_base = get_metadata(scene_edit_dir_base / "BigHorn_Base_Edit")
        metadata_overlay = get_metadata(scene_edit_dir_overlay / "BigHorn_Fire_Edit")
        assert (
            len(list((scene_edit_dir_base / "BigHorn_Base_Edit").glob("*.png")))
            == png_count
        )
        assert (
            len(list((scene_edit_dir_overlay / "BigHorn_Fire_Edit").glob("*.tif")))
            == tiff_count
        )
        assert check_metadata("size", metadata_base) == (w, h)
        assert check_metadata("size", metadata_overlay) == (w, h)

    # Reproject images
    # original projection: Geostationary
    crs = "EPSG:4326"
    editor.reproject(rasterio.crs.CRS.from_epsg(4326), preview=preview)
    oleditor.reproject(rasterio.crs.CRS.from_epsg(4326), preview=preview)
    if not preview:
        metadata_base = get_metadata(scene_edit_dir_base / "BigHorn_Base_Edit")
        metadata_overlay = get_metadata(scene_edit_dir_overlay / "BigHorn_Fire_Edit")
        assert (
            len(list((scene_edit_dir_base / "BigHorn_Base_Edit").glob("*.png")))
            == png_count
        )
        assert (
            len(list((scene_edit_dir_overlay / "BigHorn_Fire_Edit").glob("*.tif")))
            == tiff_count
        )
        assert check_metadata("crs", metadata_base)[0] == crs
        assert check_metadata("crs", metadata_overlay)[0] == crs

    # Add timestamps
    editor.add_timestamps(
        "upper-left", pytz.timezone("US/Mountain"), "MDT", preview=preview
    )
    if not preview:
        assert (
            len(list((scene_edit_dir_base / "BigHorn_Base_Edit").glob("*.png")))
            == png_count
        )
        assert (
            len(list((scene_edit_dir_overlay / "BigHorn_Fire_Edit").glob("*.tif")))
            == tiff_count
        )

    # Annotate images
    editor.add_annotation(
        "text",
        ("45d04m17s N", "106d48m40s W"),
        label="MT",
        labelopts={"fontsize": 20, "rotation": 0},
    )
    editor.add_annotation(
        "text",
        ("44d57m36s N", "106d48m40s W"),
        label="WY",
        labelopts={"fontsize": 20, "rotation": 0},
    )

    circleopts = {"radius": 7, "fill": (0, 0, 255), "outline": (0, 0, 0), "width": 1}

    editor.add_annotation(
        "circle",
        ("44d21m56s N", "107d10m28s W"),
        label="3504m elev.",
        labelopts={"fontsize": 15, "padding": (15, 15)},
        **circleopts,
    )

    editor.process_annotations(preview=preview)
    if not preview:
        assert (
            len(list((scene_edit_dir_base / "BigHorn_Base_Edit").glob("*.png")))
            == png_count
        )
        assert (
            len(list((scene_edit_dir_overlay / "BigHorn_Fire_Edit").glob("*.tif")))
            == tiff_count
        )

    # Create overlay
    editor.add_overlay(
        oleditor, overlay_session_name="Big_Horn_Overlay", preview=preview
    )
    overlay_path = base_dir / "SceneEdits" / "BigHorn" / "Big_Horn_Overlay"
    assert len(list(overlay_path.glob("*.png"))) == png_count
    if preview:
        for f in overlay_path.glob("*.png"):
            img = Image.open(f)
            img.show()
            time.sleep(0.3)

    # Create video
    neweditor = GoesSceneEditor(
        str(base_dir), scene_folder="BigHorn", session_name="Big_Horn_Overlay"
    )
    neweditor.to_video(codec="mpeg4", fps=20)
    vid_path = base_dir / "Videos" / "BigHorn" / "Big_Horn_Overlay" / "video.mp4"
    assert os.path.exists(str(vid_path))

    # Cleanup
    if delete_flag:
        try:
            shutil.rmtree(base_dir)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(pytest.main())
