import shutil
from importlib.resources import files as importfiles
from pathlib import Path


def generate_tree(
    base_dir, scenename, overlay_scene=None, copy_images=False, copy_ncfiles=False
):
    p = importfiles("goesvideo") / "tests" / "Test Images"
    p2 = importfiles("goesvideo") / "tests" / "Test NC Files"

    png_fnames = list(p.glob("*.png"))
    tiff_fnames = list(p.glob("*.tif"))

    scene_dir_base = Path(base_dir / "Scenes" / scenename)
    scene_edit_dir = Path(base_dir / "SceneEdits")
    scene_edit_dir_base = scene_edit_dir / scenename
    vid_dir = Path(base_dir / "Videos" / scenename)

    if overlay_scene:
        scene_dir_overlay = Path(base_dir / "Scenes" / overlay_scene)
        scene_edit_dir_overlay = scene_edit_dir / overlay_scene

    base_dir.mkdir(parents=True, exist_ok=True)
    scene_dir_base.mkdir(parents=True, exist_ok=True)
    scene_edit_dir.mkdir(parents=True, exist_ok=True)
    scene_edit_dir_base.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    if overlay_scene:
        scene_dir_overlay.mkdir(parents=True, exist_ok=True)
        scene_edit_dir_overlay.mkdir(parents=True, exist_ok=True)

    # Copy image files to temp folders
    if copy_images:
        for f in png_fnames:
            shutil.copyfile(str(f), str(scene_dir_base / f.name))
        if overlay_scene:
            for f in tiff_fnames:
                shutil.copyfile(str(f), str(scene_dir_overlay / f.name))

        # Copy metadata.json and timestamps.csv
        shutil.copyfile(str(p / "metadata.json"), str(scene_dir_base / "metadata.json"))
        shutil.copyfile(
            str(p / "timestamps.csv"), str(scene_dir_base / "timestamps.csv")
        )
        if overlay_scene:
            shutil.copyfile(
                str(p / "metadata.json"), str(scene_dir_overlay / "metadata.json")
            )
            shutil.copyfile(
                str(p / "timestamps_overlay.csv"),
                str(scene_dir_overlay / "timestamps.csv"),
            )

    if copy_ncfiles:
        nc_path = Path(base_dir / "goesdata")
        nc_path.mkdir(parents=True, exist_ok=True)
        nc_fnames = list(p2.glob("*.nc"))
        for f in nc_fnames:
            shutil.copyfile(str(f), str(nc_path / f.name))

    return
