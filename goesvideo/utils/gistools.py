import copy
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import rasterio
import rasterio.crs
import rasterio.warp as rwarp
import rasterio.windows as rwindows
from PIL import Image
from colorama import Fore
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import box

"""
Functions for manipulating tif and png georeferenced images
"""

# ------ Coordinate calculations and conversions


def convert_dms_to_dd(coords):
    """
    Convert Degrees-Minutes-Seconds coords to decimal degrees
    :param coords: (tup) (lat, lon) e.g. ('42d30m22s N',
    :param lon: (str) e.g. '87d15m50s W'
    :return: (tup) (lat, lon)
    """
    dd_ret_list = []
    for coord in coords:
        deg = int(coord.split("d")[0])
        minutes = int((coord.split("d")[1]).split("m")[0])
        secs = int((coord.split("m")[1]).split("s")[0])
        direction = coord[-1]
        dd = deg + (minutes / 60.0) + (secs / 3600)
        if direction == "W" or direction == "S":
            dd = -dd
        dd_ret_list.append(dd)

    dd_ret = tuple(dd_ret_list)

    return dd_ret


def convert_point(point, point_crs, geodata=None):
    geodata_copy = copy.deepcopy(geodata)
    dst_crs = geodata_copy["crs"]

    pointx, pointy = rwarp.transform(point_crs, dst_crs, [point[1]], [point[0]])

    return pointy[0], pointx[0]


def xy_from_latlon(point, geodata=None):
    """
    Transform lat/lon coordinates to pixel coordinates
    :param point: (tup) (lat, lon)
    :param geodata: (dict) {'bbox': (tup) (west, south, east, north) in image crs,
                            'crs': (rasterio.crs.CRS) crs of image proj
                            'raster_profile': (dict) geotiff profile of image
                            }
    :return: (tup) x,y pixel coords
    """
    geodata_copy = copy.deepcopy(geodata)
    crs = geodata_copy["crs"]

    # Convert point from EPSG:4326 to crs
    _transform = rasterio.Affine(*geodata_copy["transform"])
    # _transform = from_bounds(*bbox, width, height)
    pointx, pointy = rwarp.transform(
        rasterio.CRS.from_epsg(4326), crs, [point[1]], [point[0]]
    )

    # Get pixel coordinates
    x, y = rowcol(_transform, [pointx[0]], [pointy[0]])

    return int(y), int(x)


def get_crs(srctif):
    with rasterio.open(srctif, "r") as ds:
        crs = ds.crs
    return crs


def get_resolution(srctif):
    with rasterio.open(srctif, "r") as ds:
        _transform = ds.transform
        res_x = _transform[0]
        res_y = -_transform[4]

    return res_x, res_y


def transform_bbox(bbox, src_crs, dst_crs):
    """
    Transform bbox coordinates from source CRS to destination CRS

    @param bbox: (list) lat/lon image bounds [west, south, east, north]
    @param src_crs: (CRS object) Source CRS
    @param dst_crs: (CRS object) Destination CRS
    @return: (list) bbox coordinates in Destination CRS
    """

    x0, y0 = bbox[0], bbox[1]
    x1, y1 = bbox[2], bbox[3]

    x0t, y0t = rwarp.transform(src_crs, dst_crs, [x0], [y0])
    x1t, y1t = rwarp.transform(src_crs, dst_crs, [x1], [y1])

    bbox = [x0t[0], y0t[0], x1t[0], y1t[0]]

    return bbox


def get_max_bbox(base_bbox, overlaybbox):
    """
    Returns maximum common bounds of base and overlay sets
    @param: (list) bbox of base set
    @param: (list) bbox of overlay set
    @return: (list) maximum common bbox
    """
    _base = base_bbox
    base_box_adj = [[_base[0] + 180, _base[1] + 90, _base[2] + 180, _base[3] + 90]]
    overlay_box_adj = []
    for bbox in overlaybbox:
        overlay_box_adj = [[bbox[0] + 180, bbox[1] + 90, bbox[2] + 180, bbox[3] + 90]]

    bboxes = base_box_adj + overlay_box_adj
    bbox_polys = []
    for bbox in bboxes:
        bbox_polys.append(box(*bbox))

    _maxbbox = bbox_polys[0]
    for rect in bbox_polys:
        _maxbbox = _maxbbox.intersection(rect)

    maxbbox = _maxbbox.bounds
    maxbbox = [maxbbox[0] - 180, maxbbox[1] - 90, maxbbox[2] - 180, maxbbox[3] - 90]

    ret = BoundingBox(*maxbbox)
    return ret


def get_bbox_intersection(user_bbox, img_bbox):
    """
    Returns the intersection of bbox areas
    @param user_bbox: (list) user-provided bbox
    @param img_bbox: (list) image bbox
    @return: (list) bbox intersection
    """
    if user_bbox and img_bbox:
        user_bbox_adj = box(
            user_bbox[0] + 180, user_bbox[1] + 90, user_bbox[2] + 180, user_bbox[3] + 90
        )
        img_bbox_adj = box(
            img_bbox[0] + 180, img_bbox[1] + 90, img_bbox[2] + 180, img_bbox[3] + 90
        )
        intersection = img_bbox_adj.intersection(user_bbox_adj)

        if intersection.is_empty:
            ret = None
        else:
            _bounds = intersection.bounds
            ret = BoundingBox(
                _bounds[0] - 180, _bounds[1] - 90, _bounds[2] - 180, _bounds[3] - 90
            )
    else:
        ret = None

    return ret


def check_bbox(requested, actual, degtol=0, strict=False):
    """
    Check the user-requested bbox is valid. Assumes EPSG:4326 coordinates.
    @param requested: (list) user-requested bbox (west, south, east, north)
    @param actual: (list) image bbox (west, south, east, north)
    @param degtol: (float) allowed tolerance around coordinate in degrees
    @param strict: (bool) if true, check only that requested bbox is inside the actual bbox
    @return: (tup) tuple of bools
    """
    lon_ok = False
    lat_ok = False
    west_ok = False
    south_ok = False
    east_ok = False
    north_ok = False

    requested = [
        requested[0] + 180,
        requested[1] + 90,
        requested[2] + 180,
        requested[3] + 90,
    ]
    actual = [actual[0] + 180, actual[1] + 90, actual[2] + 180, actual[3] + 90]

    if not strict:
        if (actual[0] - degtol) <= requested[0] <= (actual[0] + degtol):
            west_ok = True
        if (actual[1] - degtol) <= requested[1] <= (actual[1] + degtol):
            south_ok = True
        if (actual[2] - degtol) <= requested[2] <= (actual[2] + degtol):
            east_ok = True
        if (actual[3] - degtol) <= requested[3] <= (requested[3] + degtol):
            north_ok = True
    else:
        if actual[0] <= requested[0]:
            west_ok = True
        if actual[1] <= requested[1]:
            south_ok = True
        if actual[2] >= requested[2]:
            east_ok = True
        if actual[3] >= requested[3]:
            north_ok = True

    if west_ok and east_ok:
        lon_ok = True
    if south_ok and north_ok:
        lat_ok = True

    return lat_ok, lon_ok


# ------ Image operations
def image_to_geotiff(img, geodata, outfile=None):
    """
    Converts a simple image (e.g. '*.png') to geotiff

    @param pngimg: (str) or (Path) path to png image to convert OR
                   (PIL.Image) source image object
    @param geodata: (dict) {'bbox': (tup) (west, south, east, north) in image crs,
                            'crs': (rasterio.crs.CRS) crs of image proj
                            'raster_profile': (dict) geotiff profile of image
                            }
    @param outfile: (str) save path for output image; if not provided a temporary file
                    will be created
    @return: (str) save path for output image
    """
    if isinstance(img, str):
        _path = img
    elif isinstance(img, Image.Image):
        _path = None
    else:
        _path = str(img)

    geodata_copy = copy.deepcopy(geodata)

    if _path:
        pngimg = Image.open(_path)
    else:
        pngimg = img

    # Convert image to array
    imgarr = np.array(pngimg)
    r = imgarr[:, :, 0]
    g = imgarr[:, :, 1]
    b = imgarr[:, :, 2]

    # Create new raster
    profile = geodata_copy["raster_profile"]
    width = profile["width"]
    height = profile["height"]
    dtype = profile["dtype"]
    if not geodata_copy["transform"]:
        _transform = from_bounds(*geodata_copy["bbox"], width, height)
    else:
        _transform = rasterio.Affine(*geodata_copy["transform"])

    if pngimg.mode == "RGB":
        count = 3
        imgbands = [r, g, b]
    else:
        count = 4
        a = imgarr[:, :, 3]
        imgbands = [r, g, b, a]

    kwargs = {}
    kwargs.update(
        {
            "transform": _transform,
            "width": width,
            "height": height,
            "crs": geodata_copy["crs"],
            "bounds": geodata_copy["bbox"],
            "count": count,
            "dtype": dtype,
        }
    )

    tmp_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False, suffix=".tif")
    _file = tmp_file.name

    with rasterio.open(_file, "w", **kwargs) as dst:
        for band, src in enumerate(imgbands, start=1):
            dst.write_band(band, src)

    if outfile:
        os.replace(_file, outfile)
        return outfile
    else:
        return _file


def geotiff_to_image(srctif, outfile=None):
    """
    Convert a geotiff to a simple image
    @param srctif: (str) path to source tif
    @param outfile: (str) save path for output image; if not provided then image will
                    not be saved
    @return: (PIL.Image obj) output image
    """
    ds = rasterio.open(str(srctif), "r")
    bands = ds.indexes
    data = ds.read(bands)
    _transform = rasterio.transform.from_bounds(
        0, 0, 0, 0, data.shape[2], data.shape[1]
    )
    crs = get_crs(srctif)

    tmpfile = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False)
    _file = tmpfile.name
    tmpfile.close()

    with rasterio.open(
        _file,
        "w",
        driver="PNG",
        width=data.shape[2],
        height=data.shape[1],
        count=len(bands),
        dtype=data.dtype,
        nodata=0,
        transform=_transform,
        crs=crs,
        format="PNG",
    ) as dst:
        dst.write(data, indexes=bands)

    img = Image.open(_file).convert("RGBA")
    if outfile:
        img.save(outfile)

    try:
        os.unlink(_file)
    except PermissionError:
        pass

    return img


def update_geodata(srctif, geodata):
    with rasterio.open(srctif, "r") as ds:
        profile = ds.profile
        crs = profile.pop("crs", None)
        _t = profile.pop("transform", None)
        geodata.update(
            {
                "raster_profile": profile.data,
                "crs": crs.wkt,
                "transform": [_t.a, _t.b, _t.c, _t.d, _t.e, _t.f],
                "bbox": ds.bounds,
                "res_x": ds.res[0],
                "res_y": ds.res[1],
            }
        )


def resize(srcpath, width, height, geodata=None, resample=None, outfile=None):
    """
    Resize geotiff
    @param srcpath: (str) or (Path) path to png or geotiff
    @param width: (int) desired output width
    @param height: (int) desired output height
    @param geodata: (dict) {'bbox': (tup) (west, south, east, north) in image crs,
                            'crs': (rasterio.crs.CRS) crs of image proj
                            'raster_profile': (dict) geotiff profile of image
                            }
    @param resample: (rasterio.enums.Resampling) default is Resampling.nearest
    @param delete: (bool) delete input file and replace with resized output file
    @param outfile: (str) save path for output image; if not provided and delete is False
                    then a temporary file will be created
    @return: (tup) ((str) save path for output image, (dict) updated geodata [for png only])
    """
    # Accept Path obj for srcpath argument
    if isinstance(srcpath, str):
        _path = srcpath
    else:
        _path = str(srcpath)

    # Determine suffix of srcpath
    if _path.split(".")[-1] == "png":
        is_gtiff = False
        converted = False
        geodata_copy = copy.deepcopy(geodata)
        profile_out = geodata_copy["raster_profile"]
    else:
        is_gtiff = True
        converted = False

    if not resample:
        resample = Resampling.nearest

    # Create tempfiles for writing
    tmpfile = tempfile.NamedTemporaryFile("w+b", suffix=".tif", delete=False)
    _file = tmpfile.name
    tmpfile.close()

    tmpfile2 = tempfile.NamedTemporaryFile("w+b", suffix=".tif", delete=False)
    _file2 = tmpfile2.name
    tmpfile2.close()

    while True:
        if is_gtiff or converted:
            with rasterio.open(_path, "r") as ds:
                data = ds.read(
                    out_shape=(ds.count, int(height), int(width)), resampling=resample
                )
                _transform = ds.transform * ds.transform.scale(
                    (ds.width / data.shape[-1]), (ds.height / data.shape[-2])
                )
                profile = ds.profile

                if not is_gtiff:
                    profile = profile | profile_out
                    profile_out.update({"height": height, "width": width})

                profile.update(
                    {
                        "count": ds.count,
                        "height": height,
                        "width": width,
                        "transform": _transform,
                    }
                )

                with rasterio.open(_file2, "w", **profile) as dst:
                    dst.write(data)

            if not is_gtiff:
                res_x, res_y = get_resolution(_file2)
                geodata_copy.update({"res_x": res_x, "res_y": res_y})
                geodata_copy.update(
                    {
                        "transform": [
                            _transform.a,
                            _transform.b,
                            _transform.c,
                            _transform.d,
                            _transform.e,
                            _transform.f,
                        ]
                    }
                )
            try:
                os.unlink(_file)
            except PermissionError:
                pass
            break
        else:
            _path = image_to_geotiff(_path, geodata_copy)
            converted = True

    if not is_gtiff:
        if outfile:
            geotiff_to_image(_file2, outfile)
            try:
                os.unlink(_file2)
                os.unlink(_path)
            except PermissionError:
                pass
            retpath = outfile
        else:
            outfilepath = tempfile.NamedTemporaryFile(
                "w+b", suffix=".png", delete=False
            )
            outfilepath.close()
            retpath = outfilepath.name
            geotiff_to_image(_file2, retpath)
            try:
                os.unlink(_file2)
                os.unlink(_path)
            except PermissionError:
                pass
    else:
        if outfile:
            os.replace(_file2, outfile)
            retpath = outfile
        else:
            retpath = _file2

    if not is_gtiff:
        return retpath, geodata_copy
    else:
        return retpath, None


def get_shape(srcpath, geodata=None):
    """
    Get the shape in pixels of a geotiff
    @param srcpath: (str) or (Path) path to png or geotiff image file
    @return: (tup) width in pixels, height in pixels
    """
    if isinstance(srcpath, str):
        _path = srcpath
    else:
        _path = str(srcpath)

    if srcpath.split(".")[-1] == "png":
        is_gtiff = False
        converted = False
        geodata_copy = copy.deepcopy(geodata)
    else:
        is_gtiff = True
        converted = False

    while True:
        if is_gtiff and converted:
            with rasterio.open(_path) as src:
                w = src.shape[1]
                h = src.shape[0]
            break
        else:
            _path = image_to_geotiff(_path, geodata_copy)
            converted = True

    return w, h


def reproject(srcpath, out_crs="EPSG:4326", geodata=None, outfile=None):
    """
    Reproject a geotiff or png to specified CRS.
    @param srcpath: (str) or (Path) path to png or geotiff image
    @param out_crs: (str) or (rasterio.CRS obj) desired output CRS of image
    @param geodata: (dict) {'bbox': (tup) (west, south, east, north) in image crs,
                            'crs': (rasterio.crs.CRS) crs of image proj
                            'raster_profile': (dict) geotiff profile of image
                            }
    @param outfile: (str) output .tif file name, if not given then a temporary file
                          will be created
    @return: (tup) ((str) save path for output image, (dict) updated geodata [for png only])
    """
    if isinstance(srcpath, str):
        _path = srcpath
    else:
        _path = str(srcpath)

    if _path.split(".")[-1] == "png":
        is_gtiff = False
        converted = False
        geodata_copy = copy.deepcopy(geodata)
        profile_out = geodata_copy["raster_profile"]
    else:
        is_gtiff = True
        converted = False

    tmpgtifffile = tempfile.NamedTemporaryFile("w+b", suffix=".tif", delete=False)
    gtiff_file = tmpgtifffile.name

    while True:
        if is_gtiff or converted:
            with rasterio.open(_path) as src:
                if isinstance(out_crs, str):
                    out_crs = rasterio.CRS.from_string(out_crs)
                else:
                    out_crs = out_crs

                _transform, width, height = rwarp.calculate_default_transform(
                    src.crs, out_crs, src.width, src.height, *src.bounds
                )

                if not is_gtiff:
                    bbox_out = transform_bbox(
                        geodata_copy["bbox"], geodata_copy["crs"], out_crs
                    )
                    # old_width = geodata_copy['raster_profile']['width']
                    # old_height = geodata_copy['raster_profile']['height']
                    # if old_width != width or old_height != height:
                    #    src2_path, _ = resize(_path, old_width, old_height, geodata=geodata_copy)
                    # else:
                    #    src2_path = None
                else:
                    bbox_out = transform_bbox(src.bounds, src.crs, out_crs)
                    # if width != src.width or height != src.height:
                    #    src2_path, _ = resize(_path, src.width, src.height)
                    # else:
                    #    src2_path = None

                profile = src.profile

                if not is_gtiff:
                    geodata_copy.update(
                        {
                            "crs": out_crs,
                            "bbox": bbox_out,
                            "transform": [
                                _transform.a,
                                _transform.b,
                                _transform.c,
                                _transform.d,
                                _transform.e,
                                _transform.f,
                            ],
                        }
                    )
                    profile_out.update({"width": width, "height": height})

                profile.update({"transform": _transform})
                profile.update(
                    {
                        "crs": out_crs,
                        "bounds": bbox_out,
                        "count": src.count,
                        "width": width,
                        "height": height,
                    }
                )

                src2_path = gtiff_file

                with rasterio.open(src2_path, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        rwarp.reproject(
                            rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=_transform,
                            dst_crs=out_crs,
                            resampling=Resampling.nearest,
                        )

                if not is_gtiff:
                    res_x, res_y = get_resolution(src2_path)
                    geodata_copy.update({"res_x": res_x, "res_y": res_y})
            break
        else:
            _path = image_to_geotiff(_path, geodata_copy)
            converted = True

    tmpgtifffile.close()
    if not is_gtiff:
        if outfile:
            geotiff_to_image(src2_path, outfile=outfile)
            try:
                os.unlink(src2_path)
            except PermissionError:
                pass
            retpath = outfile
        else:
            img = geotiff_to_image(src2_path)
            tmpimgfile = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False)
            imgfile = tmpimgfile.name
            img.save(imgfile)
            img.close()
            try:
                os.unlink(src2_path)
            except PermissionError:
                pass
            retpath = imgfile
    else:
        if outfile:
            os.replace(src2_path, outfile)
            retpath = outfile
        else:
            retpath = src2_path

    if converted:
        try:
            os.unlink(_path)
        except PermissionError:
            pass

    if not is_gtiff:
        return retpath, geodata_copy
    else:
        return retpath, None


def get_bounds(srcpath):
    """
    Get the bounds of a geotiff
    @param srcpath: (str) or (Path) path to geotiff
    @return: (list) bbox bounds
    """
    if isinstance(srcpath, str):
        _path = srcpath
    else:
        _path = str(srcpath)

    with rasterio.open(_path) as src:
        bounds = src.bounds

    return bounds


def crop(srcpath, new_bbox, geodata=None, outfile=None):
    """
    Crop a geotiff to specified bbox and also resize, if desired.
    @param srcpath: (str) or (Path) path to png or geotiff image
    @param new_bbox: (list) bbox bounds
    @param geodata: (dict) {'bbox': (tup) (west, south, east, north) in image crs,
                            'crs': (rasterio.crs.CRS) crs of image proj
                            'raster_profile': (dict) geotiff profile of image
                            }
    @param outfile: (str) output .tif file name, if not given then a temporary file
                          will be created
    @return: (tup) ((str) save path for output image, (dict) updated geodata [for png only])
    """
    if isinstance(srcpath, str):
        _path = srcpath
    else:
        _path = str(srcpath)

    if _path.split(".")[-1] == "png":
        is_gtiff = False
        converted = False
        geodata_copy = copy.deepcopy(geodata)
        profile_out = geodata_copy["raster_profile"]
    else:
        is_gtiff = True
        converted = False

    tmpfile = tempfile.NamedTemporaryFile("w+b", suffix=".tif", delete=False)
    gtiff_file = tmpfile.name

    # Need to transform bbox if crs is not EPSG:4326
    if not is_gtiff:
        if geodata_copy["crs"] != "EPSG:4326":
            new_bbox = transform_bbox(new_bbox, "EPSG:4326", geodata_copy["crs"])
    else:
        with rasterio.open(_path, "r") as ds:
            crs = ds.crs
        if crs != "EPSG:4326":
            new_bbox = transform_bbox(new_bbox, "EPSG:4326", crs)

    while True:
        if is_gtiff or converted:
            with rasterio.open(_path, "r") as src:
                bands = len(src.indexes)
                cropped = src.read(
                    bands, window=rwindows.from_bounds(*new_bbox, src.transform)
                )

                profile = src.profile

                if not is_gtiff:
                    profile = profile | profile_out
                    profile_out.update(
                        {"height": cropped.shape[0], "width": cropped.shape[1]}
                    )

                profile.update(
                    {
                        "bounds": new_bbox,
                        "height": cropped.shape[0],
                        "width": cropped.shape[1],
                    }
                )

                with rasterio.open(gtiff_file, "w", **profile) as dst:
                    for band in range(1, bands + 1):
                        dst.write_band(
                            band,
                            src.read(
                                band,
                                window=rwindows.from_bounds(*new_bbox, src.transform),
                            ),
                        )

            if not is_gtiff:
                res_x, res_y = get_resolution(gtiff_file)
                geodata_copy.update({"bbox": new_bbox, "res_x": res_x, "res_y": res_y})
            break
        else:
            _path = image_to_geotiff(_path, geodata_copy)
            converted = True

    tmpfile.close()
    if not is_gtiff:
        if outfile:
            geotiff_to_image(gtiff_file, outfile=outfile)
            try:
                os.unlink(gtiff_file)
            except PermissionError:
                pass
            retpath = outfile
        else:
            img = geotiff_to_image(gtiff_file)
            tmpimgfile = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False)
            imgfile = tmpimgfile.name
            img.save(imgfile)
            img.close()
            try:
                os.unlink(gtiff_file)
            except PermissionError:
                pass
            retpath = imgfile
    else:
        if outfile:
            os.replace(gtiff_file, outfile)
            retpath = outfile
        else:
            retpath = gtiff_file

    if converted:
        try:
            os.unlink(_path)
        except PermissionError:
            pass

    if not is_gtiff:
        return retpath, geodata_copy
    else:
        return retpath, None


def adjust_alpha(srcpath, alpha, outfile=None):
    """
    Adjust alpha of simple image
    @param srcpath: (str) or (Path) path to png or geotiff input image
    @param alpha: (float) desired alpha value
    @param outfile: (str)
    @return: (str) path to modified file in png format
    """
    if isinstance(srcpath, str):
        _path = srcpath
    else:
        _path = str(srcpath)

    if _path.split(".")[-1] == "png":
        is_gtiff = False
        converted = False
        img = Image.open(_path)
    else:
        is_gtiff = True
        converted = False

    tmpfile = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False)
    png_file = tmpfile.name

    while True:
        if not is_gtiff or converted:
            imgarr = np.array(img, dtype=np.uint8)
            np.place(imgarr[:, :, 3], imgarr[:, :, 3] > 0, int(alpha * 255))
            retimg = Image.fromarray(imgarr, "RGBA")
            retimg.save(png_file)
            retimg.close()
            break
        else:
            img = geotiff_to_image(_path)
            converted = True

    tmpfile.close()
    if outfile:
        os.replace(png_file, outfile)
        try:
            os.unlink(png_file)
        except PermissionError:
            pass
        retpath = outfile
    else:
        retpath = png_file

    return retpath


def trim_img_border(img):
    """
    Trim border of simple image
    @param img: (PIL.Image obj) input image
    @return: (PIL.Image obj) trimmed output image
    """
    imgarr = np.array(img, dtype=np.uint8)
    mask = np.array((255 - imgarr[:, :, 3]) / 255, dtype=np.uint8)
    r = np.ma.array(imgarr[:, :, 0], mask=mask)
    b = np.ma.array(imgarr[:, :, 1], mask=mask)
    g = np.ma.array(imgarr[:, :, 2], mask=mask)

    unmasked_slices = np.ma.notmasked_contiguous(r, axis=1)
    top_row_count = 0
    bot_row_count = 0
    for _slice in unmasked_slices:
        if not _slice:
            top_row_count += 1
        else:
            break

    for _slice in unmasked_slices[::-1]:
        if not _slice:
            bot_row_count += 1
        else:
            break

    top_row = top_row_count
    bot_row = imgarr.shape[0] - bot_row_count

    start_cols = []
    stop_cols = []
    for _slice in unmasked_slices:
        if _slice:
            start_cols.append(_slice[0].start)
            stop_cols.append(_slice[0].stop)

    left_col = min(start_cols)
    right_col = max(stop_cols)

    r = r[top_row:bot_row, left_col:right_col]
    b = b[top_row:bot_row, left_col:right_col]
    g = g[top_row:bot_row, left_col:right_col]

    newarr = np.dstack((r, b, g))

    retimg = Image.fromarray(newarr, "RGB")
    return retimg


# ------ Misc functions
def get_timestamp(file, tformatdict, tz=pytz.utc):
    """
    Applies user-provided time format string to filename to extract timestamp.
    The returned datetime also has its timezone set to that provided, or UTC if
    none is provided
    @param file: (str) filename containing datetime string
    @param tformatdict: (dict) indexes returned by _time_format_str
    @param tz: (pytz tz obj) pytz timezone object
    @return: (datetime) tz-aware datetime
    """

    kwargs = {}
    if tformatdict["Y"][0] is not None and tformatdict["Y"][1] is not None:
        year = int(file[tformatdict["Y"][0] : tformatdict["Y"][1]])
    else:
        year = None
    if tformatdict["M"][0] is not None and tformatdict["M"][1] is not None:
        month = int(file[tformatdict["M"][0] : tformatdict["M"][1]])
    else:
        month = None
    if tformatdict["D"][0] is not None and tformatdict["D"][1] is not None:
        day = int(file[tformatdict["D"][0] : tformatdict["D"][1]])
    else:
        day = None
    if tformatdict["h"][0] and tformatdict["h"][1]:
        kwargs["hour"] = int(file[tformatdict["h"][0] : tformatdict["h"][1]])
    if tformatdict["m"][0] and tformatdict["m"][1]:
        kwargs["minute"] = int(file[tformatdict["m"][0] : tformatdict["m"][1]])
    if tformatdict["s"][0] and tformatdict["s"][1]:
        kwargs["second"] = int(file[tformatdict["s"][0] : tformatdict["s"][1]])

    if not year or not month or not day:
        print(
            f"{Fore.RED}WARNING - Could not extract time from provided filename {file}. Check time "
            f"format string."
        )
        t = None

    t = tz.localize(datetime(year, month, day, **kwargs))

    return t


def time_format_str(instr):
    """
    Generates time format dictionary to be used for parsing image filenames
    @param instr: (str) input format string (e.g. "YYYY-MM-dd hh_mm_ss")
    @return: (dict) time format dict
    """
    if instr:
        chars = ["Y", "M", "D", "h", "m", "s"]
        start_idxs = []
        end_idxs = []
        start_done = False
        char_found = False

        for c in chars:
            for i, _s in enumerate(instr):
                if _s == c and not start_done:
                    start_idxs.append(i)
                    start_done = True
                    char_found = True
                elif _s != c and start_done:
                    end_idxs.append(i)
                    start_done = False
                elif _s == c and start_done and i == len(instr) - 1:
                    end_idxs.append(i)
            if not char_found:
                start_idxs.append(None)
                end_idxs.append(None)
            else:
                char_found = False

        retdict = {}
        for i, c in enumerate(chars):
            retdict[c] = (start_idxs[i], end_idxs[i])
    else:
        retdict = None

    return retdict


def replace_file_timezone(file, in_tz, out_tz, out_abbr):
    """
    Renames a file to match desired timezone
    @param file: (str) path to file
    @param in_tz: (pytz tz obj) timezone in input filename
    @param out_tz: (pytz tz obj) timezone in output filename
    @param out_abbr: (str) timezone abbreviation to use in output filename
    @return: None
    """
    if isinstance(file, str):
        tstr = file.split("\\")[-1].split(".")[0]
        svpath = "\\".join(file.split("\\")[0:-1])
        svpath = svpath.rstrip("\\") + "\\"
    elif isinstance(file, Path):
        tstr = file.stem
        svpath = file.parent
    else:
        print(
            f"{Fore.RED}WARNING - File path must be provided as a string or Path object"
        )
        return

    if not isinstance(in_tz, pytz.BaseTzInfo):
        print(
            f"{Fore.RED}WARNING - Timezone must be provided as a pytz timezone object."
        )
        return
    if not isinstance(out_tz, pytz.BaseTzInfo):
        print(
            f"{Fore.RED}WARNING - Timezone must be provided as a pytz timezone object."
        )
        return

    _s1 = tstr.split(" ")[0]
    _s2 = tstr.split(" ")[1]
    year = int(_s1.split("-")[0])
    month = int(_s1.split("-")[1])
    day = int(_s1.split("-")[2])
    hr = int(_s2.split("_")[0])
    _min = int(_s2.split("_")[1])
    sec = int(_s2.split("_")[2])

    t = datetime(year, month, day, hour=hr, minute=_min, second=sec)
    t2 = in_tz.localize(t)
    t3 = t2.astimezone(out_tz)
    newtstr = t3.isoformat()
    newtstr = newtstr.replace("T", " ").replace(":", "_")
    newtstr = newtstr[0:-6] + " " + out_abbr

    if isinstance(file, str):
        fname = svpath + newtstr
    else:
        fname = svpath / newtstr

    os.replace(file, fname)
    return
