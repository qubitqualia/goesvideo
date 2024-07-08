import os
import tempfile
from datetime import datetime
import pytz
from pathlib import Path
from shapely.geometry import box
from rasterio.coords import BoundingBox
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import transform
from pathlib import Path
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from colorama import Fore
try:
    from osgeo import gdal
except ImportError:
    print(f"{Fore.RED} WARNING! Could not import osgeo package. The resize_geotiff function will be unusable.")


def get_crs(file):
    with rasterio.open(file, 'r') as ds:
        crs = ds.crs
    return crs


def transform_bbox(bbox, src_crs, dst_crs):
    """
    Transform bbox coordinates from source CRS to destination CRS

    @param bbox: (list) lat/lon image bounds [left, bottom, right, top]
    @param src_crs: (CRS object) Source CRS
    @param dst_crs: (CRS object) Destination CRS
    @return: (list) bbox coordinates in Destination CRS
    """

    x0, y0 = bbox[0], bbox[1]
    x1, y1 = bbox[2], bbox[3]

    x0t, y0t = transform(src_crs, dst_crs, [x0], [y0])
    x1t, y1t = transform(src_crs, dst_crs, [x1], [y1])

    bbox = [x0t[0], y0t[0], x1t[0], y1t[0]]

    return bbox


def image_to_geotiff(img, bbox, outcrs, outpath=None):
    """
    Converts a simple image (e.g. '*.png') to geotiff

    @param img: (PIL Image) RGB or RGBA Image object to convert
    @param bbox: (list) lat/lon image bounds [left, bottom, right, top]
    @param outcrs: (CRS object) CRS of output image
    @param outpath: (str) save path for output image
    @return: (str) save path for output image
    """
    # Get image size
    width, height = img.size

    # Convert image to array
    imgarr = np.array(img)
    r = imgarr[:, :, 0]
    g = imgarr[:, :, 1]
    b = imgarr[:, :, 2]

    # Convert bbox from lat/lon to outcrs
    bboxout = transform_bbox(bbox, 'EPSG:4326', outcrs)

    # Create new raster
    _transform = from_bounds(*bboxout, width, height)

    if img.mode == 'RGB':
        count = 3
        imgbands = [r, g, b]
    else:
        count = 4
        a = imgarr[:, :, 3]
        imgbands = [r, g, b, a]

    kwargs = {}
    kwargs.update({'count': count,
                   'crs': outcrs,
                   'transform': _transform,
                   'width': width,
                   'height': height,
                   'dtype': np.uint8})

    if not outpath:
        tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        _file = tmp_file.name
    else:
        _file = outpath

    with rasterio.open(_file, 'w', driver='GTiff', **kwargs) as dst:
        for band, src in enumerate(imgbands, start=1):
            dst.write_band(band, src)

    return _file


def resize_geotiff(file, width, height, resample=False, delete=True):
    if isinstance(file, str):
        _path = Path(file)
    else:
        _path = file

    outpath = _path.parent / (_path.stem + '_resize' + _path.suffix)
    inds = gdal.OpenEx(str(_path))
    if resample:
        outds = gdal.Translate(str(outpath), inds, width=width, height=height,
                               resampleAlg=gdal.GRA_NearestNeighbour)
    else:
        outds = gdal.Translate(str(outpath), inds, width=width, height=height)

    del inds
    del outds
    if delete:
        os.remove(str(_path))
        os.rename(str(outpath), str(_path))
    return


def resize_image(file, width, height, resample=False):
    outpath = file.parent / (file.stem + '_tmp' + file.suffix)
    img = Image.open(str(file))
    if resample:
        img = img.resize((width, height), resample=Image.NEAREST)
    else:
        img = img.resize((width, height))
    img.save(outpath)
    img.close()
    os.remove(str(file))
    os.rename(str(outpath), str(file))
    return


def geotiff_to_image(srctif):
    ds = rasterio.open(str(srctif), 'r')
    bands = ds.indexes
    data = ds.read(bands)
    _transform = rasterio.transform.from_bounds(0, 0, 0, 0, data.shape[2], data.shape[1])
    crs = get_crs(srctif)

    dstimg = srctif.parent / (srctif.stem + '.png')
    with rasterio.open(dstimg, 'w', driver='PNG', width=data.shape[2],
                       height=data.shape[1], count=len(bands), dtype=data.dtype, nodata=0,
                       transform=_transform, crs=crs, format='PNG') as dst:
        dst.write(data, indexes=bands)

    img = Image.open(dstimg).convert('RGBA')
    return img


def get_shape_geotiff(file):

    with rasterio.open(str(file)) as src:
        src_x_pixels = src.shape[1]
        src_y_pixels = src.shape[0]

    return src_x_pixels, src_y_pixels


def get_shape_img(self, file):

    img = Image.open(str(file))
    w, h = img.size
    img.close()

    return w, h


def reproject(file, out_crs='EPSG:4326'):

    with rasterio.open(str(file)) as src:

        if isinstance(out_crs, str):
            _transform, width, height = calculate_default_transform(src.crs, {'init': out_crs}, src.width, src.height,
                                                                 *src.bounds)
        else:
            _transform, width, height = calculate_default_transform(src.crs, out_crs, src.width,
                                                                 src.height,
                                                                 *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({'crs': out_crs,
                       'transform': _transform,
                       'width': width,
                       'height': height})

        _file = file.parent / (file.stem + '_tmp' + file.suffix)

        with rasterio.open(str(_file), 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i),
                          destination=rasterio.band(dst, i),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=transform,
                          dst_crs=out_crs,
                          resampling=Resampling.nearest)

    os.remove(file)
    os.rename(_file, file)
    return


def get_bounds(file):
    with rasterio.open(str(file)) as src:
        bounds = src.bounds
    return bounds


def crop_geotiff(file, bbox, height, width, opacity=1):
    svname = file.parent / (file.stem + '_crop' + file.stem)
    skip = False

    # src = rasterio.open(str(file))
    with rasterio.open(str(file)) as src:
        if src.bounds == bbox and src.height == height and src.width == width:
            try:
                alphachannel = src.read()[3, :, :]
                _opacity = round(np.max(alphachannel) / 255, 2)
                if _opacity != opacity:
                    skip = False
                else:
                    skip = True
            except IndexError:
                skip = False
        else:
            skip = False

        if not skip:
            newbounds = bbox
            newwidth = width
            newheight = height
            dsttransform = rasterio.transform.from_bounds(*newbounds, newwidth, newheight)
            kwargs = src.meta.copy()
            kwargs.update({'crs': src.crs,
                           'transform': dsttransform,
                           'width': newwidth,
                           'height': newheight})

            with rasterio.open(str(svname), 'w', **kwargs) as final:
                for i in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, i),
                              destination=rasterio.band(final, i),
                              src_tansform=src.transform,
                              dst_transform=dsttransform,
                              dst_crs=src.crs,
                              resampling=Resampling.nearest)

    if not skip:
        os.remove(str(file))
        os.rename(str(svname), str(file))
    return


def adjust_alpha(img, alpha):
    imgarr = np.array(img, dtype=np.uint8)
    np.place(imgarr[:, :, 3], imgarr[:, :, 3] > 0, int(alpha * 255))
    retimg = Image.fromarray(imgarr, 'RGBA')

    return retimg


def trim_img_border(img):
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

    retimg = Image.fromarray(newarr, 'RGB')
    return retimg


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
    if tformatdict['Y'][0] is not None and tformatdict['Y'][1] is not None:
        year = int(file[tformatdict['Y'][0]:tformatdict['Y'][1]])
    else:
        year = None
    if tformatdict['M'][0] is not None and tformatdict['M'][1] is not None:
        month = int(file[tformatdict['M'][0]:tformatdict['M'][1]])
    else:
        month = None
    if tformatdict['D'][0] is not None and tformatdict['D'][1] is not None:
        day = int(file[tformatdict['D'][0]:tformatdict['D'][1]])
    else:
        day = None
    if tformatdict['h'][0] and tformatdict['h'][1]:
        kwargs['hour'] = int(file[tformatdict['h'][0]:tformatdict['h'][1]])
    if tformatdict['m'][0] and tformatdict['m'][1]:
        kwargs['minute'] = int(file[tformatdict['m'][0]:tformatdict['m'][1]])
    if tformatdict['s'][0] and tformatdict['s'][1]:
        kwargs['second'] = int(file[tformatdict['s'][0]:tformatdict['s'][1]])

    if not year or not month or not day:
        print(f'{Fore.RED} ERROR: Could not extract time from provided filename {file}. Check time format string.')
        t = None

    t = tz.localize(datetime(year, month, day, **kwargs))

    return t


def time_format_str(instr):
    if instr:
        chars = ['Y', 'M', 'D', 'h', 'm', 's']
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
    if isinstance(file, str):
        tstr = file.split('\\')[-1].split('.')[0]
        svpath = '\\'.join(file.split('\\')[0:-1])
        svpath = svpath.rstrip('\\') + '\\'
    elif isinstance(file, Path):
        tstr = file.stem
        svpath = file.parent
    else:
        print(f'{Fore.RED} ERROR: File path must be provided as a string or Path object')
        return

    if not isinstance(in_tz, pytz.BaseTzInfo):
        print(f'{Fore.RED} ERROR: Timezone must be provided as a pytz timezone object.')
        return
    if not isinstance(out_tz, pytz.BaseTzInfo):
        print(f'{Fore.RED} ERROR: Timezone must be provided as a pytz timezone object.')
        return

    _s1 = tstr.split(' ')[0]
    _s2 = tstr.split(' ')[1]
    year = int(_s1.split('-')[0])
    month = int(_s1.split('-')[1])
    day = int(_s1.split('-')[2])
    hr = int(_s2.split('_')[0])
    _min = int(_s2.split('_')[1])
    sec = int(_s2.split('_')[2])

    t = datetime(year, month, day, hour=hr, minute=_min, second=sec)
    t2 = in_tz.localize(t)
    t3 = t2.astimezone(out_tz)
    newtstr = t3.isoformat()
    newtstr = newtstr.replace('T', ' ').replace(':', '_')
    newtstr = newtstr[0:-6] + ' ' + out_abbr

    if isinstance(file, str):
        fname = svpath + newtstr
    else:
        fname = svpath / newtstr

    os.rename(file, fname)
    return


def get_bbox_intersection(user_bbox, img_bbox):
    if user_bbox and img_bbox:
        user_bbox_adj = box(user_bbox[0] + 180, user_bbox[1] + 90, user_bbox[2] + 180, user_bbox[3] + 90)
        img_bbox_adj = box(img_bbox[0] + 180, img_bbox[1] + 90, img_bbox[2] + 180, img_bbox[3] + 90)
        intersection = img_bbox_adj.intersection(user_bbox_adj)

        if intersection.is_empty:
            ret = None
        else:
            _bounds = intersection.bounds
            ret = BoundingBox(_bounds[0] - 180, _bounds[1] - 90, _bounds[2] - 180, _bounds[3] - 90)
    else:
        ret = None

    return ret


def check_bbox(requested, actual):
    lon_ok = False
    lat_ok = False

    if actual[0] <= requested[0] <= actual[2]:
        if (requested[0] < requested[2]) and (actual[0] <= requested[2] <= actual[2]):
            lon_ok = True

    if actual[1] <= requested[1] <= actual[3]:
        if (requested[2] < requested[3]) and (actual[1] <= requested[3] <= actual[3]):
            lat_ok = True

    return lat_ok, lon_ok
