from osgeo import gdal
from pathlib import Path
from PIL import Image
from colorama import Fore
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
import sys
import numpy as np
import os
import shutil
import copy
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm
from goesvideo.utils import get_timestamp, time_format_str


class Overlay:

    def __init__(self, baseimgpath, overlaypaths, overlay_timezones, outpath,
                 out_crs=None, bbox=None, base_timezone=None):
        """
        @param baseimgpath: (str) path to folder or file to use for base image(s). If a folder is provided
                                  the object instance will be initialized using all available geotiffs or other image
                                  formats in the folder and the base image will be dyanmic. If a single file or
                                  a folder containing a single file is provided, the base image will be static
        @param overlaypaths: (str) path to folder containing overlay images
        @param outpath: (str) path to save output images
        @param out_crs: (obj) CRS to use for output images
        @param bbox: (list) [minx, maxy, maxx, miny] if provided, the base image and overlay images will all be cropped
                            to these limits. If the base image(s) is provided as a simple image (e.g. png, bmp, jpg), then
                            the bbox parameter must be specified. All overlay images will then be cropped accordingly in
                            order to match up with the base image.
        @param base_timezone: (dict) {'timezone': (pytz tz obj), 'filename format': (str)}
        @param overlay_timezones: (dict) {'timezone': [(pytz tz obj)], 'filename format': [(str)]}

        """
        self.bbox = bbox
        self.outpath = Path(outpath)

        # Setup temporary folders
        print(f'{Fore.GREEN} Initalizing working folders...', end='')
        self.basetmpdir = tempfile.TemporaryDirectory()
        self.basepath = Path(self.basetmpdir.name)
        self.overlaytmpdirs = []
        self.overlaypaths = []
        for p in overlaypaths:
            _path = tempfile.TemporaryDirectory()
            self.overlaytmpdirs.append(_path)
            self.overlaypaths.append(Path(_path.name))

        # Copy files to tmp folders
        #    Copy base file(s) - Base can be geotiff or png
        self.baseimgfiles = []
        baseimgpath = Path(baseimgpath)

        if baseimgpath.is_dir():
            gtiffs = list(baseimgpath.glob('*.tif')) + list(baseimgpath.glob('*.tiff'))
            pngs = list(baseimgpath.glob('*.png'))
        else:
            if baseimgpath.suffix == '.tif' or baseimgpath.suffix == '.tiff':
                gtiffs = [baseimgpath]
                pngs = []
            elif baseimgpath.suffix == '.png':
                gtiffs = []
                pngs = [baseimgpath]
            else:
                print(f'{Fore.RED} ERROR: No valid image formats found. Exiting...')
                sys.exit(0)

        if gtiffs and pngs:
            print(f'{Fore.RED} WARNING: Multiple image formats found in base image folder. Using geotiffs...')
            self.base_is_geotiff = True
            for file in gtiffs:
                fname = self._copy_image(file, self.basepath)
                self.baseimgfiles.append(fname)
        elif gtiffs:
            self.base_is_geotiff = True
            for file in gtiffs:
                fname = self._copy_image(file, self.basepath)
                self.baseimgfiles.append(fname)
        elif pngs:
            self.base_is_geotiff = False
            for file in pngs:
                fname = self._copy_image(file, self.basepath)
                self.baseimgfiles.append(fname)
        else:
            print(f'{Fore.RED} ERROR: No valid image formats found. Exiting...')
            sys.exit(0)

        if len(self.baseimgfiles) == 1:
            self.base_is_static = True
        else:
            self.base_is_static = False


        # Configure timezones and filename formats
        if base_timezone:
            self.base_tz = base_timezone.pop('timezone', pytz.utc)
            base_format = base_timezone.pop('filename format', None)
            self.base_tformat = time_format_str(base_format)
        else:
            self.base_tz = None
            self.base_tformat = None

        self.overlays_tz = overlay_timezones.pop('timezone', None)
        overlays_format = overlay_timezones.pop('filename format', None)

        if not self.overlays_tz:
            self.overlays_tz = [pytz.utc] * len(overlaypaths)

        err = (f"{Fore.RED} ERROR: Timezone and filename format (e.g. 'YYYY-MM-DD hh_mm_ss XXX') must be provided for "
               f" overlay images. Exiting...")
        err2 = (f"{Fore.RED} ERROR: Timezone and filename format (e.g. 'YYYY-MM-DD hh_mm_ss XXX') must be provided for "
               f" base images. Exiting...")

        if not overlays_format:
            print(err)
            sys.exit(0)
        elif len(overlays_format) != len(overlaypaths):
            print(err)
            sys.exit(0)

        if not self.base_is_static and self.base_tformat is None:
            print(err2)
            sys.exit(0)

        self.overlay_tformats = [time_format_str(x) for x in overlays_format]

        # Copy overlay files - Overlays can only be geotiff format
        self.overlayimgfiles = []
        for i, p in enumerate(overlaypaths):
            p = Path(p)
            gtiffs = list(p.glob('*.tif')) + list(p.glob('*.tiff'))
            if not gtiffs:
                print(f'{Fore.RED} ERROR: No geotiffs found in overlay path {str(p)}. Exiting...')
                sys.exit(0)
            row = []
            for file in gtiffs:
                fname = self._copy_image(file, self.overlaypaths[i])
                row.append(fname)
            self.overlayimgfiles.append(row)
        print(f'{Fore.GREEN} Done!')

        # Calculate interval between images in each overlay path
        print(f'{Fore.GREEN} Setting up image arrays...', end='')
        overlay_intervals = []
        for i, overlay in enumerate(self.overlayimgfiles):
            overlay_intervals.append(self._calculate_interval((overlay[0], overlay[1]),
                                                              (self.overlay_tformats[i],),
                                                               (self.overlays_tz[i],)))

        # Calculate interval between base images, if applicable
        base_interval = 0
        if not self.base_is_static:
            base_interval = self._calculate_interval((self.baseimgfiles[0], self.baseimgfiles[1]),
                                                     (self.base_tformat,),
                                                     (self.base_tz,))
            all_intervals = overlay_intervals + [base_interval]
        else:
            all_intervals = overlay_intervals

        mindt_idx = all_intervals.index(min(all_intervals))
        self.base_is_reference = False
        self.overlay_reference_idx = mindt_idx

        if mindt_idx == len(all_intervals) - 1:
            if not self.base_is_static:
                self.framecount = len(self.baseimgfiles)
                reffiles = self.baseimgfiles
                refidx = 0
                self.base_is_reference = True
            else:
                self.framecount = len(self.overlayimgfiles[-1])
                reffiles = self.overlayimgfiles[-1]
                refidx = -1
                #self.base_skip_factor = 0
        else:
            if not self.base_is_static:
                self.framecount = len(self.overlayimgfiles[mindt_idx])
                reffiles = self.overlayimgfiles[mindt_idx]
                refidx = mindt_idx
                #factor = all_intervals[-1] / all_intervals[mindt_idx]
                #if factor - int(factor) > 0.5:
                    #self.base_skip_factor = int(factor) + 1
                #else:
                    #self.base_skip_factor = int(factor)
            else:
                self.framecount = len(self.overlayimgfiles[mindt_idx])
                reffiles = self.overlayimgfiles[mindt_idx]
                refidx = mindt_idx
                #self.base_skip_factor = 0

        # Fill in overlay file lists with duplicates as needed to match the frame count
        new_overlay_list = []
        for j, overlay in enumerate(self.overlayimgfiles):
            tmp_list = []
            if len(overlay) < self.framecount:
                for i, file in enumerate(reffiles):
                    if self.base_is_reference:
                        closest_idx = self._get_closest_time((file.stem, self.base_tformat),
                                                         ([x.stem for x in overlay], self.overlay_tformats[j]),
                                                             tz=(self.base_tz, self.overlays_tz[j]))
                    else:
                        closest_idx = self._get_closest_time((file.stem, self.overlay_tformats[refidx]),
                                                             ([x.stem for x in overlay], self.overlay_tformats[j]),
                                                             tz=(self.overlays_tz[refidx], self.overlays_tz[j]))

                    tmp_list.append(overlay[closest_idx])

                new_overlay_list.append(tmp_list)
            else:
                new_overlay_list.append(overlay)

        self.overlayimgfiles = new_overlay_list

        # Fill in the base file list with duplicates if needed
        if not self.base_is_static:
            if len(self.baseimgfiles) < self.framecount:
                new_base_list = []
                for i, file in self.baseimgfiles:
                    closest_idx = self._get_closest_time((file.stem, self.base_tformat),
                                                         ([x.stem for x in reffiles], self.overlay_tformats[refidx]),
                                                         tz=(self.base_tz, self.overlays_tz[refidx]))
                    new_base_list.append(reffiles[closest_idx])
                self.baseimgfiles = new_base_list

        print(f'{Fore.GREEN} Done!')
        # Default CRS is the CRS of the overlay images
        print(f'{Fore.GREEN} Setting up image projections...', end='')
        if not out_crs:
            self.out_crs = self.get_crs(self.overlayimgfiles[0][0])
        else:
            self.out_crs = out_crs

        # Check input CRS and reproject overlay and base images as necessary
        incrs_overlay = self.get_crs(self.overlayimgfiles[0][0])
        if incrs_overlay != self.out_crs:
            for overlay in self.overlayimgfiles:
                for file in overlay:
                    self._reproject(file, self.out_crs)
            incrs_overlay = self.out_crs

        if self.base_is_geotiff:
            incrs_base = self.get_crs(self.baseimgfiles[0])
            if incrs_base != self.out_crs:
                for file in self.baseimgfiles:
                    self._reproject(file, self.out_crs)
            incrs_base = self.get_crs(self.baseimgfiles[0])
        else:
            incrs_base = ''

        # If input crs is not EPSG:4326, need to convert bbox coords from geographic to output projection
        if bbox:
            if incrs_base == '':
                self.bbox_base = bbox
            elif incrs_base != 'EPSG:4326':
                self.bbox_base = self._transform_bbox(self.bbox, 'EPSG:4326', incrs_base)
            else:
                self.bbox_base = bbox

            if incrs_overlay != 'EPSG:4326':
                self.bbox_overlay = self._transform_bbox(self.bbox, 'EPSG:4326', incrs_overlay)
            else:
                self.bbox_overlay = bbox
        print(f'{Fore.GREEN} Done!')
        print()
        print(f'{Fore.GREEN} Initialization Complete!')
        print()

    def create_overlays(self, res=None, overlay_opacities=None, overlay_resample=None,
                        base_resample=None, save_overlays_gtiff=None):

        if save_overlays_gtiff:
            savepath = Path(save_overlays_gtiff)
            savepath.mkdir(exist_ok=True)
            overlaypaths = []
            for i in range(len(self.overlayimgfiles)):
                op = savepath / f'overlay_{str(i)}'
                op.mkdir(exist_ok=True)
                overlaypaths.append(op)

        for i in tqdm(range(0, self.framecount), desc='Frame Number: ', colour='green'):

            img_set = []
            img_arr = []
            basedone = False

            for row in self.overlayimgfiles:
                img_set.append(row[i])

            # Generate save name for image
            if self.base_is_reference:
                svname = self.baseimgfiles[i].stem + '.png'
            else:
                svname = (self.overlayimgfiles[self.overlay_reference_idx][i].stem + '.png')

            # Get base image path
            if self.base_is_static:
                baseimgpath = self.baseimgfiles[0]
            else:
                baseimgpath = self.baseimgfiles[i]

            overlay = 0

            for overlay, file in enumerate(img_set):
                #svnames.append(file.stem + '.png')
                # Crop images to bbox
                if self.bbox:

                    self._crop_geotiff(file, self.bbox_overlay)
                    if overlay == 0:
                        if basedone:
                            self._crop_geotiff(baseimgpath, self.bbox_base)


                # Resize images
                if not res:
                    # Resize using base image resolution if no res specified
                    if self.base_is_geotiff:
                        w, h = self._get_shape_geotiff(baseimgpath)
                    else:
                        w, h = self._get_shape_img(baseimgpath)
                else:
                    w, h = res[0], res[1]

                self._resize_geotiff(file, w, h, resample=overlay_resample[overlay])

                if overlay == 0:
                    if not basedone:
                        if self.base_is_geotiff:
                            self._resize_geotiff(baseimgpath, w, h, resample=base_resample)
                        else:
                            self._resize_image(baseimgpath, w, h, resample=base_resample)

                if save_overlays_gtiff:
                    fname = overlaypaths[overlay] / (svname.split('.')[0] + '.tif')
                    shutil.copy(file, fname)

                # Convert to images and create overlay
                overlayimg = self._geotiff_to_image(file)
                #overlayimg = self._trim_img_border(overlayimg)
                if overlay == 0:
                    if not basedone:
                        if self.base_is_geotiff:
                            base_img = self._geotiff_to_image(baseimgpath)
                        else:
                            base_img = Image.open(baseimgpath)

                    #base_img = self._trim_img_border(base_img)

                if overlay_opacities:
                    _op = overlay_opacities[overlay]
                    overlayimg = self._adjust_alpha(overlayimg, _op)

                img_arr.append(overlayimg)
                basedone = True

            base_img_copy = copy.deepcopy(base_img)
            for _img in img_arr:
                base_img_copy.alpha_composite(_img)

            base_img_copy.save(self.outpath / svname)

        return


    def get_crs(self, file):
        with rasterio.open(file, 'r') as ds:
            crs = ds.crs
        return crs


    def _copy_image(self, file, dstpath):
        fp = Path(file)
        fname = dstpath / (fp.stem + fp.suffix)
        shutil.copy(file, str(fname))
        return fname

    def _resize_image(self, file, width, height, resample=False):
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

    def _resize_geotiff(self, file, width, height, resample=False):
        outpath = file.parent / (file.stem + '_tmp' + file.suffix)
        inds = gdal.OpenEx(str(file))
        if resample:
            outds = gdal.Translate(str(outpath), inds, width=width, height=height,
                                   resampleAlg=gdal.GRA_NearestNeighbour)
        else:
            outds = gdal.Translate(str(outpath), inds, width=width, height=height)

        del inds
        del outds
        os.remove(str(file))
        os.rename(str(outpath), str(file))
        return

    def _geotiff_to_image(self, srctif):
        ds = rasterio.open(str(srctif), 'r')
        bands = ds.indexes
        data = ds.read(bands)
        transform = rasterio.transform.from_bounds(0, 0, 0, 0, data.shape[2], data.shape[1])
        crs = self.get_crs(srctif)

        dstimg = srctif.parent / (srctif.stem + '.png')
        with rasterio.open(dstimg, 'w', driver='PNG', width=data.shape[2],
                           height=data.shape[1], count=len(bands), dtype=data.dtype, nodata=0,
                           transform=transform, crs=crs, format='PNG') as dst:
            dst.write(data, indexes=bands)

        img = Image.open(dstimg).convert('RGBA')
        return img

    def _get_shape_geotiff(self, file):

        with rasterio.open(str(file)) as src:
            src_x_pixels = src.shape[1]
            src_y_pixels = src.shape[0]

        return src_x_pixels, src_y_pixels

    def _get_shape_img(self, file):

        img = Image.open(str(file))
        w, h = img.size
        img.close()

        return w, h

    def _reproject(self, file, out_crs='EPSG:4326'):

        with rasterio.open(str(file)) as src:

            if isinstance(out_crs, str):
                transform, width, height = calculate_default_transform(src.crs, {'init': out_crs}, src.width, src.height,
                                                                       *src.bounds)
            else:
                transform, width, height = calculate_default_transform(src.crs, out_crs, src.width,
                                                                       src.height,
                                                                       *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({'crs': out_crs,
                           'transform': transform,
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

    def _crop_geotiff(self, file, bbox, opacity=1):
        svname = file.parent / (file.stem + '_crop' + file.stem)

        #src = rasterio.open(str(file))
        with rasterio.open(str(file)) as src:
            left, top = bbox[0], bbox[1]
            right, bottom = bbox[2], bbox[3]

            ul_row, ul_col = src.index(left, top)
            br_row, br_col = src.index(right, bottom)

            indexes = src.indexes
            src_x_pixels = src.shape[1]
            src_y_pixels = src.shape[0]

            outbands = []
            for idx in indexes:
                outarr = np.ones((src_y_pixels, src_x_pixels), dtype=rasterio.uint8)
                src.read(indexes=idx, out=outarr)
                croparr = outarr[ul_row:br_row, ul_col:br_col]
                if idx == 4:
                    croparr = np.where(croparr > 0, int(opacity * 255), croparr)
                #croparr = outarr[ul_row:br_row, ul_col:br_col]
                outbands.append(croparr)

            transform, width, height = calculate_default_transform(src.crs, src.crs,
                                                                   outbands[0].shape[1],
                                                                   outbands[1].shape[0],
                                                                   left=left,
                                                                   bottom=bottom,
                                                                   right=right,
                                                                   top=top)
            kwargs = src.meta.copy()
            kwargs.update({'crs': src.crs,
                           'transform': transform,
                           'width': outbands[0].shape[1],
                           'height': outbands[0].shape[0]})

            with rasterio.open(str(svname), 'w', **kwargs) as final:
                for i in indexes:
                    final.write(outbands[i-1], indexes=i)

        os.remove(str(file))
        os.rename(str(svname), str(file))
        return

    def _transform_bbox(self, bbox, src_crs, dst_crs):

        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]

        x0t, y0t = rasterio.warp.transform(src_crs, dst_crs, [x0], [y0])
        x1t, y1t = rasterio.warp.transform(src_crs, dst_crs, [x1], [y1])

        bbox = [x0t[0], y0t[0], x1t[0], y1t[0]]

        return bbox

    def _adjust_alpha(self, img, alpha):
        imgarr = np.array(img, dtype=np.uint8)
        np.place(imgarr[:, :, 3], imgarr[:, :, 3] > 0, int(alpha * 255))
        retimg = Image.fromarray(imgarr, 'RGBA')

        return retimg

    def _trim_img_border(self, img):
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

    def _calculate_interval(self, files, formatdicts, tz=None):
        """
        Calculate interval in seconds between two files
        @param files: (tup) tuple of filenames containing datetime strings (e.g. '2023-01-01 10_43_00')
        @param formatdicts: (tup) tuple of dicts containing indices returned by _time_format_str
        @param tz: (tup) tuple of pytz timezone objects
        @return: interval between two files in seconds
        """
        # Extract datetime from string
        file1 = files[0].stem
        file2 = files[1].stem

        # Unpack formatdicts argument
        tformat1 = formatdicts[0]
        try:
            tformat2 = formatdicts[1]
        except IndexError:
            tformat2 = tformat1

        # Unpack tz argument
        if tz:
            tz1 = tz[0]
            try:
                tz2 = tz[1]
            except IndexError:
                tz2 = tz1
        else:
            tz1 = pytz.utc
            tz2 = pytz.utc

        t1 = get_timestamp(file1, tformat1, tz=tz1)
        t2 = get_timestamp(file2, tformat2, tz=tz2)

        # Convert times to UTC
        t1 = t1.astimezone(pytz.utc)
        t2 = t2.astimezone(pytz.utc)

        # Get delta
        delta = abs((t2 - t1).total_seconds())

        return delta

    def _get_timestamp(self, file, tformatdict, tz=pytz.utc):
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
            print(f'{Fore.RED} Exiting...')
            sys.exit(0)

        t = tz.localize(datetime(year, month, day, **kwargs))

        return t

    def _get_closest_time(self, target, srcarray, tz=None):

        if tz:
            targettz = tz[0]
            try:
                srctz = tz[1]
            except IndexError:
                srctz = targettz
        else:
            targettz = pytz.utc
            srctz = pytz.utc

        _srcarray = srcarray[0]
        src_tformat = srcarray[1]

        target_time = get_timestamp(target[0], target[1], targettz)
        src_times = [get_timestamp(x, src_tformat, srctz) for x in _srcarray]

        if targettz != srctz:
            target_time = target_time.astimezone(srctz)

        closest_time = min(src_times, key=lambda d: abs(d - target_time))
        idx = src_times.index(closest_time)
        #file = srcarray[idx]

        return idx

    def _get_separator(self, instr, startidx, startchar):
        right_sep = None
        left_sep = None
        if startidx == 0:
            c = 1
            left_sep = None
            while True:
                try:
                    _s = instr[startidx + c]
                except KeyError:
                    right_sep = None
                    break
                if _s != startchar:
                    right_sep = _s
                    break
                c += 1
        else:
            c = 0
            while True:
                try:
                    _s = instr[startidx - c]
                except KeyError:
                    left_sep = None
                    break
                if _s != startchar:
                    left_sep = _s
                    break
                c += 1
            c = 0
            while True:
                try:
                    _s = instr[startidx + c]
                except KeyError:
                    right_sep = None
                    break
                if _s != startchar:
                    right_sep = _s
                    break
                c += 1

        return left_sep, right_sep