from osgeo import gdal
from pathlib import Path
from PIL import Image
from colorama import init as colorama_init
from colorama import Fore
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
import goesvideo.exceptions
import sys
import numpy as np
import os
import shutil
import copy
from datetime import datetime, timedelta
import pytz

# overlay 1 - [img1, img

class Overlay:

    def __init__(self, baseimgpath, overlaypaths, outpath, out_crs=None, bbox=None):
        """
        @param baseimgpath: (str) path to folder or file to use for base image(s). If a folder is provided
                                  the object instance will be initialized using all available geotiffs or other image
                                  formats in the folder and the base image will be dyanmic. If a single file or
                                  a folder containing a single file is provided, the base image will be static
        @param overlaypaths:
        @param outpath:
        @param out_crs:
        @param bbox: (list) [minx, maxy, maxx, miny] if provided, the base image and overlay images will all be cropped
                            to these limits. If the base image(s) is provided as a simple image (e.g. png, bmp, jpg), then
                            the bbox parameter must be specified. All overlay images will then be cropped accordingly in
                            order to match up with the base image.

        """
        self.bbox = bbox
        self.outpath = Path(outpath)

        # Setup temporary folders
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

        #    Copy overlay files - Overlays can only be geotiff format
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

        # Calculate interval between images in each overlay path
        overlay_intervals = []
        for overlay in self.overlayimgfiles:
            overlay_intervals.append(self._calculate_interval(overlay[0], overlay[1]))

        # Calculate interval between base images, if applicable
        base_interval = 0
        if not self.base_is_static:
            base_interval = self._calculate_interval(self.baseimgfiles[0], self.baseimgfiles[1])
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
                self.base_is_reference = True
            else:
                self.framecount = len(self.overlayimgfiles[-1])
                reffiles = self.overlayimgfiles[-1]
                self.base_skip_factor = 0
        else:
            if not self.base_is_static:
                self.framecount = len(self.overlayimgfiles[mindt_idx])
                reffiles = self.overlayimgfiles[mindt_idx]
                factor = all_intervals[-1] / all_intervals[mindt_idx]
                if factor - int(factor) > 0.5:
                    self.base_skip_factor = int(factor) + 1
                else:
                    self.base_skip_factor = int(factor)
            else:
                self.framecount = len(self.overlayimgfiles[mindt_idx])
                reffiles = self.overlayimgfiles[mindt_idx]
                self.base_skip_factor = 0

        # Fill in overlay file lists with duplicates as needed to match the frame count
        new_overlay_list = []
        for overlay in self.overlayimgfiles:
            tmp_list = []
            if len(overlay) < self.framecount:
                for i, file in enumerate(reffiles):
                    closest_idx = self._get_closest_time(file.stem, [x.stem for x in overlay])
                    tmp_list.append(overlay[closest_idx])
                new_overlay_list.append(tmp_list)
            else:
                new_overlay_list.append(overlay)

        self.overlayimgfiles = new_overlay_list

        # Fill in the base file list with duplicates if needed
        if not self.base_is_static:
            new_base_list = []
            for i, file in self.baseimgfiles:
                closest_idx = self._get_closest_time(file.stem, [x.stem for x in reffiles])
                new_base_list.append(reffiles[closest_idx])
            self.baseimgfiles = new_base_list


        # Default CRS is the CRS of the overlay images
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

    def create_overlays(self, res=None, overlay_opacities=None, overlay_resample=None,
                        base_resample=None, format='png'):

        for i in range(0, self.framecount):

            img_set = []
            img_arr = []
            basedone = False

            for row in self.overlayimgfiles:
                img_set.append(row[i])

            # Generate save name for image
            if self.base_is_reference:
                svname = self.baseimgfiles[i].stem + self.baseimgfiles[i].suffix
            else:
                svname = (self.overlayimgfiles[self.overlay_reference_idx][i].stem +
                          self.overlayimgfiles[self.overlay_reference_idx][i].suffix)

            # Get base image path
            if self.base_is_static:
                baseimgpath = self.baseimgfiles[0]
            else:
                baseimgpath = self.baseimgfiles[i]

            for overlay, file in enumerate(img_set):
                #svnames.append(file.stem + '.png')
                # Crop images to bbox
                if self.bbox:

                    self._crop_geotiff(file, self.bbox_overlay)
                    if overlay == 0:
                        if not basedone:
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

    def _calculate_interval(self, file1, file2):
        """
        Calculate interval in seconds between two files
        @param file1: (str) filename containing an ISO format datetime (e.g. '2023-01-01 10_43_00')
        @param file2: (str)
        @return:
        """
        # Extract datetime from string
        file1 = file1.stem
        file2 = file2.stem

        t1 = self._get_timestamp(file1)
        t2 = self._get_timestamp(file2)

        # Get delta
        delta = abs((t2 - t1).total_seconds())

        return delta

    def _get_timestamp(self, file):
        if '+' not in str(file):
            # Current format of goesvideo geotiff filenames - given in UTC time showing offset (need to change)
            t1 = datetime.fromisoformat(' '.join(str(file).split(' ')[0:2]).replace('_', ':'))
            t1 = t1.replace(tzinfo=pytz.utc)

        else:
            # Current format of nexrad geotiff filenames - given in local time
            t1 = datetime.fromisoformat(' '.join(str(file).split('+')[0:2]).replace('_', ':'))
            t1 = pytz.timezone('Pacific/Honolulu').localize(t1).astimezone(pytz.utc)

        # Convert to UTC
        #1 = t1.astimezone(pytz.utc)

        return t1

    def _get_closest_time(self, target, srcarray):

        target_time = self._get_timestamp(target)
        src_times = [self._get_timestamp(x) for x in srcarray]

        closest_time = min(src_times, key=lambda d: abs(d - target_time))
        idx = src_times.index(closest_time)
        #file = srcarray[idx]

        return idx