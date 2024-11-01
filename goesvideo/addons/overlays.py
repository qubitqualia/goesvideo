import atexit
import copy
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
from PIL import Image
from colorama import Fore
from matplotlib import colormaps as cm
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from goesvideo.utils import gistools, editortools

"""
The GenericOverlay class is intended to produce composite images from a base set of images (typically a 
static/dynamic visible satellite image of an area or a map) and one or more overlay sets 
(e.g. fire detection). The image sets should all cover the same time frame, but they do not require 
exactly matching timestamps. The class will set the time interval between frames to the lowest 
possible value given base and overlay image sets with each set assumed to contain images with a 
constant time interval. The image sets with higher time intervals will be filled with redundant 
frames as required to match the lowest interval.
"""


class GenericOverlay:
    def __init__(
            self,
            baseimgpath,
            overlaypaths,
            overlay_timezones=None,
            base_timezone=None,
            start_time=None,
            end_time=None,
    ):
        """
        @param baseimgpath: (str) path to folder or file to use for base image(s). If a folder is
                                  provided the object instance will be initialized using all
                                  available geotiffs or pngs in the folder and the base image will
                                  be dyanmic. If a single file or a folder containing a single file
                                  is provided, the base image will be static
        @param overlaypaths: (list) paths to folders containing overlay images. Provide one path for
                                    each desired overlay set; *images must be geotiff format*
        @param base_timezone: (dict) {'timezone': (pytz tz obj), 'filename format': (str)} Image
                                     filenames must be provided as datetimes. The format can be
                                     anything as long as it is specified here
                                     (e.g. "YYYY-MM-DD HH:mm:ss"). If parameter is not specified,
                                     UTC will be used for the timezone and the function will attempt
                                     to convert the filenames on its own assuming isoformat strings
                                     with dash and underscore separators ("YYYY-MM-DD hh_mm_ss").
        @param overlay_timezones: (dict) {'timezone': [(pytz tz obj)], 'filename format': [(str)]}
                                         Image filenames must be provided as datetimes. The format
                                         can be anything as long as it is specified here
                                         (e.g. "YYYY-MM-DD HH:mm:ss"). If parameter is not
                                         specified, UTC will be used for the timezone and the
                                         function will attempt to convert the filenames assuming
                                         isoformat strings. Provide one timezone and one format for
                                         each overlay set.
        @param start_time: (str) isoformat datetime start time, by default, timezone is assumed to
                                 be the same as that provided by base_timezone. This behavior can
                                 be overridden by including the offset in the datetime string
                                 (e.g. '2023-01-01 12:00:00-03:00'). If not provided, the default
                                 behavior is to use all available images in specified folders.
        @param end_time: (str) isoformat datetime end time, by default, timezone is assumed to be
                               the same as that provided by base_timezone. This behavior can be
                               overridden by including the offset in the datetime string
                               (e.g. '2023-01-01 12:00:00-03:00'). If not provided, the default
                               behavior is to use all available images in specified folders.

        """
        # Check start and end time parameters
        if not start_time and not end_time:
            start_time = "1900-01-01 00:00:00"
            end_time = "2100-01-01 00:00:00"
        elif not start_time:
            start_time = "1900-01-01 00:00:00"
        elif not end_time:
            end_time = "2100-01-01 00:00:00"

        # Setup temporary working folders for base and overlay images
        print(f"{Fore.GREEN}Initalizing working folders...", end="")
        self._setup_working_folders(overlaypaths)
        atexit.register(self.cleanup)
        print(f"{Fore.GREEN}Done!")

        print(
            f"{Fore.GREEN} Filtering images and copying to working folders..."
        )
        # Copy base image files to working folder and filter list of filenames to match start and
        # end times
        self._setup_base_imgs(baseimgpath, base_timezone, start_time, end_time)

        # Copy overlay image files to working folders and filter filename lists to match start and
        # end times
        self._setup_overlay_imgs(
            overlaypaths, overlay_timezones, start_time, end_time)

        # Match lengths of overlay image arrays to the length of the base array
        self._setup_matched_lists()
        print(f"{Fore.GREEN} Done!")

        # Get projections and bounds for base and overlays
        if self.base_is_geotiff:
            self.base_crs = gistools.get_crs(self.baseimgfiles[0])
            self.base_bbox = gistools.get_bounds(self.baseimgfiles[0])
            self.base_bbox_latlon = gistools.transform_bbox(
                self.base_bbox, self.base_crs, "EPSG:4326"
            )
        else:
            self.base_crs = None

        self.overlay_crs = []
        self.overlay_bbox = []
        self.overlay_bbox_latlon = []
        for i, overlay in enumerate(self.overlayimgfiles):
            self.overlay_crs.append(gistools.get_crs(overlay[0]))
            self.overlay_bbox.append(gistools.get_bounds(overlay[0]))
            self.overlay_bbox_latlon.append(
                gistools.transform_bbox(
                    self.overlay_bbox[i], self.overlay_crs[i], "EPSG:4326"
                )
            )
        if not self.base_is_geotiff:
            self.base_crs = self.overlay_crs[0]
            self.base_bbox_latlon = self.overlay_bbox_latlon[0]

        print()
        print(f"{Fore.GREEN} Initialization Complete!")
        print()

    def create_overlays(
            self,
            outpath,
            out_crs=None,
            res=None,
            bbox=None,
            overlay_opacities=None,
            overlay_resample=False,
            base_resample=False,
            save_overlays_gtiff=None,
            cumulative=False,
            cumulative_colormap=None,
            output_format="simple_image",
    ):
        """

        @param outpath: (str) path to save output images
        @param out_crs: (rasterio.crs.CRS obj) CRS to use for output images
        @param res: (tup) desired output resolution (width, height)
        @param bbox: (list) [minx, maxy, maxx, miny] in lat/lon coords. If provided, an attempt will
                            be made to crop the base and overlay images to specified bounds. In the
                            event that the provided bbox overlaps partially or not at all with one
                            or more image sets, the default behavior will be to find the maximum
                            common bounds shared by all the image sets. The maximum common bounds
                            will also be used if the bbox parameter is not specified. Note that if
                            pngs are used for the base image set, the assumption is that the user
                            has ensured that the image extents already match those
                            defined by the overlay geotiffs.
        @param overlay_opacities: (list) Opacities of overlay layers. Provide one value (0-1) for
                                         each overlay set. Default is value of 1.
        @param overlay_resample: (bool) Enable resampling of overlay image set(s) upon resizing
        @param base_resample:  (bool) Enable resampling of base image set upon resizing
        @param save_overlays_gtiff: (str) Path to folder for saving individual overlay frames as
                                          separate geotiffs. Default is to discard after compositing
                                          with base image.
        @param cumulative: (bool) If true, a cumulative overlay sequence will be generated over the
                                  base frame(s)
        @param cumulative_colormap: (matplotlib.cm obj) If a colormap is provided, the frame
                                    sequence will automatically be set to cumulative and the color
                                    applied to each overlay will be incremented at each frame.
        @param output_format: (str) options are 'simple_image' which will produce pngs or 'geotiff'
        @return: None
        """
        outpath = Path(outpath)

        if not overlay_opacities:
            overlay_opacities = [1] * len(self.overlaypaths)

        if base_resample:
            base_kwargs = {"resample": True}
        else:
            base_kwargs = {}

        if overlay_resample:
            overlay_kwargs = {"resample": True}
        else:
            overlay_kwargs = {}

        # If no output resolution is specified, use that of the base image
        if not res:
            if self.base_is_geotiff:
                res = gistools.get_shape(str(self.baseimgfiles[0]))
            else:
                res = Image.open(str(self.baseimgfiles[0])).size

        # Check user-provided bbox to make sure it at least overlaps partially with the base/overlay
        # images. If not, default to using the maximal common bounds.
        # If bbox parameter is not provided, default to maximal common bounds
        if bbox and self.base_is_geotiff:
            _base_bbox = gistools.get_bbox_intersection(
                bbox, self.base_bbox_latlon)
            _overlay_bbox = []

            for overlaybbox in self.overlay_bbox_latlon:
                _overlay_bbox.append(
                    gistools.get_bbox_intersection(bbox, overlaybbox))

            _bbox_list = [_base_bbox] + _overlay_bbox
            err1 = (
                f"{Fore.RED} WARNING! Provided bbox is outside the bounds of the base and/or "
                f"overlay images. Using the maximal common bounds of the images instead."
            )
            err2 = (
                f"{Fore.RED} WARNING! Provided bbox only overlaps partially with the base and/or"
                f" overlay images. Final images will be smaller than expected."
            )
            err3 = (
                f"{Fore.RED} WARNING! Provided bbox does not overlap with all overlay sets. Using "
                f"maximal common bounds of the images instead."
            )

            if not all([x == bbox for x in _bbox_list]):
                if any([x is None for x in _bbox_list]):
                    print(err1)
                    bbox = gistools.get_max_bbox(
                        self.base_bbox_latlon, self.overlay_bbox_latlon
                    )
                else:
                    print(err2)
                    bbox1 = gistools.get_bbox_intersection(bbox, _base_bbox)
                    for i, ovbbox in enumerate(_overlay_bbox):
                        bbox1 = gistools.get_bbox_intersection(bbox1, ovbbox)
                    if not bbox1:
                        print(err3)
                        bbox = gistools.get_max_bbox(
                            self.base_bbox_latlon, self.overlay_bbox_latlon
                        )
                    else:
                        bbox = bbox1
        elif bbox and not self.base_is_geotiff:
            print(
                f"{Fore.RED} WARNING! Simple image(s) provided as base. Overlay images are assumed "
                f"to match the lat/lon bounds of the base image(s). For most accurate results, "
                f"provide geotiff(s) for base set."
            )
        else:
            bbox = gistools.get_max_bbox(
                self.base_bbox_latlon, self.overlay_bbox_latlon
            )

        # If user does not provide an output CRS then try to use the base image CRS first; if that
        # isn't available (e.g. due to pngs being used for the base image) then default to the CRS
        # of the first overlay set
        if not out_crs:
            if self.base_crs:
                out_crs = self.base_crs
            else:
                out_crs = self.overlay_crs[0]

        # Reproject base and overlay images if necessary to match the specified output CRS
        if self.base_is_geotiff:
            if self.base_crs != out_crs:
                print(
                    f"{Fore.RED} WARNING! Projection of base image(s) does not match the output "
                    f"CRS. Converting..."
                )
                for f in tqdm(self.baseimgfiles, desc="Image Number: ", colour="blue"):
                    gistools.reproject(f, out_crs)
                print(f"{Fore.RED} Done!")

        for i, overlay in enumerate(self.overlayimgfiles):
            if self.overlay_crs[i] != out_crs:
                print(
                    f"{Fore.RED} WARNING! Projection of overlay images at index {str(i)} does not "
                    f"match the output CRS. Converting..."
                )
                for f in tqdm(overlay, desc="Image Number: ", colour="blue"):
                    gistools.reproject(f, out_crs)
                print(f"{Fore.RED} Done!")

        # Transform bbox coordinates if necessary to match output CRS
        if out_crs != "EPSG:4326":
            outbbox = gistools.transform_bbox(bbox, "EPSG:4326", out_crs)
        else:
            outbbox = bbox

        # Crop base and overlay images to bbox coordinates and resize
        imgheight = res[1]
        imgwidth = res[0]
        if self.base_is_geotiff:
            print(f"{Fore.GREEN} Cropping base and overlay images...", end="")
            if self.base_is_geotiff:
                for f in self.baseimgfiles:
                    gistools.crop_geotiff(
                        f, outbbox, imgheight, imgwidth, opacity=1, **base_kwargs
                    )
            else:
                for f in self.baseimgfiles:
                    img = Image.open(f)
                    img.resize((imgwidth, imgheight))
                    img.save(f)
                    img.close()

            for i, overlay in enumerate(self.overlayimgfiles):
                for f in overlay:
                    gistools.crop_geotiff(
                        f,
                        outbbox,
                        imgheight,
                        imgwidth,
                        opacity=overlay_opacities[i],
                        **overlay_kwargs,
                    )

        print(f"{Fore.GREEN} Done!")

        # Final setup stuff
        if cumulative_colormap:
            cmap = cm.get_cmap(cumulative_colormap)
        else:
            cmap = None

        if save_overlays_gtiff:
            savepath = Path(save_overlays_gtiff)
            savepath.mkdir(exist_ok=True)
            overlaypaths = []
            for i in range(len(self.overlayimgfiles)):
                op = savepath / f"overlay_{str(i)}"
                op.mkdir(exist_ok=True)
                overlaypaths.append(op)

        # Image generation loop
        basedone = False
        img_arr_last = []
        print(f"{Fore.GREEN} Preparing overlay frames...")
        for i in tqdm(range(0, self.framecount), desc="Frame Number: ", colour="green"):
            img_set = []
            img_arr = []

            for row in self.overlayimgfiles:
                img_set.append(row[i])

            # Get base image path
            if self.base_is_static:
                baseimgpath = self.baseimgfiles[0]
            else:
                baseimgpath = self.baseimgfiles[i]

            # Generate save name for image
            if self.base_is_reference:
                if output_format == "simple_image":
                    svname = baseimgpath.stem + ".png"
                elif output_format == "geotiff":
                    svname = baseimgpath.stem + ".tif"
            else:
                if output_format == "simple_image":
                    svname = (
                        self.overlayimgfiles[self.overlay_reference_idx][i].stem
                        + ".png"
                    )
                elif output_format == "geotiff":
                    svname = (
                        self.overlayimgfiles[self.overlay_reference_idx][i].stem
                        + ".tif"
                    )

            # Convert base geotiff to image or just open image if it is a png
            if self.base_is_geotiff:
                if not basedone:
                    base_img = gistools.geotiff_to_image(baseimgpath)
                    if self.base_is_static:
                        basedone = True
            else:
                if not basedone:
                    base_img = Image.open(baseimgpath).convert("RGBA")
                    if self.base_is_static:
                        basedone = True

            if not img_arr_last:
                img_arr_last = [
                    Image.new(mode="RGBA", size=(imgwidth, imgheight)) for x in img_set
                ]

            for overlay, file in enumerate(img_set):
                if save_overlays_gtiff:
                    fname = overlaypaths[overlay] / \
                        (svname.split(".")[0] + ".tif")
                    shutil.copy(file, fname)

                # Convert overlay geotiffs to images and resize
                overlayimg = gistools.geotiff_to_image(file)
                if not self.base_is_geotiff:
                    overlayimg = overlayimg.resize((imgwidth, imgheight))

                # Apply colormap to overlay
                if cumulative_colormap:
                    pixels = overlayimg.load()
                    fname = file.stem
                    tstamp = gistools.get_timestamp(
                        fname, self.overlay_tformats[overlay], self.overlays_tz[overlay]
                    )
                    tnorm = (
                        tstamp - self.overlay_start[overlay]
                    ).total_seconds() / self.overlay_deltas[overlay]
                    color = cmap(tnorm)

                    for j in range(overlayimg.size[0]):
                        for k in range(overlayimg.size[1]):
                            if pixels[j, k][3] != 0:
                                pixels[j, k] = (
                                    int(color[0] * 255),
                                    int(color[1] * 255),
                                    int(color[2] * 255),
                                    pixels[j, k][3],
                                )

                img_arr.append(overlayimg)

            base_img_copy = copy.deepcopy(base_img)
            if cumulative:
                cum_arr = []
                for _img1, _img2 in zip(img_arr, img_arr_last):
                    _img1.alpha_composite(_img2)
                    cum_arr.append(_img1)
                img_arr = cum_arr

            for _img in img_arr:
                base_img_copy.alpha_composite(_img)

                if not outpath.exists():
                    outpath.mkdir()

                if output_format == "simple_image":
                    base_img_copy.save(str(outpath / svname))
                elif output_format == "geotiff":
                    fname = gistools.image_to_geotiff(
                        base_img_copy, bbox, out_crs)
                    shutil.copy(fname, str(outpath / svname))
                    os.remove(fname)

            img_arr_last = copy.deepcopy(img_arr)

        print(f"{Fore.GREEN} Overlays Complete!")

        return

    def cleanup(self):
        shutil.rmtree(str(self.basepath))
        for p in self.overlaypaths:
            shutil.rmtree(str(p))

    def _setup_working_folders(self, overlaypaths):
        self.basetmpdir = tempfile.TemporaryDirectory()
        self.basepath = Path(self.basetmpdir.name)
        self.overlaytmpdirs = []
        self.overlaypaths = []
        for p in overlaypaths:
            _path = tempfile.TemporaryDirectory()
            self.overlaytmpdirs.append(_path)
            self.overlaypaths.append(Path(_path.name))

    def _setup_base_imgs(self, baseimgpath, base_timezone, start_time, end_time):
        self.baseimgfiles = []
        baseimgpath = Path(baseimgpath)

        if baseimgpath.is_dir():
            gtiffs = list(baseimgpath.glob("*.tif")) + \
                list(baseimgpath.glob("*.tiff"))
            gtiffs = sorted(gtiffs, key=lambda item: item.name)
            pngs = list(baseimgpath.glob("*.png"))
            pngs = sorted(pngs, key=lambda item: item.name)
        else:
            if baseimgpath.suffix == ".tif" or baseimgpath.suffix == ".tiff":
                gtiffs = [baseimgpath]
                pngs = []
            elif baseimgpath.suffix == ".png":
                gtiffs = []
                pngs = [baseimgpath]
            else:
                print(f"{Fore.RED} ERROR: No valid image formats found. Exiting...")
                sys.exit(0)

        # Configure timezones and filename formats for base image(s) and overlay images
        if base_timezone:
            self.base_tz = base_timezone.pop("timezone", pytz.utc)
            base_format = base_timezone.pop("filename format", None)
            if base_format:
                self.base_tformat = gistools.time_format_str(base_format)
            else:
                self.base_tformat = gistools.time_format_str(
                    "YYYY-MM-DD hh_mm_ss")
        else:
            self.base_tz = pytz.utc
            self.base_tformat = gistools.time_format_str("YYYY-MM-DD hh_mm_ss")

        # Filter base image lists based on start and end time
        if len(gtiffs) > 1:
            gtiffs = self._filter_img_list(
                gtiffs, start_time, end_time, self.base_tformat, self.base_tz
            )
        if len(pngs) > 1:
            pngs = self._filter_img_list(
                pngs, start_time, end_time, self.base_tformat, self.base_tz
            )

        if gtiffs and pngs:
            print(
                f"{Fore.RED} WARNING: Multiple image formats found in base image folder. Using "
                f"geotiffs..."
            )
            self.base_is_geotiff = True
            for file in gtiffs:
                fname = editortools.copy_image(file, self.basepath)
                self.baseimgfiles.append(fname)
        elif gtiffs:
            self.base_is_geotiff = True
            for file in gtiffs:
                fname = editortools.copy_image(file, self.basepath)
                self.baseimgfiles.append(fname)
        elif pngs:
            self.base_is_geotiff = False
            for file in pngs:
                fname = editortools.copy_image(file, self.basepath)
                self.baseimgfiles.append(fname)
        else:
            print(f"{Fore.RED} ERROR: No valid image formats found. Exiting...")
            sys.exit(0)

        if len(self.baseimgfiles) == 1:
            self.base_is_static = True
        else:
            self.base_is_static = False

        err2 = (
            f"{Fore.RED} ERROR: Dynamic base image requires timezone and/or"
            f"filename format to be specified. Exiting..."
        )

        if not self.base_is_static and self.base_tformat is None:
            print(err2)
            sys.exit(0)

        return

    def _setup_overlay_imgs(
        self, overlaypaths, overlay_timezones, start_time, end_time
    ):
        if overlay_timezones:
            self.overlays_tz = overlay_timezones.pop(
                "timezone", [pytz.utc] * len(overlaypaths)
            )
            overlays_format = overlay_timezones.pop(
                "filename format", [None] * len(overlaypaths)
            )
            if all(overlays_format):
                self.overlay_tformats = [
                    gistools.time_format_str(x) for x in overlays_format
                ]
            else:
                self.overlay_tformats = [
                    gistools.time_format_str("YYYY-MM-DD hh_mm_ss")
                    for x in overlays_format
                ]
        else:
            self.overlays_tz = [pytz.utc] * len(overlaypaths)
            self.overlay_tformats = [
                gistools.time_format_str("YYYY-MM-DD hh_mm_ss")
            ] * len(overlaypaths)

        err = (
            f"{Fore.RED} ERROR: Using the `overlay_timezones` parameter requires all or none of the "
            f"timezones and/or filename formats to be specified. Exiting..."
        )

        if len(self.overlays_tz) != len(overlaypaths) or len(
            self.overlay_tformats
        ) != len(overlaypaths):
            print()
            print(err)
            sys.exit(0)

        # Copy overlay files - Overlays can only be geotiff format
        self.overlayimgfiles = []
        for i, p in enumerate(overlaypaths):
            p = Path(p)
            gtiffs = list(p.glob("*.tif")) + list(p.glob("*.tiff"))
            gtiffs = sorted(gtiffs, key=lambda item: item.name)
            if not gtiffs:
                print(
                    f"{Fore.RED} ERROR: No geotiffs found in overlay path {str(p)}. Exiting..."
                )
                sys.exit(0)
            gtiffs = self._filter_img_list(
                gtiffs,
                start_time,
                end_time,
                self.overlay_tformats[i],
                self.overlays_tz[i],
            )
            row = []
            for file in gtiffs:
                fname = editortools.copy_image(file, self.overlaypaths[i])
                row.append(fname)
            self.overlayimgfiles.append(row)

        return

    def _setup_matched_lists(self):
        self.overlay_start = []
        self.overlay_end = []
        for i, overlay in enumerate(self.overlayimgfiles):
            self.overlay_start.append(
                gistools.get_timestamp(
                    overlay[0].stem, self.overlay_tformats[i], self.overlays_tz[i]
                )
            )
        for i, overlay in enumerate(self.overlayimgfiles):
            self.overlay_end.append(
                gistools.get_timestamp(
                    overlay[-1].stem, self.overlay_tformats[i], self.overlays_tz[i]
                )
            )
        self.overlay_deltas = [
            (tend - tstart).total_seconds()
            for tend, tstart in zip(self.overlay_end, self.overlay_start)
        ]

        self.trange_norms = [(np.linspace(1, x, 1) / x)
                             for x in self.overlay_deltas]

        # Calculate interval between images in each overlay path
        overlay_intervals = []
        for i, overlay in enumerate(self.overlayimgfiles):
            overlay_intervals.append(
                self._calculate_interval(
                    (overlay[0], overlay[1]),
                    (self.overlay_tformats[i],),
                    (self.overlays_tz[i],),
                )
            )

        # Calculate interval between base images, if applicable
        base_interval = 0
        if not self.base_is_static:
            base_interval = self._calculate_interval(
                (self.baseimgfiles[0], self.baseimgfiles[1]),
                (self.base_tformat,),
                (self.base_tz,),
            )
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
        else:
            if not self.base_is_static:
                self.framecount = len(self.overlayimgfiles[mindt_idx])
                reffiles = self.overlayimgfiles[mindt_idx]
                refidx = mindt_idx
            else:
                self.framecount = len(self.overlayimgfiles[mindt_idx])
                reffiles = self.overlayimgfiles[mindt_idx]
                refidx = mindt_idx

        new_overlay_list = []
        for j, overlay in enumerate(self.overlayimgfiles):
            tmp_list = []
            if len(overlay) < self.framecount:
                for i, file in enumerate(reffiles):
                    if self.base_is_reference:
                        closest_idx = self._get_closest_time(
                            (file.stem, self.base_tformat),
                            ([x.stem for x in overlay], self.overlay_tformats[j]),
                            tz=(self.base_tz, self.overlays_tz[j]),
                        )
                    else:
                        closest_idx = self._get_closest_time(
                            (file.stem, self.overlay_tformats[refidx]),
                            ([x.stem for x in overlay], self.overlay_tformats[j]),
                            tz=(self.overlays_tz[refidx], self.overlays_tz[j]),
                        )

                    tmp_list.append(overlay[closest_idx])

                new_overlay_list.append(tmp_list)
            else:
                new_overlay_list.append(overlay)

        self.overlayimgfiles = new_overlay_list

        # Fill in the base file list with duplicates if needed
        if not self.base_is_static:
            if len(self.baseimgfiles) < self.framecount:
                new_base_list = []
                for i, file in enumerate(self.baseimgfiles):
                    closest_idx = self._get_closest_time(
                        (file.stem, self.base_tformat),
                        ([x.stem for x in reffiles],
                         self.overlay_tformats[refidx]),
                        tz=(self.base_tz, self.overlays_tz[refidx]),
                    )
                    new_base_list.append(self.baseimgfiles[closest_idx])
                self.baseimgfiles = new_base_list
        return

    def _get_time(self, mode="base"):
        if mode == "base":
            _times = []
            for f in self.baseimgfiles:
                _times.append(
                    gistools.get_timestamp(
                        f, self.base_tformat, tz=self.base_tz)
                )
            _times.sort()
            start = _times[0]
            end = _times[-1]
        elif mode == "overlay":
            _start = []
            _end = []
            _times = []
            for i, overlay in enumerate(self.overlayimgfiles):
                for f in overlay:
                    _times.append(
                        gistools.get_timestamp(
                            f, self.overlay_tformats[i], tz=self.overlays_tz[i]
                        )
                    )
                _times.sort()
                _start.append(_times[0])
                _end.append(_times[-1])
            start = max(_start)
            end = min(_end)

        else:
            start = None
            end = None

        return start.isoformat(), end.isoformat()

    @staticmethod
    def _bbox_error_msg(self, coord, index=None):
        if not index:
            err = (
                f"{Fore.RED} WARNING! Provided bbox {coord} extent is beyond the bounds of the base "
                f"image. Using image bounds instead."
            )
        else:
            err = (
                f"{Fore.RED} WARNING! Provided bbox {coord} extent is beyond the bounds of the "
                f"overlay image at index {str(index)}. Using image bounds instead."
            )
        return err

    @staticmethod
    def _calculate_interval(files, formatdicts, tz=None):
        """
        Calculate interval in seconds between two files
        @param files: (tup) tuple of filenames containing datetime strings
                      (e.g. '2023-01-01 10_43_00')
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

        t1 = gistools.get_timestamp(file1, tformat1, tz=tz1)
        t2 = gistools.get_timestamp(file2, tformat2, tz=tz2)

        # Convert times to UTC
        t1 = t1.astimezone(pytz.utc)
        t2 = t2.astimezone(pytz.utc)

        # Get delta
        delta = abs((t2 - t1).total_seconds())

        return delta

    @staticmethod
    def _filter_img_list(imglist, start_time, end_time, tformatdict, tz):
        """
        Filter a list of image filenames between start and end times
        """

        # Convert start and end times to UTC
        try:
            tstart = datetime.fromisoformat(start_time)
            if tstart.tzinfo is None:
                tstart = tz.localize(tstart)
                tstart = tstart.astimezone(pytz.utc)
            else:
                tstart = tstart.astimezone(pytz.utc)

        except TypeError:
            tstart = None

        try:
            tend = datetime.fromisoformat(end_time)
            if tend.tzinfo is None:
                tend = tz.localize(tend)
                tend = tend.astimezone(pytz.utc)
            else:
                tend = tend.astimezone(pytz.utc)

        except TypeError:
            tend = None

        new_list = []
        if imglist:
            for img in imglist:
                t = gistools.get_timestamp(img.stem, tformatdict, tz)
                t = t.astimezone(pytz.utc)
                if tend and not tstart:
                    if t <= tend:
                        new_list.append(img)
                elif tstart and not tend:
                    if t >= tstart:
                        new_list.append(img)
                elif tstart and tend:
                    if tstart <= t <= tend:
                        new_list.append(img)
                else:
                    new_list.append(img)

        return new_list

    @staticmethod
    def _get_closest_time(target, srcarray, tz=None):
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

        target_time = gistools.get_timestamp(target[0], target[1], targettz)
        src_times = [gistools.get_timestamp(
            x, src_tformat, srctz) for x in _srcarray]

        if targettz != srctz:
            target_time = target_time.astimezone(srctz)

        closest_time = min(src_times, key=lambda d: abs(d - target_time))
        idx = src_times.index(closest_time)
        # file = srcarray[idx]

        return idx

    @staticmethod
    def _get_separator(instr, startidx, startchar):
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


def create_video(vidsavepath, finaloverlaypath, fps=20, codec="rawvideo", **kwargs):
    img_filenames = list(Path(finaloverlaypath).glob("*.png"))
    img_filenames = sorted(img_filenames, key=lambda item: item.name)

    tmpdir = tempfile.TemporaryDirectory()
    tmppath = Path(tmpdir.name)
    newfiles = []

    if kwargs:
        timestamps = kwargs.get("timestamps", None)
        if timestamps:
            for f in img_filenames:
                fnew = str(tmppath / (f.stem + ".png"))
                newfiles.append(fnew)
                fmod = f.stem.replace("_", ":")
                flist = fmod.split(" ")
                timestamps["label"] = (" ").join(flist[0:2])
                kwargs["timestamps"] = timestamps
                img = Image.open(str(f))
                img = editortools.modify_image(img, **kwargs)
                if codec == "rawvideo" or codec == "png":
                    img = editortools.convert_color(img)
                img.save(fnew)
                img.close()
        else:
            for f in img_filenames:
                fnew = str(tmppath / (f.stem + ".png"))
                newfiles.append(fnew)
                img = Image.open(str(f))
                img = editortools.modify_image(img, **kwargs)
                if codec == "rawvideo" or codec == "png":
                    img = editortools.convert_color(img)
                img.save(fnew)
                img.close()

    imgarr = [str(f) for f in newfiles]
    clip = ImageSequenceClip(imgarr, fps=fps)
    clip.write_videofile(vidsavepath, codec=codec)
    shutil.rmtree(str(tmppath))

    return
