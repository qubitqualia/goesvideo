import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

import boto3
import boto3.exceptions
import botocore.exceptions
import numpy as np
import pytz
import rasterio.crs
import satpy
import satpy.utils
import trollimage.colormap as cm
import urllib3
import yaml
from PIL import Image
from botocore import UNSIGNED
from botocore.config import Config
from colorama import Fore
from matplotlib import colormaps
from moviepy.editor import ImageSequenceClip
from pyresample.geometry import AreaDefinition
from satpy.writers import to_image
from tqdm import tqdm

import goesvideo.exceptions as exceptions

# import goesvideo.utils.editortools as utils
from goesvideo.utils import editortools as utils
from goesvideo.utils import gistools


class GoesBase:
    def __init__(self, sat, region, product):
        self.region_list = ["full", "conus"]
        self.band_list = [
            "C01",
            "C02",
            "C03",
            "C04",
            "C05",
            "C06",
            "C07",
            "C08",
            "C09",
            "C10",
            "C11",
            "C12",
            "C13",
            "C14",
            "C15",
        ]
        self.satellite = sat

        if self.satellite == "goes-east":
            self.bucket = "noaa-goes16"
        elif self.satellite == "goes-west":
            self.bucket = "noaa-goes18"
        else:
            raise exceptions.InvalidArgumentError(
                "Satellite must be either 'goes-east' or 'goes-west'."
            )

        self.region = region

        if self.region not in self.region_list:
            raise exceptions.InvalidArgumentError(
                "Region must be either 'full' or 'conus'."
            )

        if product[-1] == "F" or product[-1] == "C":
            if "FDC" not in product:
                self.product = product
                # print(
                #    f"{Fore.RED} WARNING - Product should be specified without the trailing 'F' or "
                #    f"'C' indicating region."
                # )
            else:
                if "FDCC" in product or "FDCF" in product:
                    self.product = product
                    # print(
                    #    f"{Fore.RED} WARNING - Product should be specified without the trailing 'F' "
                    #    f"or 'C' indicating region."
                    # )
                else:
                    if self.region == "conus":
                        self.product = product + "C"
                    else:
                        self.product = product + "F"
        else:
            if self.region == "conus":
                self.product = product + "C"
            else:
                self.product = product + "F"

        file = files("goesvideo") / "etc" / "abibands.yaml"
        with open(str(file), "r") as f:
            self.scene_reqs = yaml.safe_load(f)

    def _get_scene_bands(self, scenename):
        """Returns the CMI bands required to produce the given scene name"""
        try:
            req_bands = self.scene_reqs["composites"][scenename]["bands"]
        except KeyError:
            req_bands = [scenename]
            if scenename not in self.band_list:
                print()
                print(
                    f"{Fore.RED}WARNING - Could not find scene name in 'abibands.yaml'. If results "
                    f"are unexpected, try adding this scene to the yaml file."
                )

        return req_bands

    @staticmethod
    def _get_band_from_filename(filename):
        """Extracts the band identifier from AWS filenames"""

        ch = ""
        if "ABI-L2-CMI" in filename:
            for _s in filename.split("_"):
                if _s.startswith("ABI"):
                    _s = _s.split("-")[-1]
                    ch = _s[-3:]
        else:
            raise exceptions.InvalidArgumentError(
                "AWS filename does not correspond to the ABI-L2-CMI product. "
            )

        return ch

    @staticmethod
    def _get_timestamp_from_filename(filename):
        """
        Returns the datetime corresponding to the end time of the provided nc filename
        @param filename: (str) nc filename
        @return: datetime object
        """
        if isinstance(filename, Path):
            filename = str(filename)

        if "\\" in filename:
            f = filename.split("\\")[-1].split("_")
        elif "/" in filename:
            f = filename.split("/")[-1].split("_")
        else:
            f = filename.split("_")

        t = None
        for _s in f:
            if _s.startswith("e"):
                _s = _s.lstrip("e")
                _year = int(_s[0:4])
                _day = int(_s[4:7])
                _hour = int(_s[7:9])
                _min = int(_s[9:11])
                _sec = int(_s[11:13])
                t = datetime(
                    _year, 1, 1, hour=_hour, minute=_min, second=_sec
                ) + timedelta(days=_day - 1)

        return t

    def _filter_by_interval(self, filedict, interval):
        """
        Filters a file dictionary spanning a time range using the specified interval
        @param filedict: (dict) filenames aggregated by band identifier (e.g. {'C01': [<filenames>],
               'C02':...}
        @param interval: (int) desired time interval between nc file end times in minutes
        @return: (dict) filtered dict of filenames
        """
        retdict = {}
        for key in filedict:
            if key not in retdict:
                retdict[key] = {
                    "filenames": [],
                    "sizes": [],
                    "etags": [],
                    "start_time": None,
                    "end_time": None,
                }
            filenames = filedict[key]["filenames"]
            sizes = filedict[key]["sizes"]
            etags = filedict[key]["etags"]
            tstamps = [self._get_timestamp_from_filename(f) for f in filenames]
            last = tstamps[0]
            filenames_filtered = []
            sizes_filtered = []
            etags_filtered = []
            for i, _t in enumerate(tstamps):
                if i == 0:
                    filenames_filtered.append(filenames[i])
                    sizes_filtered.append(sizes[i])
                    etags_filtered.append(etags[i])

                diff = int((_t - last).total_seconds() / 60)

                if interval <= diff <= interval + 21:
                    filenames_filtered.append(filenames[i])
                    sizes_filtered.append(sizes[i])
                    etags_filtered.append(etags[i])
                    last = tstamps[i]

            retdict[key]["filenames"] += filenames_filtered
            retdict[key]["sizes"] += sizes_filtered
            retdict[key]["etags"] += etags_filtered

            retdict[key]["start_time"] = filedict[key]["start_time"]
            retdict[key]["end_time"] = last

        return retdict


class GoesDownloader(GoesBase):
    def __init__(self, sat, region, product, base_dir=None):
        super().__init__(sat, region, product)
        s3options = {"us_east_1_regional_endpoint": "regional"}
        self.s3client = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                s3=s3options,
                connect_timeout=5,
                read_timeout=5,
                region_name="us-east-1",
            ),
        )

        if base_dir:
            self.basedir = Path(base_dir)
            self.datapath = self.basedir / "goesdata"
            self.datapath.mkdir(exist_ok=True)
        else:
            print(
                f"{Fore.RED}WARNING - No base directory provided. Data will be saved to current "
                f"working directory."
            )
            self.basedir = Path.cwd()
            self.datapath = self.basedir / "GoesVideo - Data"
            self.datapath.mkdir(exist_ok=True)

        self.remote_scene_files = {}
        self._init_verbosity_on = False
        self._scene_dict_ready = False

    def download_all(
        self, start_time, end_time, interval=5, checksum=False, force=False
    ):
        """
        Download all available files between start and end times at the
        specified interval
        @param start_time: (str) isoformat datetime string (utc time)
        @param end_time: (str) isoformat datetime string (utc time)
        @param interval: (int) minutes between file timestamps
        @param checksum: (bool) If true, ensures file integrity after download
        @param force: (bool) If true, forces completion of function bypassing any
                             input prompts
        @return: None
        """

        tstart = datetime.fromisoformat(start_time)
        tend = datetime.fromisoformat(end_time)

        needed_files, sizes, etags = self.check_for_needed_files(
            tstart, tend, interval=interval
        )

        kwargs = {}
        if checksum:
            kwargs["etags"] = etags

        kwargs["force"] = force

        if needed_files:
            self._boto_bulk_download(needed_files, sizes, **kwargs)

        return

    def download_scene(
        self, scenename, start_time, end_time, interval=5, checksum=False, force=False
    ):
        """
        Downloads all files required to produce the specified scene between the
        start and end times
        @param scenename: (str) name of Satpy scene (e.g. 'true_color')
        @param start_time: (str) isoformat datetime string (utc time)
        @param end_time: (str) isoformat datetime string (utc time)
        @param interval: (int) minutes between file timestamps
        @param checksum: (bool) If true, ensures file integrity after download
        @param force: (bool) If true, forces completion of function bypassing any
                             input prompts
        @return: None
        """

        tstart = datetime.fromisoformat(start_time)
        tend = datetime.fromisoformat(end_time)

        needed_files, sizes, etags = self.check_for_needed_files(
            tstart, tend, scenename=scenename, interval=interval
        )

        kwargs = {}
        if checksum:
            kwargs["etags"] = etags

        kwargs["force"] = force

        if needed_files:
            self._boto_bulk_download(needed_files, sizes, **kwargs)

        return

    def download_files(self, ncfilenames, ncfilesizes, etags=None):
        # Determine if nc files are given as full AWS path or stem

        if "/" not in ncfilenames[0]:
            # Extract year, day of year and hour
            awsfilenames = []
            for file in ncfilenames:
                _strsplit = file.split("_")
                for _s in _strsplit:
                    if _s.startswith("e"):
                        _s = _s.lstrip("e")
                        year = _s[0:4]
                        day = _s[4:7]
                        hour = _s[7:9]

                awsname = f"{self.product}/{year}/{day}/{hour}/{file}"
                awsfilenames.append(awsname)
        else:
            awsfilenames = ncfilenames

        # Download files
        kwargs = {"etags": etags, "force": True}
        self._boto_bulk_download(awsfilenames, ncfilesizes, **kwargs)

        return

    def get_scene_dict(self, scenename):
        """
        Returns the filenames required to produce the given scene name as
        a dictionary aggregated by CMI band
        @param scenename: (str) name of Satpy scene (e.g. 'true_color')
        @return: (dict) filenames
        """
        # Currently, the downloader can handle the ABI-L2-CMIP and ABI-L2-FDC products
        if not scenename.startswith("Power") and not scenename.startswith("Temp"):
            req_bands = self._get_scene_bands(scenename)
        else:
            req_bands = ["FDC"]

        if self._scene_dict_ready:
            retdict = {}
            for key in req_bands:
                if key not in retdict:
                    retdict[key] = []
                files = self.remote_scene_files[key]["filenames"]
                for f in files:
                    _s = f.split("/")[-1]
                    fn = self.datapath / _s
                    retdict[key].append(str(fn))

            self._scene_dict_ready = False
        else:
            retdict = None

        return retdict

    def check_for_needed_files(self, start_time, end_time, scenename=None, interval=5):
        """
        Determines the files not available locally which require downloading from AWS
        @param start_time: (str) isoformat datetime string (utc time)
        @param end_time: (str) isoformat datetime string (utc time)
        @param scenename: (str) name of Satpy scene (e.g. 'true_color'); if not provided
                          the function will check all CMI bands
        @param interval: (int) minutes between file timestamps
        @return: (tup) filenames to be downloaded [(str)], sizes in bytes [(int)]
        """
        needed_files = []
        needed_sizes = []
        self._check_existing()

        if not isinstance(start_time, datetime):
            start_time = datetime.fromisoformat(start_time)
        if not isinstance(end_time, datetime):
            end_time = datetime.fromisoformat(end_time)

        # Refresh remote file dict
        self._refresh_remote_list(start_time, end_time)

        # Filter filenames using interval
        key = list(self.remote_scene_files.keys())[0]
        if len(self.remote_scene_files[key]) > 2:
            filtered_remote = self._filter_by_interval(
                self.remote_scene_files, interval
            )
        else:
            filtered_remote = self.remote_scene_files

        # Currently, the downloader can handle the ABI-L2-CMIP and ABI-L2-FDC products
        if scenename:
            if not scenename.startswith("Power") and not scenename.startswith("Temp"):
                req_bands = self._get_scene_bands(scenename)
            else:
                req_bands = ["FDC"]
        else:
            req_bands = filtered_remote.keys()

        # Check consistency of filtered dict and report any problems
        fkey = list(filtered_remote.keys())[0]
        length_compare = len(filtered_remote[fkey]["filenames"])

        for key in filtered_remote:
            length = len(filtered_remote[key]["filenames"])
            if length != length_compare:
                print()
                diff = length_compare - length
                if diff > 0:
                    print(
                        f"{Fore.RED}WARNING - Band '{key}' is missing {str(diff)} data files on "
                        f"AWS."
                    )
                else:
                    print(
                        f"{Fore.RED}WARNING - Band '{key}' has {str(diff)} extra data files on "
                        f"AWS."
                    )

        # Check which remote files are already available locally

        needed_files = []
        needed_sizes = []
        needed_etags = []

        for b in req_bands:
            existing_files = []
            existing_sizes = []
            existing_etags = []
            remote = filtered_remote[b]["filenames"]
            remotesizes = filtered_remote[b]["sizes"]
            remoteetags = filtered_remote[b]["etags"]
            remotestem = [f.split("/")[-1].rstrip(".nc") for f in remote]
            for file in self.datapath.iterdir():
                if file.is_file():
                    stem = file.stem
                    if stem in remotestem:
                        idx = remotestem.index(stem)
                        existing_files.append(remote[idx])
                        existing_sizes.append(remotesizes[idx])
                        existing_etags.append(remoteetags[idx])

            needed_files += list(
                set(filtered_remote[b]["filenames"]).difference(set(existing_files))
            )
            needed_sizes += list(
                set(filtered_remote[b]["sizes"]).difference(set(existing_sizes))
            )
            needed_etags += list(
                set(filtered_remote[b]["etags"]).difference(set(existing_etags))
            )

        if needed_files:
            ret = (needed_files, needed_sizes, needed_etags)
        else:
            ret = ([], [], [])

        return ret

    def get_remote_filenames(self, start_time, end_time=None):
        """
        Queries AWS for data filenames between start and end times.

        @param start_time: (datetime obj) start time (utc time)
        @param end_time: (datetime obj) end time (utc time) (optional)
        @return: (dict) filename dict aggregated by CMI band
        """
        # Convert times to UTC
        start = pytz.utc.localize(start_time)
        if end_time:
            end = pytz.utc.localize(end_time)
        else:
            end = start_time

        filedict = {}
        band_complete = {}

        year = start.timetuple().tm_year
        day = start.timetuple().tm_yday

        _exit = False

        if self.satellite == "goes-east":
            satprefix = "G16"
        else:
            satprefix = "G18"

        totkeys = 0
        totsize = 0

        if self._init_verbosity_on:
            print()
            print(f"{Fore.GREEN}Initializing...", end="")
        else:
            print()
            print(f"{Fore.GREEN}Querying files on AWS...")

        while True:
            # Prepare payload for S3 client request
            prefix = f"{self.product}/{year}/{day:03.0f}/"
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}

            # Grab pages from AWS
            while True:
                if not self._init_verbosity_on:
                    print(
                        f"{Fore.GREEN}File count: {str(totkeys)}       Size (MB): {totsize:.2f}",
                        end="\r",
                    )
                resp = self.s3client.list_objects_v2(**kwargs)

                if "Contents" in resp:
                    for obj in resp["Contents"]:
                        key = obj["Key"]
                        if "M6" in key:
                            if satprefix in key:
                                size = obj["Size"]
                                etag = obj["ETag"]
                                totsize += float(size / 1000000)
                                totkeys += 1

                                # Get band from filename
                                if "ABI-L2-CMI" in key:
                                    slashsplit = key.split("/")
                                    ch = ""
                                    for _s in slashsplit:
                                        if _s.startswith("OR"):
                                            ch = _s.split("_")[1].split("-")[-1][-3:]
                                    numbands = 16
                                else:
                                    ch = key.split("/")[0].split("-")[-1][:-1]
                                    numbands = 1

                                # Add entry to dict
                                if ch not in filedict:
                                    filedict[ch] = {
                                        "filenames": [],
                                        "sizes": [],
                                        "etags": [],
                                        "start_time": None,
                                        "end_time": None,
                                    }
                                    band_complete[ch] = False

                                # Get time from filename and add file to dict if it is within range
                                tfile = self._get_timestamp_from_filename(key)
                                tfile = pytz.utc.localize(tfile)

                                if start <= tfile <= end:
                                    if not band_complete[ch]:
                                        filedict[ch]["filenames"].append(key)
                                        filedict[ch]["sizes"].append(size)
                                        filedict[ch]["etags"].append(etag)

                                elif tfile > end:
                                    band_complete[ch] = True

                                if len(band_complete.keys()) == numbands:
                                    if all(band_complete.values()):
                                        _exit = True

                                if _exit:
                                    break

                else:
                    _exit = True
                    break

                if not _exit:
                    try:
                        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
                    except KeyError:
                        break
                else:
                    break

            if _exit:
                break
            else:
                year = (tfile + timedelta(days=1)).timetuple().tm_year
                day = (tfile + timedelta(days=1)).timetuple().tm_yday

        print(f"{Fore.GREEN}Done!")
        self._init_verbosity_on = False

        return filedict

    def _boto_bulk_download(self, filenames, sizes, etags=None, force=False):
        """
        Perform bulk download of GOES data files from AWS.

        @param filenames: [(str)] list of AWS filenames
        @param sizes: [(int)] list of file sizes
        @param etags: [(str)] list of md5 checksums for files
        @param force: (bool) If true, forces completion of function bypassing any
                             input prompts
        @return: None
        """
        maxtries = 3
        _exit = False

        if len(filenames) == 0:
            print()
            print(f"{Fore.GREEN}All data is locally available. Skipping all downloads.")
            _exit = True
        else:
            sz = sum(sizes) / 1000000
            if sz > 50000:
                sz = sz / 1000
                if not force:
                    ask = input(
                        f"{Fore.RED}WARNING - Requested files have total size of {str(int(sz))} "
                        f"GB. Proceed? (Y/N): "
                    )
                else:
                    ask = "Y"

                if ask.upper() != "Y":
                    _exit = True
            else:
                print()
                if sz != 0:
                    print(
                        f"{Fore.GREEN}Downloading {str(len(filenames))} file(s) with total size "
                        f"{str(int(sz))}MB..."
                    )
                else:
                    print(f"{Fore.GREEN}Downloading {str(len(filenames))} file(s)...")

        if not _exit:
            total_failures = 0
            for i, file in enumerate(tqdm(filenames, colour="green")):
                out = str(self.datapath / file.split("/")[-1])
                tries = 0

                if total_failures > 5:
                    print()
                    print(f"{Fore.RED}Could not download needed files. Exiting...")
                    sys.exit(0)

                while True:
                    try:
                        self.s3client.download_file(self.bucket, file, out)
                        # Checksum not working correctly
                        # if etags:
                        #    md5hash = self._etag_checksum(out)
                        #    etag = etags[i].replace('\"', '')
                        #    if etag != md5hash:
                        #        print()
                        #        print(f"{Fore.RED} ERROR - Checksum failure for file: {file}")
                        #        print()
                        #        print(f"{Fore.GREEN} Trying again...")
                        #        tries += 1
                        #    else:
                        #        break
                        # else:
                        #    break

                    except botocore.exceptions.ClientError as error:
                        if error.response["Error"]["Code"] == "LimitExceededException":
                            if tries <= maxtries:
                                print()
                                print(
                                    f"{Fore.RED}AWS call limit exceeded. Waiting to retry..."
                                )
                            tries += 1
                            time.sleep(60)
                        elif error.response["Error"]["Code"] == "404":
                            if tries <= maxtries:
                                print()
                                print(
                                    f"{Fore.RED}404 Not Found error for file: {file}."
                                )
                                print("Retrying...")
                            tries += 1
                            time.sleep(0.5)
                    except botocore.exceptions.ConnectTimeoutError:
                        if tries <= maxtries:
                            print()
                            print(
                                f"{Fore.RED}Timeout while trying to connect. Waiting to retry..."
                            )
                        tries += 1
                        time.sleep(60)
                    except botocore.exceptions.ConnectionError:
                        if tries <= maxtries:
                            print()
                            print(f"{Fore.RED}Connection error. Waiting to retry...")
                        tries += 1
                        time.sleep(60)
                    except botocore.exceptions.ConnectionClosedError:
                        if tries <= maxtries:
                            print()
                            print(f"{Fore.RED}Connection closed. Waiting to retry...")
                        tries += 1
                        time.sleep(60)
                    except boto3.exceptions.RetriesExceededError:
                        print()
                        print(f"{Fore.RED}Max retries exceeded. Waiting to retry...")
                        tries += 1
                        time.sleep(60)
                    except urllib3.exceptions.SSLError:
                        print()
                        print(f"{Fore.RED}Encountered SSL error. Waiting to retry...")
                        tries += 1
                    finally:
                        if tries > maxtries:
                            print()
                            print(
                                f"{Fore.RED}WARNING - Download failed for file: {file}. Skipping "
                                f"this file."
                            )
                            total_failures += tries
                            break
                        elif os.path.exists(out):
                            break

    def _check_existing(self):
        """Returns list of existing nc data files stored locally and removes any partial files
        from the datapath"""
        existing = []
        for file in self.datapath.iterdir():
            if file.is_file():
                if file.suffix != ".nc" and ".nc" in file.stem:
                    os.remove(str(file))
                else:
                    existing.append(str(file))

        return existing

    @staticmethod
    def _etag_checksum(filename, chunk_size=8 * 1024 * 1024):
        md5s = []
        with open(filename, "rb") as f:
            for data in iter(lambda: f.read(chunk_size), b""):
                md5s.append(hashlib.md5(data).digest())
        m = hashlib.md5(b"".join(md5s))

        return "{}-{}".format(m.hexdigest(), len(md5s))

    def _refresh_remote_list(self, start_time, end_time):
        """
        Refreshes the remote file dictionary. Dict is structured as:
        {'C01': {'filenames': [],
                 'sizes': [],
                 'start_time': datetime obj,
                 'end_time': datetime obj}}
        @param start_time: (str) isoformat datetime string (utc time)
        @param end_time: (str) isoformat datetime string (utc time)
        @return: None
        """

        # Convert isoformat times to datetimes
        if not isinstance(start_time, datetime):
            start_time = datetime.fromisoformat(start_time)
        if not isinstance(end_time, datetime):
            end_time = datetime.fromisoformat(end_time)

        # Get remote file list between start and end time
        self.remote_scene_files = self.get_remote_filenames(start_time, end_time)
        start_times = []
        end_times = []

        for ch in self.remote_scene_files:
            try:
                start_times.append(
                    self._get_timestamp_from_filename(
                        self.remote_scene_files[ch]["filenames"][0]
                    )
                )
                end_times.append(
                    self._get_timestamp_from_filename(
                        self.remote_scene_files[ch]["filenames"][-1]
                    )
                )
            except IndexError:
                self._handle_index_error()

        start_ch = []
        end_ch = []
        for i, key in enumerate(self.remote_scene_files):
            _start = start_times[i]
            _end = end_times[i]
            self.remote_scene_files[key]["start_time"] = _start
            self.remote_scene_files[key]["end_time"] = _end

            if abs((_start - start_time).total_seconds()) / 60 > 30:
                start_ch.append(key)
            if abs((_end - end_time).total_seconds()) / 60 > 30:
                end_ch.append(key)

        if start_ch:
            keystr = (",").join(start_ch)
            keystr = keystr.rstrip(",")
            print(
                f"{Fore.RED}WARNING - Available start time for channel(s) {keystr} is more than 30 minutes outside "
            )
            print(f"{Fore.RED}the requested time of {datetime.isoformat(start_time)}. ")
            print()
            print(f"{Fore.RED} Using new start time of {datetime.isoformat(_start)}")
        if end_ch:
            keystr = (",").join(end_ch)
            keystr = keystr.rstrip(",")
            print(
                f"{Fore.RED}WARNING - Available end time for channel(s) {keystr} is more than 30 minutes outside "
            )
            print(f"{Fore.RED}the requested time of {datetime.isoformat(end_time)}. ")
            print()
            print(f"{Fore.RED}Using new end time of {datetime.isoformat(_end)}")

        self._scene_dict_ready = True
        return

    def _handle_index_error(self):
        _str = ",".join(self.remote_scene_files)
        _str = _str.rstrip(",")
        raise exceptions.UnavailableDataError(
            f"No data found on AWS for the requested time frame for channels: {_str}"
        )


class GoesCompositor(GoesBase):
    def __init__(self, sat, region, product, base_dir=None):
        super().__init__(sat, region, product)

        if not base_dir:
            print(
                f"{Fore.RED}WARNING - No base directory provided. Images will be saved to current "
                f"working directory. "
                f"Any existing data files will not be available for compositing."
            )
            self.base_dir = Path.cwd()
            self.imgsvpath = self.base_dir / "GoesVideo - Images"
            self.datapath = self.base_dir / "GoesVideo - Data"
            self.downloader = GoesDownloader(sat, region, product)
        else:
            self.base_dir = Path(base_dir)
            self.imgsvpath = self.base_dir / "Scenes"
            self.datapath = self.base_dir / "goesdata"
            self.downloader = GoesDownloader(
                sat, region, product, base_dir=self.base_dir
            )

        self.base_dir.mkdir(exist_ok=True)
        self.imgsvpath.mkdir(exist_ok=True)
        self.datapath.mkdir(exist_ok=True)
        self.coastoptions = {}
        self.satpy_debug = False
        self.satpy_cache = False
        self.satpy_cache_dir = None
        self.last_img_folder = None
        self.default_resample_res = (
            500
        )  # this is the most resolution available from GOES; if desired can change
        # to 1000 or 2000 if working with channels with lower resolution

    def create_composites(
        self,
        scenename,
        start_time,
        end_time,
        bbox=None,
        interval=1,
        output_format="simple_image",
        keep_filenames=False,
        folder_name=None,
        tzinfo=None,
        coastlines=False,
        resampling=None,
        force=False,
        delete_data=False,
        **kwargs,
    ):
        """

        @param scenename: (str) name of Satpy scene (e.g. 'true_color')
        @param start_time: (str) isoformat datetime string (utc time)
        @param end_time: (str) isoformat datetime string (utc time)
        @param bbox: (tup) latitude-longitude coordinates for cropping the output scene given as
                           (west, south, east, north)
        @param interval: (int) desired interval in minutes between composite images
        @param output_format: (str) file format for output images (e.g. 'simple_image', 'geotiff')
        @param keep_filenames: (bool) If true, output images will be saved using the original nc
                                      filename template.Otherwise, ouput images will be saved using
                                      a more readable timestamp. Timezone info for the timestamp
                                      can be provided using the 'tzinfo' parameter, if desired.
        @param folder_name: (str) Desired name of the subfolder to which images will be saved. If
                                  none is provided a random name will be generated.
        @param tzinfo: (tup) timezone info for generating readable output filenames, provided as a
                             tuple containing a pytz timezone object and 3 character timezone
                             abbreviation. If not provided, filenames generated when
                             'keep_filenames' is True will be returned as UTC
        @param coastlines: (bool) If true, coastlines will be added to the output images. Options
                                  for coastlines must be set beforehand by calling
                                  'set_coastlines_options'
        @param resampling: (tup) Resampling options for satpy resampler provided as a tuple
                                 containing the resampling area and resampler method to use.
                                 Options for the resampling area include 'finest', 'coarsest',
                                 'lowest area' or an area explictly defined in the satpy
                                 areas.yaml file. If not provided, 'finest' is used as the default
                                 area. This typically works well and produces a high quality image.
                                 However, it can fail for some composites such as those with
                                 day-night features. The default behavior is for the function to
                                 attempt using 'finest' and if that fails then fall back to
                                 'lowest area'. The options for the resampler method include
                                 'native', 'nearest', etc. To disable resampling completely
                                 pass a tup containing ('none', 'none').
        @param force: (bool) If true, forces completion of function bypassing any
                             input prompts
        @param delete_data (bool) If true, function will delete all underlying datasets used to
                                  produce composites upon completion
        @param kwargs: (dict) function keyword arguments and/or kwargs to be passed to satpy
                              compositor
        @return: None
        """

        # Turn off debugging by default
        if not self.satpy_debug:
            satpy.utils.debug_off()

        # Pack outgoing kwargs
        if not kwargs:
            kwargs = {}

        if start_time:
            kwargs["start_time"] = start_time
        if end_time:
            kwargs["end_time"] = end_time
        if bbox:
            kwargs["bbox"] = bbox
        if output_format:
            kwargs["output_format"] = output_format
        if tzinfo:
            kwargs["tzinfo"] = tzinfo
        if delete_data:
            kwargs["delete_data"] = delete_data

        kwargs["keep_filenames"] = keep_filenames
        kwargs["coastlines"] = coastlines
        kwargs["resampling"] = resampling

        # Gather local data required to produce scene or offer to download it
        # Update local and remote file lists and see which remote files are needed to prepare
        # the scene
        needed_files, sizes, etags = self.downloader.check_for_needed_files(
            start_time, end_time, scenename=scenename, interval=interval
        )

        if needed_files:
            if not force:
                print()
                ask = input(
                    f"{Fore.RED}Requested scene is not available using the existing data path(s). "
                    f"Download data now? (Y/N): "
                )
            else:
                ask = "Y"

            if ask.upper() == "Y":
                self.downloader.download_files(needed_files, sizes, etags=etags)

                needed_files = [f.split("/")[-1] for f in needed_files]
                for file in self.datapath.iterdir():
                    if file.is_file():
                        try:
                            idx = needed_files.index(file.stem + file.suffix)
                            needed_files.pop(idx)
                        except ValueError:
                            pass
                if len(needed_files) > 0:
                    raise exceptions.GenericDownloadError(
                        "There was a problem downloading the necessary files from AWS."
                    )
            else:
                sys.exit(0)

        # Create subfolder and save metadata
        if not folder_name:
            folder_name = str(uuid.uuid4())

        try:
            (self.imgsvpath / folder_name).mkdir(exist_ok=False)
        except FileExistsError:
            if not force:
                print()
                ask = input(
                    f"{Fore.RED}WARNING - Image subfolder provided by the 'folder_name' argument "
                    f"already exists. \n Existing images will be overwritten! Continue anyway? (Y/N): "
                )
            else:
                ask = "Y"

            if ask.upper() == "Y":
                (self.imgsvpath / folder_name).mkdir(exist_ok=True)
                files = (self.imgsvpath / folder_name).glob("*.*")
                for f in files:
                    os.remove(f)
            else:
                print()
                print(f"{Fore.GREEN} Exiting...")
                sys.exit(0)

        # Generate composites and save output files to instance save path
        # Cleanup temp directory after saving

        scene_dict = self.downloader.get_scene_dict(scenename)
        self.composites_from_files(
            scenename, scene_dict, folder_name=folder_name, **kwargs
        )

        retfolder = self.imgsvpath / folder_name

        return retfolder

    def composites_from_files(self, scenename, scene_dict, folder_name=None, **kwargs):
        """
        Prepare a satpy composite for a given scene name using the nc filenames
        contained in scene_dict
        @param scenename: (str) satpy composite name
        @param scene_dict: (dict) {'C01': [nc_file_t0.....nc_file_tf], 'C02':....}
        @param folder_name: (str) name of subfolder in Images path to save the images, if
                                  none provided a temp folder will be used
        @param kwargs: (dict) kwargs used by this function are removed and whatever is left
                              over is passed to the satpy compositor
        @return: (TemporaryDirectory) temp folder where files have been saved
        """

        # Unpack kwargs
        start_time = kwargs.pop("start_time", "")
        end_time = kwargs.pop("end_time", "")
        keep_filenames = kwargs.pop("keep_filenames", False)
        interval = kwargs.pop("interval", 1)
        bbox = kwargs.pop("bbox", None)
        tzinfo = kwargs.pop("tzinfo", None)
        resample = kwargs.pop("resampling", None)
        coastlines = kwargs.pop("coastlines", None)
        output_format = kwargs.pop("output_format", "simple_image")
        delete_data = kwargs.pop("delete_data", False)
        reader = kwargs.pop("reader", "abi_l2_nc")

        # Create temp directory
        if folder_name:
            folder_path = Path(self.imgsvpath / folder_name)
        else:
            folder = tempfile.TemporaryDirectory()
            folder_path = Path(folder.name)

        # Write metadata.json
        with open(str(folder_path / "metadata.json"), "w") as f:
            _dict = {
                "Scene": scenename,
                "Satellite": self.satellite,
                "Product": self.product,
                "Region": self.region,
                "Start_Time": start_time,
                "End_Time": end_time,
                "Resolution": self.default_resample_res,
                "Keep_Names": keep_filenames,
                "Interval": interval,
            }
            if bbox:
                _dict["Crop_Box"] = bbox
            if tzinfo:
                _dict["Timezone"] = tzinfo[0].zone
                _dict["TZ_Abbr"] = tzinfo[1]

            pretty = json.dumps(_dict, indent=4)
            f.write(pretty)

        # Determine number of time steps contained in scene_dict
        for key in scene_dict:
            tsteps = len(scene_dict[key])

        # Main loop for generating composite images
        print()
        print(f"{Fore.GREEN}Preparing scene...")
        time.sleep(0.5)

        counter = 0
        for i in tqdm(range(tsteps), colour="green"):
            kwargs = {}
            skip = False

            if self.product.startswith("ABI"):
                filenames = []
                for key in scene_dict:
                    try:
                        filenames.append(scene_dict[key][i])
                    except IndexError:
                        skip = True

                if not skip:
                    try:
                        scene = satpy.Scene(reader=reader, filenames=filenames)
                        scene.load([scenename])
                    except KeyError:
                        pass
                    except OSError as e:
                        print()
                        print(
                            f"{Fore.RED}WARNING - Encountered a corrupted data file while generating "
                            f"composite. Attempting to download it again..."
                        )
                        print(e)
                        p = Path(e.filename)
                        file = p.stem + p.suffix
                        self.downloader.download_files([file], [0])

                        try:
                            scene = satpy.Scene(reader="abi_l2_nc", filenames=filenames)
                            scene.load([scenename])
                            print()
                            print(
                                f"{Fore.GREEN}Success! Continuing scene preparation..."
                            )
                        except KeyError:
                            pass
                        except OSError:
                            print()
                            print(
                                f"{Fore.RED}WARNING - Still having an issue with this file after "
                                f"downloading it again. Skipping..."
                            )
                            skip = True

                if not skip:
                    # Prep kwargs for resampling function
                    if self.satpy_cache_dir:
                        resampler_kwargs = {"cache_dir": self.satpy_cache_dir}
                    else:
                        resampler_kwargs = {}

                    # Determine lowest area in scene
                    _area_list = []
                    _key_list = []
                    for key in scene.keys():
                        _area = scene[key].attrs["area"].size
                        _area_list.append(_area)
                        _key_list.append(key)
                    idx = _area_list.index(min(_area_list))
                    minkey = _key_list[idx]

                    # Resample scene - default behavior is to try resampling at finest area, this
                    # fails for some scenes however, so if it does we'll fall back to the lowest
                    # area in the scene. Otherwise, use the area and resampler provided by the user
                    if not resample:
                        if bbox:
                            if self.satellite == "goes-east":
                                proj4_goes = (
                                    "+proj=geos +lon_0=-75 +h=35786023 +x_0=0 +y_0=0 +ellps=GRS80 +units=m "
                                    "+no_defs +sweep=x +type=crs"
                                )
                                _lon = "-75"
                            elif self.satellite == "goes-west":
                                proj4_goes = (
                                    "+proj=geos +lon_0=-137 +h=35786023 +x_0=0 +y_0=0 +ellps=GRS80 +units=m "
                                    "+no_defs +sweep=x +type=crs"
                                )
                                _lon = "-137"
                            bbox_proj_coords = gistools.transform_bbox(
                                bbox,
                                rasterio.crs.CRS.from_epsg(4326),
                                rasterio.crs.CRS.from_proj4(proj4_goes),
                            )
                            width = int(
                                abs((bbox_proj_coords[0] - bbox_proj_coords[2]))
                                / self.default_resample_res
                            )
                            height = int(
                                abs((bbox_proj_coords[1] - bbox_proj_coords[3]))
                                / self.default_resample_res
                            )

                            custom_area = AreaDefinition(
                                "goes_custom",
                                "bbox-bounded goes area",
                                "proj_id_1",
                                {
                                    "ellps": "GRS80",
                                    "h": "35786023",
                                    "lon_0": _lon,
                                    "no_defs": "None",
                                    "proj": "geos",
                                    "sweep": "x",
                                    "type": "crs",
                                    "units": "m",
                                    "x_0": "0",
                                    "y_0": "0",
                                },
                                width,
                                height,
                                bbox_proj_coords,
                            )
                            new_scene = scene.resample(custom_area)
                        else:
                            try:
                                new_scene = scene.resample(
                                    scene.finest_area(),
                                    resampler="native",
                                    **resampler_kwargs,
                                )
                            except ValueError:
                                new_scene = scene.resample(
                                    scene[minkey].attrs["area"],
                                    resampler="nearest",
                                    **resampler_kwargs,
                                )
                    else:
                        area = resample[0]
                        resampler = resample[1]

                        if area == "finest":
                            new_scene = scene.resample(
                                scene.finest_area(),
                                resampler=resampler,
                                **resampler_kwargs,
                            )
                        elif area == "coarsest":
                            new_scene = scene.resample(
                                scene.coarsest_area(),
                                resampler=resampler,
                                **resampler_kwargs,
                            )
                        elif area == "lowest area":
                            new_scene = scene.resample(
                                scene[minkey].attrs["area"],
                                resampler=resampler,
                                **resampler_kwargs,
                            )
                        elif area == "none":
                            new_scene = scene
                        else:
                            new_scene = scene.resample(
                                area, resampler=resampler, **resampler_kwargs
                            )

                    # Crop scene to bbox if provided
                    # if bbox:
                    #    new_scene_crop = new_scene.crop(ll_bbox=bbox)
                    #    new_scene = new_scene_crop.resample(resampler='native')

                    # If keep_filenames, the original template found in the nc data file will be
                    # used for saving the images. Otherwise, an isoformat time string will be used
                    # as the filename template
                    if keep_filenames:
                        repl = filenames[0].split("\\")[-1].split("_")[1]
                        substr = repl.split("-")
                        substr = "-".join(substr[0:-1])
                        svname = filenames[0].split("\\")[-1].replace(repl, substr)
                        svname = svname.rstrip(".nc")
                    else:
                        tstamp = self._get_timestamp_from_filename(filenames[0])
                        if tzinfo:
                            tstamp = pytz.utc.localize(tstamp).astimezone(tzinfo[0])
                            suffix = tzinfo[1]
                        else:
                            tstamp = pytz.utc.localize(tstamp)
                            suffix = "UTC"

                        svname = (
                            tstamp.isoformat()[0:-6].replace("T", " ") + " " + suffix
                        )
                        svname = svname.replace(":", "_")

                    # Add coastlines to the scene using options stored in the instance
                    if coastlines:
                        kwargs["overlay"] = self.coastoptions

                    # Save the scene to the specified output format. The temporary directory
                    # is used for saving. The directory object will be returned by the
                    # function. It will be necessary to then save the files in a permanent folder
                    # and delete the temporary directory by calling its cleanup() function
                    if output_format == "simple_image":
                        # Write geotiff data to metadata.json for future use
                        if counter == 0:
                            tmpfile = tempfile.NamedTemporaryFile(
                                "w+b", suffix=".tif", delete=False
                            )
                            tmpfile.close()
                            _tifout = tmpfile.name
                            new_scene.save_dataset(
                                scenename,
                                writer="geotiff",
                                filename=_tifout,
                                fill_value=False,
                                **kwargs,
                            )
                            with rasterio.open(_tifout, "r") as src:
                                profile = src.profile

                            with open(str(folder_path / "metadata.json"), "r+") as f:
                                _dict = json.load(f)
                                _dict["geodata"] = profile.data
                                crs_str = _dict["geodata"]["crs"].wkt
                                _dict["geodata"].pop("crs", None)
                                _t = _dict["geodata"].pop("transform", None)
                                _dict["geodata"]["transform"] = [
                                    _t.a,
                                    _t.b,
                                    _t.c,
                                    _t.d,
                                    _t.e,
                                    _t.f,
                                ]
                                _dict["geodata"]["crs"] = crs_str
                                f.seek(0)
                                json.dump(_dict, f, indent=4)

                            try:
                                os.unlink(_tifout)
                            except PermissionError:
                                pass

                        counter += 1
                        svname += ".png"
                        new_scene.save_dataset(
                            scenename,
                            writer="simple_image",
                            filename=str(folder_path / svname),
                            fill_value=False,
                            **kwargs,
                        )
                    elif output_format == "geotiff":
                        svname += ".tif"
                        # These settings are specific to the ABI-L2-FDC product
                        if scenename == "Power" or scenename == "Temp":
                            img = to_image(new_scene[scenename])

                            cmap = cm.Colormap(
                                (1, (1.0, 0.729, 0.729, 0.6)),
                                (10, (0.988, 0.51, 0.51, 0.6)),
                                (100, (0.988, 0.239, 0.239, 0.6)),
                                (200, (1.0, 0.0, 0.0, 0.6)),
                            )
                            img.convert("LA")
                            # img.apply_pil(convert, output_mode="LA", fun_args="LA")
                            img.colorize(cmap)
                            img.rio_save(str(folder_path / svname))
                        else:
                            new_scene.save_dataset(
                                scenename,
                                writer="geotiff",
                                filename=str(folder_path / svname),
                                keep_palette=True,
                                **kwargs,
                            )

                        if counter == 0:
                            with rasterio.open(str(folder_path / svname)) as src:
                                profile = src.profile

                            with open(str(folder_path / "metadata.json"), "r+") as f:
                                _dict = json.load(f)
                                _dict["geodata"] = profile.data
                                crs_str = _dict["geodata"]["crs"].wkt
                                _dict["geodata"].pop("crs", None)
                                _t = _dict["geodata"].pop("transform", None)
                                _dict["geodata"]["transform"] = [
                                    _t.a,
                                    _t.b,
                                    _t.c,
                                    _t.d,
                                    _t.e,
                                    _t.f,
                                ]
                                _dict["geodata"]["crs"] = crs_str
                                f.seek(0)
                                json.dump(_dict, f, indent=4)

                        counter += 1

                    with open(str(folder_path / "timestamps.csv"), "a") as f:
                        tstr = self._get_timestamp_from_filename(
                            filenames[0]
                        ).isoformat()
                        fname = svname
                        f.write(f"{tstr},{fname}\n")

                    # Cleanup
                    if delete_data:
                        for f in filenames:
                            os.remove(f)

            else:
                # Need to add other readers
                pass

        return folder_path

    def set_coastlines_options(
        self, path=None, color=(235, 235, 71), res="h", width=2, **kwargs
    ):
        if path:
            self.coastoptions["coast_dir"] = path
        self.coastoptions["color"] = color
        self.coastoptions["resolution"] = res
        self.coastoptions["width"] = width
        if kwargs:
            for key in kwargs:
                self.coastoptions[key] = kwargs[key]

    def set_satpy_cache_dir(self, path):
        self.satpy_cache_dir = path
        self.satpy_cache = True

    def turn_on_satpy_debug(self):
        self.satpy_debug = True
        return

    def turn_off_satpy_debug(self):
        self.satpy_debug = False
        return


class GoesAnimator(GoesBase):
    def __init__(self, sat, region, product, base_dir=None):
        super().__init__(sat, region, product)

        if not base_dir:
            print(
                f"{Fore.RED} WARNING - No base directory provided. Videos will be saved to current "
                f"working directory. Any existing data files will not be available for "
                f"compositing/animating."
            )
            self.base_dir = Path.cwd()
            self.imgsvpath = self.base_dir / "GoesVideo - Images"
            self.vidsvpath = self.base_dir / "GoesVideo - Videos"
            self.datapath = self.base_dir / "GoesVideo - Data"
            self.tmpdir = self.base_dir / "GoesVideo - temp"
            self.tmpimgpath = self.tmpdir / "Scenes"
            self.compositor = GoesCompositor(sat, region, product)
        else:
            self.base_dir = Path(base_dir)
            self.imgsvpath = self.base_dir / "Scenes"
            self.vidsvpath = self.base_dir / "Videos"
            self.datapath = self.base_dir / "goesdata"
            self.tmpdir = self.base_dir / "temp"
            self.tmpimgpath = self.tmpdir / "Scenes"
            self.compositor = GoesCompositor(
                sat, region, product, base_dir=self.base_dir
            )

        self.base_dir.mkdir(exist_ok=True)
        self.imgsvpath.mkdir(exist_ok=True)
        self.vidsvpath.mkdir(exist_ok=True)
        self.datapath.mkdir(exist_ok=True)
        self.tmpdir.mkdir(exist_ok=True)
        self.tmpimgpath.mkdir(exist_ok=True)

        self.existing_img_dirs = []
        self.existing_img_scenes = []

        self.default_img_height = 1080
        self.default_img_width = 1920

        self.timestamp_options = None

    def preview(
        self,
        scenename=None,
        utctime=None,
        use_cached=True,
        use_image_file=None,
        display=True,
        **kwargs,
    ):
        """
        Generates a quick preview image. Accepts kwargs for the create_composites and create_video
        functions. This function is useful for sizing and positioning text and other annotations
        prior to compositing the entire time series of images. It is advisable to use the same
        resolution as that intended for the final output video when using this function.

        @param scenename: (str) satpy scene name
        @param utctime: (str) isoformat datetime target for image in UTC; if not given the function
                              will use the current time as search target. Note that the returned
                              image will be within a window of +/- 2 hours of the target.
        @param use_cached: (bool) if true, will attempt to find a pre-existing local dataset/image
                                  to use for generating the preview image
        @param use_image_file: (str) if desired, a specific image can be used for generating the
                                     preview image by supplying a filepath
        @param display: (bool) if false, the preview image will not be automatically displayed
        @param kwargs: (dict) image-related keywords for 'create_video' and/or compositor keywords
        @return: PIL Image
        """
        # Unpack kwargs for image operations. Remaining kwargs will be passed on to the compositor
        timestamps = kwargs.pop("timestamps", None)
        cmap = kwargs.pop("cmap", None)
        text = kwargs.pop("text", None)
        res = kwargs.pop("res", "auto")
        arrow = kwargs.pop("arrow", None)
        circle = kwargs.pop("circle", None)

        # Check if anything is left to pass on to compositor
        if kwargs:
            compositor_reqd = True
        else:
            compositor_reqd = False

        if not use_image_file:
            # Get the search timestamp
            if utctime:
                tsearch = datetime.fromisoformat(utctime)
            else:
                tsearch = datetime.utcnow()

            # Determine bands required to produce scene
            if scenename:
                req_bands = self._get_scene_bands(scenename)
            else:
                req_bands = "C01"

            # If use_cached then need to search through temp folder for data and/or images that
            # can be used
            datastore = {}
            imgstore = []
            ideal_data = False
            ideal_imgs = False
            if use_cached:
                for folder in self.tmpimgpath.iterdir():
                    if folder.is_dir():
                        try:
                            with open(folder / "metadata.json") as file:
                                _json = json.load(file)

                            scene = _json["Scene"]
                            start_time = datetime.fromisoformat(_json["Start_Time"])
                            end_time = datetime.fromisoformat(_json["End_Time"])
                            interval = _json["Interval"]
                            if scene == scenename:
                                if start_time <= tsearch <= end_time:
                                    if interval < 120:
                                        for imgfile in folder.glob("*.png"):
                                            tfile = self._get_timestamp_from_filename(
                                                str(imgfile)
                                            )
                                            if (
                                                tfile - tsearch
                                            ).total_seconds() / 60 <= 120:
                                                imgstore.append(imgfile)
                                                ideal_imgs = True
                        except FileNotFoundError:
                            pass
                        except KeyError:
                            pass

                # Loop through existing data files checking to see if there are any that are within
                # 2 hours of the requested time. Also, ensure that all data files required to
                # produce scene are available. If they are, then set the ideal_data flag to True

                req_bands_pop = req_bands
                data_files = self.tmpdir.glob("*.nc")
                if data_files:
                    for f in data_files:
                        tstamp = self._get_timestamp_from_filename(str(f))
                        ch = self._get_band_from_filename(str(f))
                        if ch not in datastore:
                            datastore[ch] = []
                        if (tstamp - tsearch).total_seconds() / 60 < 120:
                            try:
                                idx = req_bands.index(ch)
                                req_bands_pop.pop(idx)
                                if len(datastore[ch]) == 0:
                                    datastore[ch].append(f)
                            except ValueError:
                                pass
                            except IndexError:
                                ideal_data = True
                                break

                # Get ideal image if it is available and compositing is not required. Otherwise,
                # call the compositor to generate an image
                imgname = None
                img = None
                if ideal_imgs and not compositor_reqd:
                    imgname = imgstore[0]
                    img = Image.open(imgstore[0])
                else:
                    if ideal_data:
                        tmpdir = self.compositor.composites_from_files(
                            scenename, datastore, **kwargs
                        )
                        tmpdirpath = Path(tmpdir.name)
                        for file in tmpdirpath.iterdir():
                            if file.is_file():
                                shutil.copyfile(str(file), self.tmpimgpath / file.name)
                                try:
                                    os.unlink(str(file))
                                except PermissionError:
                                    pass
                        os.rmdir(tmpdirpath)
                        imgname = list(self.tmpimgpath.glob("*.png"))[0]

                        img = Image.open(imgname)
                    else:
                        if not utctime:
                            _tnow = datetime.now(pytz.utc) - timedelta(minutes=60)
                            start_time = _tnow.isoformat().split(".")[0]
                            end_time = _tnow + timedelta(minutes=30)
                            end_time = end_time.isoformat().split(".")[0]
                        else:
                            if isinstance(utctime, str):
                                start_time = utctime
                                _t = datetime.fromisoformat(start_time)
                                end_time = _t + timedelta(minutes=10)
                                end_time = end_time.isoformat()

                            else:
                                raise exceptions.InvalidArgumentError(
                                    " Time should be an isoformat string."
                                )

                        gc = GoesCompositor(
                            self.satellite,
                            self.region,
                            self.product[:-1],
                            base_dir=self.tmpdir,
                        )
                        gc.create_composites(
                            scenename,
                            start_time=start_time,
                            end_time=end_time,
                            force=True,
                            keep_filenames=True,
                            **kwargs,
                        )
                        subdirs = []
                        for sub in self.tmpimgpath.iterdir():
                            if sub.is_dir():
                                subdirs.append(str(sub))
                        latest = Path(max(subdirs, key=os.path.getmtime))
                        imgname = list(latest.glob("*.png"))[0]

                        img = Image.open(str(imgname))

        if use_image_file:
            img = Image.open(use_image_file)
            imgname = use_image_file

        # Pack kwargs for img modification
        img_kwargs = {}

        if timestamps:
            img_kwargs["timestamps"] = {}
            img_kwargs["timestamps"]["label"] = timestamps.pop("label")
            img_kwargs["timestamps"]["tdict"] = timestamps

        if res == "auto":
            res = self._get_optimal_resolution(img)

        img_kwargs["res"] = res
        img_kwargs["cmap"] = cmap
        img_kwargs["text"] = text
        img_kwargs["arrow"] = arrow
        img_kwargs["circle"] = circle

        # Apply modifications to image
        img = utils.modify_image(img, **img_kwargs)

        if display:
            img.show()

        return img

    def create_video(
        self,
        scenename,
        start_time=None,
        end_time=None,
        from_existing_imgs=False,
        interval=5,
        bbox=None,
        timestamps=False,
        coastlines=False,
        cmap=None,
        text=None,
        res="auto",
        fps=20,
        codec="mpeg4",
        delete_images=False,
        delete_data=False,
        force=False,
        **kwargs,
    ):
        """
        Creates a video from GOES image composites. Function can try to find local and/or remote
        data files needed to produce the specified scene, or can look in local folders for existing
        composite images by using the 'from_existing_imgs' argument. The function may produce
        multiple videos if the 'from_existing_imgs' argument is used.

        **NOTE: This function relies on the base class having access to a file entitled
        'abibands.yaml'. This file is a simplified version of the satpy configuration file entitled
        'abi.yaml'. The user may customize the 'abibands.yaml' to match any customizations made to
        'abi.yaml'. The former file simply maps the required ABI bands for each scene.

        @param scenename: (str) satpy scenename or 'available' (only when using
                                'from_existing_imgs')
        @param start_time: (str) isoformat datetime for GOES imagery in video
        @param end_time: (str) isoformat datetime for GOES imagery in video
        @param from_existing_imgs: (bool) If true, will automatically search through subfolders in
                                          the Images directory to determine which ones have not yet
                                          been converted to videos. Then, the function attempts to
                                          generate videos corresponding to the requested scene
                                          using the existing images. If the requested scene cannot
                                          be made using the existing images, the user will be
                                          prompted to answer if they would like all available
                                          scenes to be made instead. The user can also request all
                                          available scenes be made from existing images using
                                          'available' for the scenename parameter.
        @param interval: (int) time interval between GOES images in video in minutes
        @param bbox: (tup) tuple of floats specifying crop region in latitude-longitude coordinates
                           (west, south, east, north)
        @param timestamps: (bool) Add timestamp to each video frame. The set_timestamp_options()
                                  function must be called prior to creating the video.
        @param coastlines: (bool) Show coastlines on images used for the video. Coastline options
                                  can be changed from their default values via the accessor
                                  GoesAnimator.GoesCompositor.set_coastlines_options() NOTE: pycoast
                                  package may need to be installed in order for this to work
        @param cmap: (str) Matplotlib colormap name or path to a json file containing a colormap
                           produced using the online app at https://sciviscolor.org/color-moves-app/
        @param text: (dict) add custom text to each frame of the video. Dictionary should be
                            provided with the same
                            keywords used for the timestamps parameter except without the 'timezone'
                            key. The text string to be displayed should be assigned to a new 'label'
                            key.
        @param res: (str or tup) desired resolution for the video. Can be a tuple of ints (w, h)
                                 specifying width and height or 'auto' to optimally size the video
                                 for most displays while maintaining the aspect ratio of the images
                                 or 'full' to retain the resolution of the input images
        @param fps: (int) frames per second for the video
        @param codec: (str) FFMPEG codec (e.g. 'mpeg4', 'avi')
        @param delete_images: (bool) If true, function will delete all images used to produce the
                                     video upon completion
        @param delete_data: (bool) If true, function will delete all underlying datasets used to
                                   produce the video upon completion
        @param force: (bool) If true, any user inputs will be supressed and the function will be
                             forced to completion
        @param kwargs: (dict) function keyword arguments and/or kwargs to be passed to
                              GoesCompositor.create_composites
        @return: None
        """
        # Check timestamps
        if timestamps:
            if not self.timestamp_options:
                raise exceptions.InvalidArgumentError(
                    "Must set timestamp options using 'set_timestamp_options"
                )

        # Check to see if user wants to convert existing images to video or make
        # new images using the compositor. Program exits if conflicting keyword
        # arguments were passed for the case when 'from_existing_imgs' is True

        if from_existing_imgs:
            if not isinstance(from_existing_imgs, bool):
                raise TypeError
            else:
                _exit = False
                suffix = ""
                if start_time:
                    suffix += " start_time,"
                    _exit = True
                if end_time:
                    suffix += " end_time,"
                    _exit = True
                if bbox:
                    suffix += " bbox,"
                    _exit = True
                if coastlines:
                    suffix += " coastlines,"
                    _exit = True
                if _exit:
                    raise exceptions.InvalidArgumentError(
                        f"Cannot pass arguments {suffix.rstrip(',')} for "
                        f"compositor when using 'from_existing_imgs'."
                    )

        # If user wants to use existing image folders, need to check that folder exists in the
        # current directory. Also, need to make sure that the requested scene can be generated.
        img_folders = []
        if from_existing_imgs:
            self._populate_existing_imgs()
            if self.existing_img_dirs and scenename not in self.existing_img_scenes:
                print()
                if not force:
                    ask = input(
                        f"{Fore.RED}ERROR - Requested scene is not available for any of the "
                        f"existing image folders. \nWould you like to continue and produce videos "
                        f"for all available scenes in the existing \nimage folders? (Y/N): "
                    )
                else:
                    ask = "Y"

                if ask.upper() == "Y":
                    scenename = "available"
                else:
                    print()
                    print(f"{Fore.RED}Exiting...")
                    sys.exit(0)
            else:
                img_folders = []
                for i, _scene in enumerate(self.existing_img_scenes):
                    if _scene == scenename:
                        img_folders.append(self.existing_img_dirs[i])

        # Add compositor keywords to kwargs

        if kwargs:
            if bbox:
                kwargs["bbox"] = bbox
            if coastlines:
                kwargs["coastlines"] = coastlines
        else:
            kwargs = {}

            if bbox:
                kwargs["bbox"] = bbox
            if coastlines:
                kwargs["coastlines"] = coastlines

        # Handle cmap argument

        if cmap:
            if ".json" in cmap:
                cmap = utils.build_cmap(cmap)
            else:
                cmap = colormaps.get_cmap(cmap)

        if not from_existing_imgs:
            _exit = False

            # Create composite images - if data is not available, compositor will download them
            if scenename != "available":
                img_folder = self.compositor.create_composites(
                    scenename, start_time, end_time, interval=interval, **kwargs
                )
                img_folders = [img_folder]

        if not img_folders:
            print()
            print(
                f"{Fore.RED} Could not find any image folders to convert to video. Exiting..."
            )
            sys.exit(0)

        # Create video folders

        vid_folders = []
        ask_done = False
        for i, folder in enumerate(img_folders):
            vid_folders.append(self.vidsvpath / folder.stem)
            try:
                folder.mkdir(exist_ok=False)
            except FileExistsError:
                if from_existing_imgs:
                    vfolder = self.vidsvpath / folder.stem
                    files = (
                        list(vfolder.glob("*.avi"))
                        + list(vfolder.glob("*.mp4"))
                        + list(vfolder.glob("*.wmv"))
                    )
                    if files and not ask_done:
                        if not force:
                            ask = input(
                                f"{Fore.RED} WARNING - Some videos already exist. Do you still want "
                                f"to proceed and overwrite them? (Y/N): "
                            )
                        else:
                            ask = "Y"

                        ask_done = True
                        if ask.upper() == "Y":
                            pass
                        else:
                            print()
                            print(f"{Fore.RED}Exiting...")
                            sys.exit(0)

        # Create video filenames
        fnames = []
        if from_existing_imgs:
            for folder in img_folders:
                try:
                    with open(str(folder / "metadata.json"), "r") as file:
                        _json = json.load(file)

                    sat = _json["Satellite"]
                    scenename = _json["Scene"]
                    start = _json["Start_Time"]
                    end = _json["End_Time"]
                    fname = (
                        sat
                        + "_"
                        + scenename
                        + "_"
                        + start.replace(":", "_")
                        + "_thru_"
                        + end.replace(":", "_")
                    )
                    fnames.append(fname)
                except FileNotFoundError:
                    print()
                    print(
                        f"{Fore.RED}WARNING - Could not find metadata.json in existing image path: "
                        f"{str(folder)}"
                    )

        else:
            fname = (
                self.satellite
                + "_"
                + scenename
                + "_"
                + start_time.replace(":", "_")
                + "_thru_"
                + end_time.replace(":", "_")
            )
            fnames.append(fname)

        # Create video
        print()
        print(f"{Fore.GREEN}Preparing video frames...")

        # Temporary folder used to store images to be used for the video
        tmpfolder = tempfile.TemporaryDirectory()
        tmpfolderpath = Path(tmpfolder.name)

        for k, folder in enumerate(img_folders):
            img_filenames = list(folder.glob("*.png"))
            img_filenames = sorted(img_filenames, key=lambda item: item.name)
            timestamps_file = list(folder.glob("*.csv"))[0]
            tstamps, data_filenames = self._get_timestamps(timestamps_file)

            img_array = []
            _str = f"Image Folder {str(k + 1)} of {str(len(img_folders))}: "

            for i, file in enumerate(tqdm(img_filenames, colour="green", desc=_str)):
                # Perform image modifications

                size = os.path.getsize(file)
                if Image.MAX_IMAGE_PIXELS:
                    if size > Image.MAX_IMAGE_PIXELS:
                        Image.MAX_IMAGE_PIXELS = None

                img = Image.open(file)

                if res == "auto":
                    width, height = self._get_optimal_resolution(img)
                    img = img.resize((width, height))
                elif res == "full":
                    pass
                else:
                    img = img.resize((res[0], res[1]))

                if cmap:
                    img = img.convert("L")
                    imgarr = np.array(img) / 255
                    img = Image.fromarray(np.uint8(cmap(imgarr) * 255))

                if timestamps:
                    tdict = deepcopy(self.timestamp_options)
                    img = utils.add_timestamps(img, tstamps[i], **tdict)

                if text:
                    img = utils.add_text(img, **text)

                if codec == "rawvideo" or codec == "png":
                    img = utils.convert_color(img)

                imgname = tmpfolderpath / ("image_" + str(i) + ".png")
                img.save(imgname)
                img_array.append(imgname)

            print()
            print(f"{Fore.GREEN}Creating video...")

            if codec == "mpeg4" or codec == "libx264":
                ext = ".mp4"
            elif codec == "rawvideo" or codec == "png":
                ext = ".avi"
            elif codec == "libvpx":
                ext = ".wmv"
            else:
                ext = ".mp4"

            # fname = fnames[k] + ext
            vid_folders[k].mkdir(exist_ok=True)
            fname = "video" + ext

            imgarr = [str(f) for f in img_array]
            clip = ImageSequenceClip(imgarr, fps=fps)
            clip.write_videofile(str(vid_folders[k] / fname), codec=codec)
            time.sleep(0.5)

            # Cleanup
            if delete_images:
                shutil.rmtree(str(folder))

            if delete_data:
                for f in data_filenames:
                    os.remove(f)

    def list_existing_image_dirs(self):
        """Prints a list of existing directories found to contain image folders within the base
        directory"""
        for item in self.existing_img_dirs:
            print(item)
        return self.existing_img_dirs

    def set_timestamp_options(
        self,
        fontpath,
        position="upper-left",
        fontcolor=(0, 0, 0),
        fontsize=None,
        timezone=None,
    ):
        """
        Sets timestamp options for the video
        @param fontpath: (str) path to ttf file (required)
        @param position: (tup) or (str) position of timestamp label can be specified by
                         giving a precise (x,y) tuple of pixel location or by using
                         'upper-left', 'upper-center', 'upper-right', 'lower-left',
                         'lower-center', 'lower-right'
        @param fontcolor: (tup) RGB(A) color of font given in range of 0-255
        @param fontsize: (int) fontsize in pixels or None for auto-sizing
        @param timezone: (tup) ((obj) pytz timezone, (str) timezone abbreviation (e.g. 'CDT')
        @return: None
        """
        if not isinstance(self.timestamp_options, dict):
            self.timestamp_options = {}

        self.timestamp_options["fontpath"] = fontpath
        self.timestamp_options["position"] = position
        self.timestamp_options["fontcolor"] = fontcolor
        self.timestamp_options["fontsize"] = fontsize
        self.timestamp_options["timezone"] = timezone

        return

    @staticmethod
    def _get_timestamps(csvfile):
        """Extracts timestamps for images listed in the timestamps.csv file found in existing image
        directories"""
        tstamps_list = []
        filenames_list = []
        with open(str(csvfile), newline="") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                tstr = row[0]
                tstamps_list.append(datetime.fromisoformat(tstr))
                filenames_list.append(row[1])

        return tstamps_list, filenames_list

    def _get_optimal_resolution(self, img):
        """Determines the optimal display resolution for the video"""
        width, height = img.size
        aspect = float(width / height)

        new_width = int(aspect * self.default_img_height)
        new_height = self.default_img_height

        if new_width > self.default_img_width:
            factor = float(new_width / self.default_img_width)
        elif new_height > self.default_img_height:
            factor = float(new_height / self.default_img_height)
        else:
            factor = 1

        new_width = int(new_width / factor)
        new_height = int(new_height / factor)

        return new_width, new_height

    def _populate_existing_imgs(self):
        """Searches the base_dir Images folder for existing image folders"""
        self.existing_img_dirs = []
        self.existing_img_scenes = []
        for path in self.imgsvpath.iterdir():
            if path.is_dir():
                # Open metadata json to see if image content of folder matches
                # the specification in class constructor
                try:
                    with open(str(path / "metadata.json"), "r") as file:
                        _json = json.load(file)

                    sat = _json["Satellite"]
                    region = _json["Region"]
                    product = _json["Product"]
                    scene = _json["Scene"]

                    if self.satellite == sat:
                        if self.region == region:
                            if self.product == product:
                                self.existing_img_dirs.append(path)
                                self.existing_img_scenes.append(scene)

                except FileNotFoundError:
                    print()
                    print(
                        f"{Fore.RED}WARNING - Could not find metadata.json in existing image path: "
                        f"{str(path)}"
                    )

        return
