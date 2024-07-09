import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from goesvideo import GoesDownloader


def test_download():
    # Create temp folder
    tmpfolder = tempfile.TemporaryDirectory()

    # Intervals to test
    intervals = [10, 1440, 10080]
    nfiles = 3

    # Scenes to test
    scene = "true_color"

    # Satellites to test
    sat = "goes-east"

    # Regions to test
    region = "full"

    # Product to test
    product = "ABI-L2-CMIP"

    # Define start and end times
    start_time = "2023-01-01 00:00:00"
    end_times = []
    tstart = datetime.fromisoformat(start_time)
    for i in range(len(intervals)):
        tend = tstart + timedelta(minutes=int(intervals[i] * nfiles * 1.05))
        end_times.append(tend.isoformat())

    needed_files = []
    sz = []
    chk = []

    # Initialize downloader
    dl = GoesDownloader(sat, region, product, base_dir=tmpfolder.name)

    for i, tint in enumerate(intervals):
        # Get list of files to download
        needed_files, sz, chk = dl.check_for_needed_files(
            start_time, end_times[i], scenename=scene, interval=tint
        )

        # Check file intervals
        tstamps = [dl._get_timestamp_from_filename(f) for f in needed_files]

        # Check bands
        req_bands = dl._get_scene_bands(scene)
        actual_bands = list(set([dl._get_band_from_filename(f) for f in needed_files]))
        remainder = list(set(actual_bands).difference(set(req_bands)))

        # Check satellite
        if sat == "goes-east":
            code = "G16"
        elif sat == "goes-west":
            code = "G18"

        satnames = [f for f in needed_files if code not in f]

        # Verifies the following:
        # - Correct interval between files
        # - Correct bands for the scene
        # - Correct satellite in all files

        tstamps = [t - timedelta(seconds=t.second) for t in tstamps]
        tstamps = sorted(list(set(tstamps)))

        assert (
            tint - 11
            <= abs(int((tstamps[1] - tstamps[0]).total_seconds() / 60))
            <= tint + 11
        )
        assert len(remainder) == 0
        assert len(satnames) == 0
        diff1 = np.array(
            list(
                abs((tstamps[j] - tstamps[j - 1]).total_seconds() / 60)
                for j in range(1, len(tstamps) - 1)
            )
        )
        diff2 = np.array(
            list(
                abs((tstamps[j + 1] - tstamps[j]).total_seconds() / 60)
                for j in range(1, len(tstamps) - 1)
            )
        )
        diff = abs(diff1 - diff2)
        assert all(diff <= 11)

    return


if __name__ == "__main__":
    sys.exit(pytest.main())
