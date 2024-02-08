from osgeo import gdal
from pathlib import Path
from PIL import Image
from colorama import init as colorama_init
from colorama import Fore
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile

class Overlay:
    def __init__(self, baseimgpath, overlaypaths, outpath, out_crs='EPSG:4326'):
        self.baseimgpath = baseimgpath
        self.overlaypaths = overlaypaths
        self.outpath = Path(outpath)
        self.baseimg = Image.open(self.baseimgpath)
        self.outcrs = out_crs
        self.overlayimgs = []
        self.tmpdirs = []

        # Grab all geotiffs from provided paths, check crs and reproject to geographic, if necessary
        for p in self.overlaypaths:
            tmpdir = tempfile.TemporaryDirectory()
            self.tmpdirs.append(tmpdir)
            p = Path(p)
            files = p.glob('*.tif')
            if files:
                self.overlayimgs.append(files)
                for f in files:
                    self._reproject(f, tmpdir)

            else:
                print(f"{Fore.RED} No valid geotiffs found in path {str(p)}")

    def create_overlays(self, res=None, ):
        pass
    def _reproject(self, file, tmppath):
        with rasterio.open(str(file)) as src:
            if not src.crs.is_geographic:
                transform, width, height = calculate_default_transform(src.crs, {'init': self.outcrs}, src.width, src.height,
                                                                       *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({'crs': self.outcrs,
                               'transform': transform,
                               'width': width,
                               'height': height})

                fname = file.stem + file.suffix
                outpath = Path(tmppath.name)
                with rasterio.open(str(outpath / fname), 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, i),
                                  destination=rasterio.band(dst, i),
                                  src_transform=src.transform,
                                  src_crs=src.crs,
                                  dst_transform=transform,
                                  dst_crs=self.outcrs,
                                  resampling=Resampling.nearest)

