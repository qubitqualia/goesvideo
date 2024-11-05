from pathlib import Path
import goesvideo
from goesvideo.utils import gistools

"""
This demo creates a video clip of the Hawaiian Islands region using data from GOES-West
The compositor selected for this video is 'true_color_with_night_ir', which is a standard compositor defined in
the Satpy file etc/composites/abi.yaml and is also included in the GoesVideo file etc/abibands.yaml. If custom 
compositors are defined by the user in Satpy, the user should be sure to also update etc/abibands.yaml.
options in abi.yaml.
"""

#---- Custom options to be set by user
basedir = str(Path.home()) + '/goesimagery'
fontpath = str(Path.home()) + '/.local/share/fonts/meslolgldz-nerd-font/MesloLGSDZNerdFont-Bold.ttf'  # Required for timestamps
coastpath = str(Path.home()) + '/goes-workspace/coastlines'   # Required for coastlines/borders; this should be the root path
                                                              # to the GHHSG shapefiles that can be downloaded here:
                                                              # https://www.soest.hawaii.edu/pwessel/gshhg/

#---- Coords for bbox in DMS units
uldms = ('24d56m50s N', '165d08m15s W')
lrdms = ('15d19m31s N', '150d51m23s W')

#---- Convert to DD units for bbox
uldd = gistools.convert_dms_to_dd(uldms)
lrdd = gistools.convert_dms_to_dd(lrdms)
bbox = (uldd[1], lrdd[0], lrdd[1], uldd[0])

#---- Create animator
animator = goesvideo.GoesAnimator('goes-west', 'full', 'ABI-L2-CMIP',
                                  base_dir=basedir)

# Configure coastline/border options for video
coastopts = {'level_coast': 1, 'level_borders': 2}
animator.compositor.set_coastlines_options(path=coastpath, **coastopts)

# Create video
animator.create_video('true_color_with_night_ir',
                      start_time='2024-10-12 17:00:00',   #Supplied as UTC time
                      end_time='2024-10-12 23:00:00',     #Supplied as UTC time
                      bbox=bbox,
                      folder_name='Hawaii_Sample_Scene',
                      interval=1,
                      timestamps=False,
                      coastlines=True,
                      res='auto',
                      codec='rawvideo',
                      fps=20)



