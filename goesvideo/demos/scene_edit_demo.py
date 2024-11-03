from pathlib import Path
import pytz
from rasterio.enums import Resampling
from goesvideo.addons import sceneeditor

"""
This demo relies on images composited in 'create_video_goes_west_demo.py', which generates a scene of the
Hawaiian Islands using GOES-West imagery. 
"""

#---- Demo options
preview = False           # Generate a single preview image for each edit function
test_annotations = True   # Add annotations to image(s)
test_timestamps = True    # Add timestamps to image(s)
test_recrop = False       # Recrop image(s) to smaller region
test_resize = False       # Resize image(s)
test_reproject = False    # Reproject image(s) to EPSG:4326
test_tovideo = True       # Create video output

#----Setup editor
basedir = str(Path.home() / 'goesbase')
scene = 'Hawaii_Sample_Scene'
session_name = 'Hawaii_Sample_Scene_Edit_4'
editor = sceneeditor.GoesSceneEditor(basedir, scene, session_name='Hawaii_Sample_Scene_Edit_1')
fontpath = str(Path.home()) + '/.local/share/fonts/meslolgldz-nerd-font/MesloLGSDZNerdFont-Bold.ttf'
editor.set_font(fontpath, (255, 0, 1), 30)

if test_recrop:
    new_bbox = ('156d45m23s W', '20d29m22s N', '155d51m25s W', '21d08m50s N')
    editor.recrop(new_bbox, preview=preview)

if test_resize:
    editor.resize(1024, 760, resample=Resampling.nearest, preview=preview)

if test_reproject:
    editor.reproject("EPSG:4326", preview=preview)

if test_timestamps:
    editor.set_font(fontsize=60)
    editor.add_timestamps('upper-left',
                          pytz.timezone('US/Hawaii'), 'HDT',
                          preview=preview)

if test_annotations:
    if test_recrop:
      circleopts = {'radius': 7,
                    'fill': (0, 0, 255),
                    'outline': (0, 0, 0),
                    'width': 2}
      editor.add_annotation('circle', ('20d46m09s N', '156d15m15s W'),
                            label="Maui",
                            labelopts={'padding': (5, 5)},
                            **circleopts)
    else:
      editor.set_font(fontsize=50)
      circleopts = {'radius': 7,
                    'fill': (255, 0, 0),
                    'outline': (0, 0, 0),
                    'width': 2
                   }
      arrowopts = {'tiplength': 0.25,
                   'width': 3,
                   'color': (255, 0, 0)
                   }
      editor.add_annotation('circle', ('19d40m05s N', '155d27m38s W'),
                            label="Hawaii",
                            labelopts={'padding': (10, 10)},
                            **circleopts)
      editor.add_annotation('arrow', [('20d58m26s N', '158d03m37s W'), ('21d18m51s N', '157d52m02s W')],
                            label="Honolulu",
                            labelopts={'padding': (3, 3)},
                            **arrowopts)
      editor.add_annotation('text', 'upper-center',
                            label='Hawaiian Islands - GOES-West',
                            labelopts={'fontsize': 80, 'rotation': 0})

    editor.process_annotations(preview=preview)

if test_tovideo:
  editor.to_video(codec='rawvideo', fps=20)







