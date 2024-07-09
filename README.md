# GoesVideo
___
This package is a [Satpy](https://github.com/pytroll/satpy) wrapper that facilitates and streamlines the 
downloading, compositing and animating of imagery from the GOES-East
and GOES-West satellites. Basic annotation tools are also
included to simplify the preparation of images and videos for presentations.

Note, that the current version has only been tested using Level 2 CMI data. Results may
be unexpected with other GOES products.

**New in version 2024.1.0**: Overlays can now be generated using a base image set (e.g. GOES-18 visible imagery)
and one or more overlay sets (e.g. fire detection). This tool is general in scope and does not require GOES data
products in order to work. 

## Installation
___

GoesVideo can be installed from PyPI with pip:

```python
pip install goesvideo
```

## Usage
___
There are three main classes that can be invoked depending on the needs
of the user. To download GOES data files from [AWS](https://registry.opendata.aws/noaa-goes/), use the
`GoesDownloader` class:

```python
import goesvideo

dl = goesvideo.GoesDownloader('goes-east', 'full', 'ABI-L2-CMIP', base_dir=mypath)
dl.download_all('2023-05-01 10:30:00', '2023-05-10 12:17:00', interval=1440)
```
This example will download all full-disk images from the GOES-East product ABI-L2-CMIP between
May 5th and May 10th, 2023 at an interval of 1440 minutes.

Compositing of imagery to produce Satpy scenes can be accomplished using the `GoesCompositor` class:

```python
import goesvideo

compositor = goesvideo.GoesCompositor('goes-west', 'conus', 'ABI-L2-CMIP', base_dir=mypath)
compositor.create_composites('true_color', '2023-05-01 10:30:00', '2023-05-10', interval=30)
compositor.create_composites('color_infrared', '2023-07-01 08:00:00', '2023-07-03 13:00:00', 
                             interval=180,
                             bbox=(-117.3, 30.1, -93.0, 43.5),
                             coastlines=True,
                             resampling=('finest', 'native'),
                             folder_name='July-cropped')
```
The compositor will invoke the downloader as needed to retrieve the remote datasets necessary to produce the desired
Satpy scene. For the ABI instrument, Satpy includes a configuration file named `abi.yaml` that defines all of the 
available scenes. Note that the `goesvideo` package also has a configuration file
`abibands.yaml` that is intended to mirror the band requirements in the Satpy file. It is advisable to update
this file anytime updates are made to the Satpy file.

Videos can be produced using the `GoesAnimator` class:

```python
import goesvideo

animator = goesvideo.GoesAnimator('goes-west', 'conus', 'ABI-L2-CMIP', base_dir=mypath)
animator.create_video('true_color', start_time='2023-07-10 12:34:00',
                      end_time='2023-07-13 10:10:00',
                      interval=15,
                      timestamps=True,
                      coastlines=True,
                      res=(1920, 1080),
                      fps=20)
animator.create_video('C01', start_time='2023-07-10 12:34:00',
                      end_time='2023-07-13 10:10:00',
                      interval=15,
                      cmap='Spectral')
animator.create_video('color_infrared', from_existing_imgs=True)
```
The animator will invoke the compositor and downloader as necessary to produce the desired video.
In the last example, the animator will search for matching composites that
have previously been generated and saved in the base directory.

Since compositing and animating GOES imagery can be a lengthy process, a preview function has been provided to ensure the
video will turn out as expected:

```python
import goesvideo

animator = goesvideo.GoesAnimator('goes-west', 'conus', 'ABI-L2-CMIP', base_dir=mypath)
animator.preview('true_color', use_cached=True, **kwargs)
```
In this example, a preview will be generated for the true color composite. The kwargs can contain any of the compositor 
keywords and/or annotation-related keywords such as `text`, `circle` and `arrow`.

If desired, output videos can be annotated post-production using the experimental
`GoesClip` class. 


Finally, a note on folder structure. The `base_dir` is either set by the user or defaults to the current working directory.
The folder is then automatically setup as follows:

```
base_dir
|
|---goesdata [contains NC data files]
|
|---Images [contains composited images]
|  |
|  |---Image subfolder #1 (random or user-provided)
|  |  |
|  |  |---image1.png
|  |  |---image2.png
|  |  |---...
|  |  |---metadata.json
|  |  |---timestamps.csv      
|  |
|  |---Image subfolder #2 (random or user-provided)
|  |  |
|  |  |---image1.png
|  |  |---image2.png
|  |  |---...
|  |  |---metadata.json
|  |  |---timestamps.csv
|  |
|  |--- ...
|  
|---Videos [contains videos]
   |
   |---Video subfolder #1 (same name as Image subfolder)
   |  |
   |  |---video.mp4
   |
   |---Video subfolder #2 (same name as Image subfolder)
   |  |
   |  |---video.mp4
   |
   |---...
```
To ensure proper functioning of this package, this folder structure
should not be modified. Also, the `metadata.json` and `timestamps.csv` files
added to each image subfolder should not be deleted or moved.

For additional information, please see the docstrings in the source code.
