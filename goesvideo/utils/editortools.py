import shutil
import copy
import json
from datetime import datetime
from pathlib import Path

import matplotlib.colors as colors
import numpy as np
import pytz
from PIL import ImageDraw, ImageFont, Image
from colorama import Fore
from matplotlib import colormaps
from moviepy.editor import VideoFileClip
from moviepy.video.fx import freeze as frz

try:
    import cv2
except ModuleNotFoundError:
    print(
        f"{Fore.RED}WARNING - OpenCV package not detected. 'add_arrow' function will be unavailable."
    )


# ------ Image Annotations
def build_cmap(fname):
    """
    Builds a matplotlib cmap for a colormap specification generated using the app available at
    sciviscolor.org
    @param fname: (str) path to colormap json
    @return: (matplotlib.cm obj) colormap
    """
    try:
        with open(fname) as jsonfile:
            _json = json.load(jsonfile)
        points = _json["colormaps"][0]["points"]
        clist = []
        for p in points:
            c = (p["x"], (p["r"], p["g"], p["b"]))
            clist.append(c)

        cmap = colors.LinearSegmentedColormap.from_list("new", clist, N=256)
        cmap = cmap.reversed()
    except FileNotFoundError:
        cmap = colormaps.get_cmap(fname)

    return cmap


def copy_image(file, dstpath):
    """
    Copy an image to a new location
    @param file: (str) path to image file
    @param dstpath: (str) save path
    @return: (str) full save path to image
    """
    fp = Path(file)
    fname = dstpath / (fp.stem + fp.suffix)
    shutil.copy(file, str(fname))
    return fname


def add_text(img, **kwargs):
    """
    Add custom text to an image frame
    @param img: (PIL.Image) input image
    @param kwargs: (dict) see 'modify_image' function for options
    @return: (PIL.Image) output image
    """
    position = kwargs.get("position", "upper-left")
    fontcolor = kwargs.get("fontcolor", (0, 0, 0))
    fontpath = kwargs["fontpath"]
    fontsize = kwargs.get("fontsize", None)
    label = kwargs["label"]
    opacity = kwargs.get("opacity", 1)
    rotation = kwargs.get("rotation", 0)

    # Convert input image to RGB
    img = img.convert("RGB")

    # If fontsize not provided, attempt to scale to image resolution
    if not fontsize:
        fontsize = int(0.02 * img.height)

    font = ImageFont.truetype(fontpath, fontsize)

    # If opacity equals 1 then text can be added directly to the input image
    # Otherwise, need to create a new image layer for alpha compositing

    fontcolor = (fontcolor[0], fontcolor[1], fontcolor[2], int(opacity * 255))
    layercolor = (fontcolor[0], fontcolor[1], fontcolor[2], 0)
    txtimg = Image.new("RGBA", img.size, layercolor)
    draw = ImageDraw.Draw(txtimg)

    upper = 0
    lower = img.height
    left = 0
    right = img.width
    woffsetfrac = 0.01
    hoffsetfrac = 0.03
    txtlength = draw.textlength(label, font=font, font_size=fontsize)

    if position == "upper-left":
        x = left + int(woffsetfrac * img.width)
        y = upper + int(hoffsetfrac * img.height + fontsize)
    elif position == "upper-center":
        x = int((left + right) / 2.0)
        y = upper + int(hoffsetfrac * img.height + fontsize)
    elif position == "upper-right":
        x = right - int(woffsetfrac * img.width + txtlength)
        y = upper + int(hoffsetfrac * img.height + fontsize)
    elif position == "lower-left":
        x = left + int(woffsetfrac * img.width)
        y = lower - int(hoffsetfrac * img.height + fontsize)
    elif position == "lower-center":
        x = int((left + right) / 2.0)
        y = lower - int(hoffsetfrac * img.height + fontsize)
    elif position == "lower-right":
        x = right - int(woffsetfrac * img.width + txtlength)
        y = lower - int(hoffsetfrac * img.height + fontsize)
    else:
        x = position[0]
        y = position[1]

    draw.text((0, 0), label, fill=fontcolor, font=font, font_size=fontsize, **kwargs)
    img = img.convert("RGBA")

    # Get bounding box for text and crop to size before rotating
    bbox = draw.textbbox((0, 0), label, font=font, font_size=fontsize)
    txtimg = txtimg.crop(bbox)
    txtimg = txtimg.rotate(-rotation, expand=True)

    img.paste(txtimg, (int(x), int(y)), txtimg)

    img = img.convert("RGB")
    return img


def add_timestamps(img, label, **kwargs):
    """
    Add timestamp to image
    @param img: (PIL.Image) input image
    @param label: (str) datetime string to be added
    @param kwargs: (dict) see 'modify_image' function for options
    @return: (PIL.Image) output image
    """
    tzinfo = None
    if kwargs:
        tzinfo = kwargs.pop("timezone", None)  # tuple

    if tzinfo:
        _tz = tzinfo[0]
        _abbr = tzinfo[1]
        if not isinstance(label, datetime):
            _t = datetime.fromisoformat(label)
        else:
            _t = label
        try:
            tstamp = pytz.utc.localize(_t).astimezone(_tz)
        except ValueError:
            tstamp = _t.astimezone(_tz)
        label = tstamp.isoformat()[0:-6].replace("T", " ") + " " + _abbr
    else:
        label = label.replace("T", " ") + " " + "UTC"

    if kwargs:
        kwargs["label"] = label
    else:
        kwargs = {}
        kwargs["label"] = label

    img = add_text(img, **kwargs)

    return img


def add_circle(img, centerpos, radius, label=None, fill=None, outline=None, width=None):
    """
    Add a circle to an image
    @param img: (PIL.Image) input image
    @param centerpos: (tup) center position of circle in pixels (w, h)
    @param radius: (int) radius of circle in pixels
    @param label: (dict) label dictionary
    @param fill: (str) color to use for filling circle
    @param outline: (str) color to use for circle outline
    @param width: (int) line width in pixels
    @return: (PIL.Image) output image
    """

    # Unpack label dict
    if label:
        padding = label.get("padding", (20, 20))
        txt = label["label"]
        fontpath = label["fontpath"]
        fontsize = label.get("fontsize", 20)
        font = ImageFont.truetype(fontpath, fontsize)

    draw = ImageDraw.Draw(img)

    # Convert coords to bounding box
    xcenter = centerpos[0]
    ycenter = centerpos[1]
    x0 = xcenter - radius
    x1 = xcenter + radius
    y0 = ycenter - radius
    y1 = ycenter + radius
    xy = [(x0, y0), (x1, y1)]

    # Add circle
    draw.ellipse(xy, fill=fill, outline=outline, width=width)

    # Add anchor text if label has been provided
    if label:
        txtlength = draw.textlength(txt, font=font, font_size=fontsize)
        _x = xcenter + padding[0]
        if _x + txtlength > img.width:
            _x = _x - txtlength
        _y = ycenter - padding[1]

        kwargs = label
        kwargs.pop("padding")
        kwargs["position"] = (_x, _y)

        img = add_text(img, **kwargs)

    return img


def add_arrow(
    img,
    startpos,
    endpos,
    label=None,
    tiplength=0.1,
    width=8,
    color=(255, 0, 0),
    **kwargs,
):
    """
    Add an arrow to the image
    @param img: (PIL.Image obj) input image
    @param startpos: (tup) start position of arrow in pixels (w, h)
    @param endpos: (tup) end position of arrow in pixels (w, h)
    @param label: (dict) label dictionary
    @param tiplength: (int) length of arrow tip in pixels
    @param width: (int) width of line in pixels
    @param color: (tup) BGR color
    @param kwargs: (dict) see 'modify_image' function for options
    @return: (PIL.Image obj) output image
    """

    # Unpack label dict if provided
    if label:
        padding = label.get("padding", (20, 20))
        txt = label["label"]
        fontpath = label["fontpath"]
        opacity = label.get("opacity", 1)
        fontcolor = label.get("fontcolor", (0, 0, 0))
        fontsize = label.get("fontsize", 20)
        font = ImageFont.truetype(fontpath, fontsize)
        draw = ImageDraw.Draw(img)

    # Unpack arrow opacity
    if kwargs:
        arrow_opacity = kwargs.get("opacity", 1)
    else:
        arrow_opacity = 1

    # If opacity less than 1, need to create new image layer for alpha compositing
    if arrow_opacity != 1:
        img = img.convert("RGB")
        arrowimg = img
        img = img.convert("RGBA")
        imgarr = np.array(arrowimg)
        imgarr = cv2.arrowedLine(
            imgarr, startpos, endpos, color, thickness=width, tipLength=tiplength
        )
        arrowimg = Image.fromarray(imgarr)
        arrowimg = arrowimg.convert("RGBA")
        arrowimg.putalpha(int(arrow_opacity * 255))
        img = Image.alpha_composite(img, arrowimg)
    else:
        img = img.convert("RGB")
        imgarr = np.array(img)
        imgarr = cv2.arrowedLine(
            imgarr, startpos, endpos, color, thickness=width, tipLength=tiplength
        )
        img = Image.fromarray(imgarr)

    # Add label if provided
    if label:
        # Calculate position of label using start position of arrow
        label_x = 0
        label_y = 0
        # Put the text to the left if arrow points from left to right
        if startpos[0] <= endpos[0]:
            txtlength = int(draw.textlength(txt, font=font, font_size=fontsize))
            _x = startpos[0] - (padding[0] + txtlength)
            if _x < 0:
                _x = startpos[0] + padding[0]

            # Put the text below the arrow if points from bottom to top
            if startpos[1] <= endpos[1]:
                _y = startpos[1] - padding[1]
                if _y > img.height:
                    _y = startpos[1]

            # Put the text above the arrow if points from top to bottom
            else:
                _y = startpos[1] + padding[1]
                if _y < 0:
                    _y = 0

            label_x = _x
            label_y = _y

        # Put the text to the right if arrow points from right to left
        else:
            txtlength = int(draw.textlength(txt, font=font, font_size=fontsize))
            _x = startpos[0] + padding[0]
            if _x + txtlength > img.width:
                _x = img.width - txtlength

            # Put the text below the arrow if points from bottom to top
            if startpos[1] <= endpos[1]:
                _y = startpos[1] - padding[1]
                if _y > img.height:
                    _y = startpos[1]

            # Put the text above the arrow if points from top to bottom
            else:
                _y = startpos[1] + padding[1]
                if _y < 0:
                    _y = 0

            label_x = _x
            label_y = _y

        position = (label_x, label_y)

        # Build text dict
        textdict = {
            "position": position,
            "label": txt,
            "fontpath": fontpath,
            "fontcolor": fontcolor,
            "fontsize": fontsize,
            "opacity": opacity,
        }

        kwargs = {**textdict, **kwargs}

        # Add text to image
        img = add_text(img, **kwargs)

    return img


def add_triangle(img, xy, fill=None, outline=None, width=1):
    """
    Add a triangle to an image
    @param img: (PIL.Image obj) input image
    @param xy: (tup) location of triangle in pixels (w, h)
    @param fill: (str) color to use for filling triangle
    @param outline: (str) color to use for outline
    @param width: (int) width of outline in pixels
    @return: (PIL.Image obj) output image
    """
    draw = ImageDraw.Draw(img)
    draw.polygon(xy, fill=fill, outline=outline, width=width)

    return img


def convert_color(img):
    sub = img.convert("RGBA")
    data = np.array(sub)
    red, green, blue, alpha = data.T
    data = np.array([blue, green, red, alpha])
    data = data.transpose()
    sub = Image.fromarray(data)

    return sub


def modify_image(img, **kwargs):
    """
    Helper function to perform image modifications

    @param img: (PIL.Image obj) image to be modified
    @param kwargs: (dict) Valid keyword arguments include:
                   - 'res': (tup) output resolution w, h in pixels
                   - 'cmap': (str) or (matplotlib.colormap) applies a colormap to the image
                             If string is provided, it must either be a name of an existing
                             matplotlib colormap or a path to a json containing a colormap
                             generated by the online app at https://sciviscolor.org/color-moves-app/
                   - 'timestamps': (dict) adds a timestamp to the image
                                   {'label': (str) timestamp text,
                                    'tdict': {'position': (tup) of x,y pixel coords e.g. (1000, 200) OR
                                                          (str) location string e.g. 'upper-left',
                                              'fontpath': (str) path to ttf font,
                                              'fontcolor': (tup) RGB color of font,
                                              'fontsize': (int) size of font in pixels,
                                              'tzinfo': (tup) of pytz timezone object and 3-char tz
                                                        abbr.
                                              }
                                    }
                   - 'text': (dict) adds custom test to the image
                                    {'label': (str) text to add,
                                     'position': (tup) of x,y pixel coords e.g. (1000, 200)
                                     'fontpath': (str) path to ttf font
                                     'fontcolor': (tup) RGB color of font,
                                     'fontsize': (int) size of font in pixels
                                     'rotation': (float) angle in degrees for clockwise rotation
                                     }
                   - 'arrow': (dict) add an arrow to the image, and optionally, an anchored text
                                     string
                                     {'start_positon': (tup) x, y pixel coord of start position
                                      'end_position': (tup) x, y pixel coord of end position
                                      'tiplength': (float) length of arrow tip as fraction of arrow length
                                      'width': (int) width of arrow in pixels
                                      'color': (tup) BGR color
                                      'label': (dict) {'label': (str) text string to display,
                                                       'padding': (tup) padding between label and arrow start in pixels
                                                       'fontpath': (str) path to ttf font,
                                                       'fontcolor': (tup) RGB color,
                                                       'fontsize': (int) font size in pixels
                                                       },
                                      }
                   - 'circle': (dict) add a circle to the image, and optionally, an anchored text
                                string
                                      {'centerpos': (tup) x, y pixel coord of center position
                                       'radius': (int) radius of circle in pixels,
                                       'label': (dict) {'label': (str) text string to display,
                                                        'padding': (tup) padding between label and
                                                                   circle in pixels (x, y),
                                                        'fontpath': (str) path to ttf font,
                                                        'fontcolor': (tup) RGB color,
                                                        'fontsize': (int) font size in pixels
                                                        },
                                       'fill': (tup) RGB fill color for circle,
                                       'outline': (tup) RGB color of circle outline,
                                       'width': (int) width of circle outline in pixels,
                                       }
                   - 'triangle': (dict) {'coords': (list) x, y pixel coordinates of vertices as a tuple
                                         'fill': (tup) RGB fill color for triangle,
                                         'outline': (tup) RGB color of triangle outline,
                                         'width': (int) width of triangle outline in pixels

    @return: PIL Image
    """
    kwargs_copy = copy.deepcopy(kwargs)
    # Unpack kwargs
    cmap = kwargs.get("cmap", None)
    timestamps = kwargs.get("timestamps", None)
    text = kwargs.get("text", None)
    res = kwargs.get("res", None)
    arrow = kwargs.get("arrow", None)
    circle = kwargs_copy.get("circle", None)
    triangle = kwargs.get("triangle", None)
    if cmap:
        if ".json" in cmap:
            cmap = build_cmap(cmap)
        else:
            cmap = colormaps.get_cmap(cmap)

        img = img.convert("L")
        imgarr = np.array(img) / 255
        img = Image.fromarray(np.uint8(cmap(imgarr) * 255))

    if timestamps:
        ftime = timestamps.get("label")
        tdict = timestamps.get("tdict")

        img = add_timestamps(img, ftime, **tdict)

    if text:
        text_new = copy.deepcopy(text)
        if isinstance(text, list):
            for t in text:
                text_new = copy.deepcopy(t)
                img = add_text(img, **text_new)
        else:
            text_new = copy.deepcopy(text)
            img = add_text(img, **text_new)

    if arrow:
        if isinstance(arrow, list):
            for a in arrow:
                arrow_new = copy.deepcopy(a)

                startpos = arrow_new.pop("start_position")
                endpos = arrow_new.pop("end_position")

                img = add_arrow(img, startpos, endpos, **arrow_new)
        else:
            arrow_new = copy.deepcopy(arrow)

            startpos = arrow_new.pop("start_position")
            endpos = arrow_new.pop("end_position")

            img = add_arrow(img, startpos, endpos, **arrow_new)

    if circle:
        if isinstance(circle, list):
            for c in circle:
                circle_new = copy.deepcopy(c)
                radius = circle_new.pop("radius", None)

                centerpos = circle_new.pop("centerpos", None)
                img = add_circle(img, centerpos, radius, **circle_new)
        else:
            circle_new = copy.deepcopy(circle)

            centerpos = circle_new.pop("centerpos", None)
            radius = circle_new.pop("radius", None)

            img = add_circle(img, centerpos, radius, **circle_new)

    if res:
        img = img.resize((res[0], res[1]))

    if triangle:
        xy = triangle.pop("coords", None)
        fill = triangle.pop("fill", None)
        outline = triangle.pop("outline", None)
        width = triangle.pop("width", 1)
        img = add_triangle(img, xy, fill=fill, outline=outline, width=width)

    kwargs = kwargs_copy
    img.convert("RGB")
    return img


# ------ Video annotations
class GoesClip(VideoFileClip):
    def __init__(self, filepath):
        super().__init__(filepath)

    def preview(self, tmark=None, display=True, **kwargs):
        """
        Grabs a sample frame from the clip and applies desired annotations

        :param tmark: (float or tup or str) time to grab frame; if float it should represent
                                            the time mark in seconds; if tup it should be in the
                                            form (hours, mins, secs); if string it should be given
                                            as 'hh:mm:ss.ms'
        :param kwargs: annotation keywords
        :return: PIL Image
        """

        # Grab frame
        if tmark:
            nparr = self.get_frame(tmark)
        else:
            _t = np.random.Generator.integers(0, high=self.duration)
            nparr = self.get_frame(_t)

        # Convert to image
        img = Image.fromarray(nparr)

        # Apply modifications
        img = modify_image(img, **kwargs)
        if display:
            img.show()

        return img

    def annotate(
        self,
        t_edit_start=None,
        t_edit_end=None,
        freeze=False,
        freeze_img=None,
        codec=None,
        **kwargs,
    ):
        """
        Add annotations to an existing animation. Modified video will be saved to the same
        path.
        :param filepath: (str) path to video file
        :param svname: (str) name of output video; if None then name will be the same
                             as the input file suffixed by '_annotated'
        :param t_edit_start: (float or tup or str) start time to apply annotation; if float it should represent
                                            the time mark in seconds; if tup it should be in the form
                                            (hours, mins, secs); if string it should be given as
                                            'hh:mm:ss.ms'
        :param t_edit_end: (float or tup or str) end time to apply annotation; if float it should represent
                                            the time mark in seconds; if tup it should be in the form
                                            (hours, mins, secs); if string it should be given as
                                            'hh:mm:ss.ms'
        :param freeze: (bool) If true, the video will pause while annotations are displayed
        :param freeze_img: (PIL Image) If provided, this image will be displayed while the video is frozen
        :param kwargs: annotation-related kwargs
        :return: None
        """

        if t_edit_start and t_edit_end:
            tstart = t_edit_start
            tend = t_edit_end
        elif t_edit_end:
            tstart = 0
            tend = t_edit_end
        elif t_edit_start:
            tstart = t_edit_start
            tend = self.duration
        else:
            tstart = 0
            tend = self.duration

        if not freeze:
            clip = self.fl(
                lambda get_frame, t: (
                    get_frame(t)
                    if tstart >= t or t >= tend
                    else np.array(
                        modify_image(Image.fromarray(get_frame(t)), **kwargs),
                        dtype=np.uint8,
                    )
                )
            )
        else:
            if not freeze_img:
                img = Image.fromarray(self.get_frame(tstart))
                if kwargs:
                    img = modify_image(img, **kwargs)
                nparr = np.array(img, dtype=np.uint8)
                frzclip = frz.freeze(
                    self, t=t_edit_start, freeze_duration=t_edit_end - t_edit_start
                )
                clip = frzclip.fl(
                    lambda get_frame, t: nparr if tstart <= t <= tend else get_frame(t)
                )
            else:
                if (
                    self.size[0] != freeze_img.size[0]
                    or self.size[1] != freeze_img.size[1]
                ):
                    kwargs["res"] = (self.size[0], self.size[1])

                if kwargs:
                    freeze_img = modify_image(freeze_img, **kwargs)

                freeze_img = freeze_img.convert("RGB")
                nparr = np.array(freeze_img, dtype=np.uint8)
                frzclip = frz.freeze(
                    self, t=t_edit_start, freeze_duration=t_edit_end - t_edit_start
                )
                clip = frzclip.fl(
                    lambda get_frame, t: nparr if tstart <= t <= tend else get_frame(t)
                )

        if codec == "rawvideo":
            retclip = clip.fl(lambda get_frame, t: np.array(convert_color(img)))
        else:
            retclip = clip

        return retclip
