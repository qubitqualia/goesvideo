import json
import pytz
import cv2
from PIL import ImageDraw, ImageFont, Image
import matplotlib.colors as colors
from matplotlib import colormaps
from datetime import datetime
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from colorama import Fore


# Functions for image modifications
def build_cmap(fname):
    """Builds a matplotlib cmap for a colormap specification generated using the app at sciviscolor.org"""
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


def add_text(img, **kwargs):
    """Adds custom text to an image frame"""
    position = kwargs.get("position", "upper-left")
    fontcolor = kwargs.get("fontcolor", (0, 0, 0))
    fontpath = kwargs["fontpath"]
    fontsize = kwargs.get("fontsize", None)
    label = kwargs["label"]
    opacity = kwargs.get("opacity", 1)

    # Convert input image to RGB
    img = img.convert("RGB")

    # If fontsize not provided, attempt to scale to image resolution
    if not fontsize:
        fontsize = int(0.02 * img.height)

    font = ImageFont.truetype(fontpath, fontsize)

    # If opacity equals 1 then text can be added directly to the input image
    # Otherwise, need to create a new image layer for alpha compositing
    if opacity != 1:
        fontcolor = (fontcolor[0], fontcolor[1], fontcolor[2], int(opacity * 255))
        layercolor = (fontcolor[0], fontcolor[1], fontcolor[2], 0)
        txtimg = Image.new("RGBA", img.size, layercolor)
        draw = ImageDraw.Draw(txtimg)
    else:
        draw = ImageDraw.Draw(img)

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

    if opacity != 1:
        draw.text((x, y), label, fill=fontcolor, font_size=fontsize, **kwargs)
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, txtimg)
    else:
        draw.text(
            (x, y), label, fill=fontcolor, font=font, font_size=fontsize, **kwargs
        )

    return img


def add_timestamps(img, label, **kwargs):
    """Adds a timestamp to an image frame"""

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
        tstamp = pytz.utc.localize(_t).astimezone(_tz)
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


def add_circle(
    img, centerpos, radius, label=None, fill=None, outline=None, width=None, **kwargs
):
    """Adds a circle to the image"""

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
    """Adds an arrow to the image"""

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
        arrow_color = (color[0], color[1], color[2], 0)
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
            txtlength = draw.textlength(txt, font=font, font_size=fontsize)
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
            txtlength = draw.textlength(txt, font=font, font_size=fontsize)
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


def modify_image(img, **kwargs):
    """
    Helper function to perform image modifications

    :param img: (PIL Image) image to be modified
    :param kwargs: (dict) Valid keyword arguments include:
                   - 'cmap': (str) or (matplotlib.colormap) applies a colormap to the image
                             If string is provided, it must either be a name of an existing
                             matplotlib colormap or a path to a json containing a colormap
                             generated by the online app at https://sciviscolor.org/color-moves-app/
                   - 'timestamps': (dict) adds a timestamp to the image
                                   {'label': (str) timestamp text,
                                    'tdict': {'position': (tup) of x,y pixel coords or (str) location,
                                              'fontpath': (str) path to ttf font,
                                              'fontcolor': (tup) RGB color of font,
                                              'fontsize': (int) size of font in pixels,
                                              'tzinfo': (tup) of pytz timezone object and 3-char tz abbr.
                                              }
                                    }
                   - 'text': (dict) adds custom test to the image
                                    {'label': (str) text to add,
                                     'position': (tup) of x, y pixel coords
                                     'fontpath': (str) path to ttf font
                                     'fontcolor': (tup) RGB color of font,
                                     'fontsize': (int) size of font in pixels
                                     }
                   - 'res': (tup) output resolution w, h in pixels
                   - 'arrow': (dict) add an arrow to the image, and optionally, an anchored text string
                                     {'start_positon': (tup) x, y pixel coord of start position,
                                      'end_position': (tup) x, y pixel coord of end position,
                                      'padding': (tup) padding between label and arrow start in pixels (x, y),
                                      'label': (dict) {'label': (str) text string to display,
                                                       'padding': (tup) padding between label and circle in pixels (x, y),
                                                       'fontpath': (str) path to ttf font,
                                                       'fontcolor': (tup) RGB color,
                                                       'fontsize': (int) font size in pixels
                                                       },
                                      }
                   - 'circle': (dict) add a circle to the image, and optionally, an anchored text string
                                      {'centerpos': (tup) x, y pixel coord of center position,
                                       'radius': (int) radius of circle in pixels,
                                       'label': (dict) {'label': (str) text string to display,
                                                        'padding': (tup) padding between label and circle in pixels (x, y),
                                                        'fontpath': (str) path to ttf font,
                                                        'fontcolor': (tup) RGB color,
                                                        'fontsize': (int) font size in pixels
                                                        },
                                       'fill': (tup) RGB fill color for circle,
                                       'outline': (tup) RGB color of circle outline,
                                       'width': (int) width of circle outline in pixels,
                                       }

    :return: PIL Image
    """

    # Unpack kwargs
    cmap = kwargs.get("cmap", None)
    timestamps = kwargs.get("timestamps", None)
    text = kwargs.get("text", None)
    res = kwargs.get("res", None)
    arrow = kwargs.get("arrow", None)
    circle = kwargs.get("circle", None)

    if cmap:
        if ".json" in cmap:
            cmap = build_cmap(cmap)
        else:
            cmap = colormaps.get_cmap(cmap)

        img = img.convert("L")
        imgarr = np.array(img) / 255
        img = Image.fromarray(np.uint8(cmap(imgarr) * 255))

    if timestamps:
        ftime = timestamps.pop("label")

        img = add_timestamps(img, ftime, **timestamps)

    if text:
        img = add_text(img, **text)

    if arrow:
        startpos = arrow.pop("start_position")
        endpos = arrow.pop("end_position")

        img = add_arrow(img, startpos, endpos, **arrow)

    if circle:
        centerpos = circle.pop("centerpos")
        radius = circle.pop("radius")

        img = add_circle(img, centerpos, radius, **circle)

    if res:
        img = img.resize((res[0], res[1]))

    return img


def annotate_video(
    filepath, svname=None, t_edit_start=None, t_edit_end=None, freeze=False, **kwargs
):
    """
    Add annotations to an existing animation. Modified video will be saved to the same
    path.
    :param filepath: (str) path to video file
    :param svname: (str) name of output video; if None then name will be the same
                         as the input file suffixed by '_annotated'
    :param t_edit_start: (float) start time for edit in seconds
    :param t_edit_end: (float) end time for edit in seconds
    :param freeze: If true, the video will pause while annotations are displayed
    :param kwargs: annotation-related kwargs
    :return: None
    """
    basepath = "\\".join(filepath.split("\\")[0:-1]) + "\\"
    # Try to get codec from savename or if not provided use avi
    if not svname:
        svname = basepath + "_annotated.avi"
        codec = "rawvideo"
    else:
        svname = basepath + svname
        if svname.split(".")[1] == "avi":
            codec = "rawvideo"
        elif svname.split(".")[1] == "mp4":
            codec = "mpeg4"
        elif svname.split(".")[1] == "wmv":
            codec = "libvpx"
        else:
            codec = "mpeg4"
            print()
            print(
                f"{Fore.RED} WARNING - Unrecognized file extension. Trying mpeg4 codec."
            )

    clip = VideoFileClip(filepath)

    if not t_edit_start:
        t_edit_start = 0
    if not t_edit_end:
        t_edit_end = clip.duration

    imgrepl = []
    imgbuffer_1 = []
    imgbuffer_2 = []
    _enter = False
    _exit = False
    for t, frame in clip.iter_frames(with_times=True, logger="bar", dtype="uint8"):
        img = Image.fromarray(frame)
        if t_edit_start <= t <= t_edit_end:
            if freeze:
                if not _enter:
                    editimg = modify_image(img, **kwargs)
                    imgrepl.append(np.array(editimg))
                    _enter = True
                    imgbuffer_2.append(np.array(img))
                else:
                    imgrepl.append(np.array(editimg))
                    imgbuffer_2.append(np.array(img))
            else:
                editimg = modify_image(img, **kwargs)
                imgbuffer_1.append(np.array(editimg))

        elif t < t_edit_start:
            imgbuffer_1.append(np.array(img))
        else:
            imgbuffer_2.append(np.array(img))

    imgarr = imgbuffer_1 + imgrepl + imgbuffer_2
    if codec == "avi":
        imgarr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in imgarr]

    outclip = ImageSequenceClip(imgarr, fps=clip.fps)
    outclip.write_videofile(svname, codec=codec)

    return


# Class for video editing
class GoesClip(VideoFileClip):
    def __init__(self, filepath):
        super().__init__(filepath)

    def preview(self, tmark=None, **kwargs):
        """
        Grabs a sample frame from the clip and applies desired annotations

        :param tmark: (float or tup or str) time to grab frame; if float it should represent
                                            the time mark in seconds; if tup it should be in the form
                                            (hours, mins, secs); if string it should be given as
                                            'hh:mm:ss.ms'
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

        return img

    def annotate(
        self,
        t_edit_start=None,
        t_edit_end=None,
        freeze=False,
        freeze_img=None,
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

        tstart = t_edit_start
        tend = t_edit_end

        if not freeze:
            clip = self.fl(
                lambda get_frame, t: get_frame(t)
                if tstart >= t or t >= tend
                else np.array(
                    modify_image(Image.fromarray(get_frame(t)), **kwargs),
                    dtype=np.uint8,
                )
            )
        else:
            if not freeze_img:
                img = Image.fromarray(self.get_frame(tstart))
                if kwargs:
                    img = modify_image(img, **kwargs)
                nparr = np.array(img, dtype=np.uint8)
                clip = self.fl(
                    lambda get_frame, t: nparr if tstart <= t <= tend else get_frame(t)
                )
            else:
                if self.size != freeze_img.size:
                    kwargs["res"] = (self.size[0], self.size[1])

                if kwargs:
                    freeze_img = modify_image(freeze_img, **kwargs)

                nparr = np.array(freeze_img, dtype=np.uint8)
                clip = self.fl(
                    lambda get_frame, t: nparr if tstart <= t <= tend else get_frame(t)
                )

        return clip
