# ASCII Video Converter

Convert Video and Images to ASCII form! Achieve Real-Time Color ASCII Rendering using NumPy Vectorization.

See this algorithm work on a webcam stream: https://github.com/AlexEidt/ASCII-Vision

## Usage

```
positional arguments:
  filename      File name of the input image.
  output        File name of the output image.

optional arguments:
  -h, --help    show this help message and exit
  -chars CHARS  ASCII chars to use in media.
  -r            Reverse the character order.
  -f [F]        Font size.
  -b [B]        Boldness of characters. Recommended boldness is 1/10 of Font size.
  -bg [BG]      Background color. Must be either 255 for white or 0 for black.
  -m M          Color to use for Monochromatic characters in "R,G,B" format.
  -c            Clip characters to not go outside of image bounds.
  -font [FONT]  Font to use.
  -a            Add audio from the input file to the output file.
  -q            Quality of the output video. (0-10), 0 worst, 10 best.
```

Instead of a filename, a directory name can be used as the `filename` argument. All media in this directory will be converted to ASCII Form and be placed in a directory specified by the `output` argument. Note that this means if a directory is passed as the `filename` argument, the `output` argument must also be a directory and all media in `filename` must be either images and/or videos.

## Dependencies

* Python 3.7+
* `imageio`
* `imageio-ffmpeg`
* `numpy`
* `PIL`
* `tqdm`

```
pip install numpy pillow tqdm imageio imageio-ffmpeg
```

# Images

<img src="Documentation/butterfly.jpg" alt="Butterfly" />

<img src="Documentation/butterfly-ascii-color.png" alt="Butterfly ASCII Color" />

<img src="Documentation/butterfly-ascii-mono.png" alt="Butterfly ASCII Monochrome" />


<img src="Documentation/houses.jpg" alt="Houses" />

<img src="Documentation/houses-ascii-color.png" alt="Houses ASCII Color" />

<img src="Documentation/houses-ascii-mono.png" alt="Houses ASCII Monochrome" />


# Video

<img src="Documentation/donuts.gif" alt="Donuts">

<img src="Documentation/donuts-ascii.gif" alt="Donuts ASCII">

# Seedhollow: an AI agent village

`village.py` is a whole agent village in one file. Seven villagers and SEED-1, a farm robot, live through the day. They plan around hunger, energy and company, walk the map, farm, bake, fish, forge, chat, and pass along a rumor. One frame is one village minute (`--speed` changes that).

By default the agents are rule-based: a utility function picks each action and dialogue comes from templates. With `--llm`, Claude writes the conversations. Nothing in the story is scripted. The rumor is whatever remarkable thing happens first:
- a pumpkin the robot kept watering after it ripened grows giant, or
- Finn lands a golden carp.

Across seeds 1 to 20, 18 produce a rumor within two days. Seed 3, the default, finds its rumor on day 1.

```
python village.py                                  # 45 s day: village.mp4, village_ascii.mp4, village_vertical.mp4
python village.py --seconds 60 --speed 2 --seed 9  # two days in one minute
python village.py --shorts ../treasuretavern       # treasuretavern burns the caption into the vertical render
python village.py --llm                            # Claude writes the dialogue (anthropic package + API credentials)
python -m unittest test_village                    # tests
```

Outputs go to `village_out/`:
- `village.mp4`: the pixel render, 768x432.
- `village_ascii.mp4`: the same frames through `ascii.py`.
- `village_vertical.mp4`: a 1080x1920 layout for Shorts, with the pixel view, the ASCII view and a rumor tracker.
- `village_log.json`: every event and each villager's memory.

The same `--seed` always produces the same day. A 50 s render with every output takes about 50 s on 4 CPUs; most of that is encoding the vertical video.
