#!/usr/bin/env python3
'''
Seedhollow: an AI agent village in one file.

Eight villagers (seven people and SEED-1, a farm robot) live through a day.
Each has needs (hunger, energy, company), a job, a home and a memory. They
plan with a small utility brain, walk the map with BFS, farm, bake, fish,
forge, gossip, and pass a rumor from mouth to mouth.

The day is rendered to video three ways:
    village.mp4          pixel-art render (Pillow)
    village_ascii.mp4    the same frames through ascii.py's renderer
    montages/*.mp4       optional 9:16 Short cut by treasuretavern

Usage:
    python village.py                       # 45 s video + ASCII version
    python village.py --seconds 20 --no-ascii
    python village.py --shorts ../treasuretavern
    python village.py --llm                 # Claude writes the dialogue

Dependencies: numpy, pillow, imageio, imageio-ffmpeg (same as ascii.py).
--llm also needs the anthropic package and Claude API credentials.
--shorts needs ffmpeg + ffprobe on PATH and treasuretavern's node_modules.
'''

import os
import sys
import json
import math
import random
import argparse
import subprocess
from collections import deque

import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TILE = 16
GW, GH = 48, 27
W, H = GW * TILE, GH * TILE  # 768 x 432

# ---------------------------------------------------------------- world map

ROAD_ROW = 14
ROAD_COL = 23

# name: (x, y, w, h, door, wall, roof, label)
BUILDINGS = {
    'tavern':  (18, 5, 9, 6, (22, 11), '#b98b5a', '#7b3f2a', 'Tavern'),
    'house1':  (3, 9, 4, 3, (5, 12), '#d8c7a0', '#8a4b3b', None),
    'house2':  (9, 9, 4, 3, (11, 12), '#cdbb95', '#5b5f8a', None),
    'house3':  (29, 9, 4, 3, (31, 12), '#d8c7a0', '#3f6b4a', None),
    'house4':  (35, 9, 4, 3, (37, 12), '#cdbb95', '#8a6a3b', None),
    'house5':  (41, 9, 4, 3, (43, 12), '#d8c7a0', '#6b3f6b', None),
    'bakery':  (27, 16, 5, 4, (29, 15), '#e3cfa6', '#b0643a', 'Bakery'),
    'smithy':  (33, 16, 5, 4, (35, 15), '#8f8a84', '#3d3a38', 'Smithy'),
    'library': (39, 16, 5, 4, (41, 15), '#c9b8d6', '#4a3d6b', 'Library'),
    'dock':    (15, 16, 4, 3, (17, 15), '#9aa7ad', '#48606b', 'Dock'),
}
FIELD = (2, 17, 12, 8)          # x, y, w, h in tiles, 2x2 plots
POND = (40.5, 23.2, 5.2, 2.6)   # cx, cy, rx, ry in tiles
WELL = (25, 12)
FISH_SPOT = (34, 23)

VILLAGERS = [
    # name, role, home, color, skin
    ('Ada', 'farmer', 'house1', '#c9a227', '#f1c7a1'),
    ('Bram', 'baker', 'house2', '#e8e2d0', '#d9a07a'),
    ('Cora', 'smith', 'house3', '#7a2e2e', '#a8714f'),
    ('Dell', 'scribe', 'house4', '#3b4f8f', '#f0cfae'),
    ('Edda', 'innkeeper', 'tavern', '#8f3b6b', '#c98e6b'),
    ('Finn', 'fisher', 'house5', '#2f7f86', '#e6b894'),
    ('Mira', 'bard', 'tavern', '#d4622f', '#8d5a3b'),
    ('SEED-1', 'robot', 'dock', '#9fb3bf', None),
]



def tile_center(t):
    return (t[0] * TILE + 8, t[1] * TILE + 12)


def in_pond(x, y):
    cx, cy, rx, ry = POND
    return ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1


def build_walkable():
    walk = np.ones((GH, GW), dtype=bool)
    for x, y, w, h, door, *_ in BUILDINGS.values():
        walk[y:y + h, x:x + w] = False
        walk[door[1], door[0]] = True
    for y in range(GH):
        for x in range(GW):
            if in_pond(x, y):
                walk[y, x] = False
    walk[WELL[1], WELL[0]] = False
    return walk


WALKABLE = build_walkable()


def bfs(start, goal):
    '''Shortest 4-connected tile path from start to goal (inclusive of goal).'''
    if start == goal:
        return [goal]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < GW and 0 <= ny < GH and (nx, ny) not in prev and WALKABLE[ny, nx]:
                prev[(nx, ny)] = cur
                q.append((nx, ny))
    if goal not in prev:
        return [start]
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


# ---------------------------------------------------------------- farm plots

RIPE = 1.0
GIANT = 1.5   # a ripe crop that keeps getting watered instead of picked overgrows
MAX_GROWTH = 1.6


class Plot:
    def __init__(self, tx, ty, growth):
        self.tx, self.ty = tx, ty
        self.growth = growth
        self.water = 0       # minutes of watering left

    @property
    def tile(self):
        return (self.tx, self.ty)

    @property
    def ripe(self):
        return self.growth >= RIPE

    @property
    def giant(self):
        return self.growth >= GIANT


def make_plots(rng):
    fx, fy, fw, fh = FIELD
    plots = []
    for j in range(fh // 2):
        for i in range(fw // 2):
            plots.append(Plot(fx + i * 2, fy + j * 2, rng.uniform(0.0, 0.8)))
    return plots


# ---------------------------------------------------------------- dialogue

GREET = {
    'morning': ['Morning, {b}.', 'Up early, {b}?', 'Fine sunrise, eh {b}?'],
    'day': ['Busy day, {b}?', 'Hey {b}, how goes it?', '{b}! Just the person.'],
    'evening': ['Long day, {b}.', 'Pull up a stool, {b}.', 'Evening, {b}.'],
    'night': ['Still awake, {b}?', 'Quiet night, {b}.'],
}
TOPIC = {
    'farmer': ['The east rows need rain.', 'Pumpkins are coming in fat.', 'Soil smells right today.'],
    'baker': ['Fresh loaves at the Bakery.', 'Ran short on flour again.', 'Crust came out perfect.'],
    'smith': ['Forged three hoes this morning.', 'The bellows need patching.', 'Hot work, honest work.'],
    'scribe': ['I am writing it all down.', 'The ledger balances. Barely.', 'History is gossip with dates.'],
    'innkeeper': ['Stew is on the fire.', 'Tavern opens at dusk.', 'Rooms are full tonight.'],
    'fisher': ['The pond is generous today.', 'Caught one this long!', 'Fish bite before the rain.'],
    'bard': ['I have a new verse.', 'Every village needs a song.', 'Tell me something worth singing.'],
    'robot': ['WATER LEVELS NOMINAL.', 'SOIL MOISTURE OPTIMAL.', 'BATTERY OK. SOIL OK. HELLO.'],
}
REPLY = ['Ha, is that so?', 'Tell me more.', 'Good to hear.', 'You always say that.', 'Mm, fair enough.']
TELL = ['Did you hear? {r}!', 'Word is {r}.', 'Swear on my boots: {r}!']

# Things remarkable enough to become the village rumor. Whichever happens first wins.
RUMORS = {
    'pumpkin': ('SEED-1 grew a pumpkin as big as a cart',
                ['No! Truly?', 'The robot? Our robot?', 'I have to see that.', 'That is going in a song.']),
    'carp': ('Finn pulled a golden carp from the pond',
             ['Golden? Like coins?', 'Finn always exaggerates.', 'I have to see that.', 'That is going in a song.']),
}
GOLDEN_CARP_CHANCE = 0.03
HARVEST_MINUTES = 15


def time_of_day(hour):
    if 5 <= hour < 11:
        return 'morning'
    if 11 <= hour < 17:
        return 'day'
    if 17 <= hour < 22:
        return 'evening'
    return 'night'


def parse_exchange(text, a_name, b_name):
    '''Pulls "A: ..." and "B: ..." lines out of a reply. Returns (line_a, line_b) or None.'''
    found = {}
    for raw in text.splitlines():
        line = raw.replace('*', '').strip()  # tolerate **Name:** markdown
        for name in (a_name, b_name):
            if name not in found and line.lower().startswith(name.lower() + ':'):
                found[name] = line[len(name) + 1:].strip().strip('"')[:60]
    if a_name in found and b_name in found and found[a_name] and found[b_name]:
        return found[a_name], found[b_name]
    return None


class ClaudeWriter:
    '''
    Optional: asks Claude for a two-line exchange between villagers.

    Returns None whenever it cannot help (package missing, no credentials, API error,
    refusal, unparseable reply) and the sim falls back to its template lines. Missing
    credentials disable it for the rest of the run instead of failing on every chat.
    Anything else is a bug in this file and is allowed to raise.
    '''

    def __init__(self, model, max_calls, client=None):
        self.model, self.max_calls, self.calls = model, max_calls, 0
        self.client = client
        self.failures = 0
        self.anthropic = None
        if client is None:
            try:
                import anthropic
            except ImportError:
                print('[llm] disabled: pip install anthropic', file=sys.stderr)
                return
            self.anthropic = anthropic
            self.client = anthropic.Anthropic()

    def _errors(self):
        if self.anthropic is None:
            return (ConnectionError,)
        return (self.anthropic.APIStatusError, self.anthropic.APIConnectionError)

    def exchange(self, a, b, clock, rumor=None):
        if not self.client or self.calls >= self.max_calls:
            return None
        self.calls += 1
        news = f' {a.name} is excited to tell {b.name} this rumor: "{rumor}".' if rumor else ''
        prompt = (
            f'Village of Seedhollow, {clock}. {a.name} the {a.role} meets {b.name} the {b.role}.'
            f' {a.name} last did: {a.memory[-1] if a.memory else "nothing yet"}.{news}'
            f' Write their exchange as exactly two lines, "{a.name}: ..." then "{b.name}: ...".'
            ' Each line at most 9 words, plain words, no emoji. SEED-1 speaks in capitals.'
        )
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                output_config={'effort': 'low'},
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',
                messages=[{'role': 'user', 'content': prompt}],
            )
        except TypeError as e:
            # The SDK raises TypeError at request time when it finds no credentials.
            # Any other TypeError (a bad argument, an SDK too old for `fallbacks`) is a real bug.
            if 'authentication' not in str(e):
                raise
            print('[llm] disabled: no Claude API credentials found', file=sys.stderr)
            self.client = None
            return None
        except self._errors() as e:
            self.failures += 1
            print(f'[llm] call failed, using template lines: {e}', file=sys.stderr)
            return None
        if resp.stop_reason == 'refusal':
            return None
        text = ''.join(blk.text for blk in resp.content if blk.type == 'text')
        return parse_exchange(text, a.name, b.name)


# ---------------------------------------------------------------- agents

class Agent:
    def __init__(self, name, role, home, color, skin, rng):
        self.name, self.role, self.home, self.color, self.skin = name, role, home, color, skin
        self.rng = rng
        self.tile = BUILDINGS[home][4]
        self.x, self.y = tile_center(self.tile)
        self.path = []
        self.goal = None
        self.action = 'sleep'
        self.timer = 0
        self.target_plot = None
        self.hunger = rng.uniform(35, 60)
        self.energy = rng.uniform(70, 90)
        self.social = rng.uniform(20, 50)
        self.battery = 100.0
        self.harvest = 0
        self.wake = rng.randint(5 * 60 + 35, 6 * 60 + 40)
        self.knows = False
        self.memory = []
        self.bubble = None      # (text, frames_left)
        self.chat = None        # (partner, frames_left)
        self.moving = False
        self.hidden = True

    @property
    def is_robot(self):
        return self.role == 'robot'

    def say(self, text, frames=26):
        self.bubble = (text, frames)

    def go(self, tile):
        self.goal = tile
        self.path = bfs(self.tile, tile)
        if self.path and self.path[0] == self.tile:
            self.path.pop(0)

    def step(self, speed):
        self.moving = False
        if not self.path:
            return
        tx, ty = tile_center(self.path[0])
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist <= speed:
            self.x, self.y = tx, ty
            self.tile = self.path.pop(0)
        else:
            self.x += dx / dist * speed
            self.y += dy / dist * speed
        self.moving = True


class Village:
    def __init__(self, seed, start_minute, writer=None):
        self.rng = random.Random(seed)
        self.minute = start_minute   # minutes since midnight of day 1
        self.rumor = None            # (key, text) of the first remarkable event
        self.plots = make_plots(self.rng)
        self.agents = [Agent(*v, rng=self.rng) for v in VILLAGERS]
        self.by_name = {a.name: a for a in self.agents}
        self.stock = {'crops': 2, 'bread': 3, 'fish': 1, 'tools': 0, 'pages': 0}
        self.events = []
        self.pair_cd = {}
        self.writer = writer
        robot = self.by_name['SEED-1']
        robot.action, robot.hidden = 'idle', False

    # -- helpers
    @property
    def hour(self):
        return (self.minute // 60) % 24

    @property
    def day(self):
        return self.minute // 1440 + 1

    @property
    def clock(self):
        return f'{self.hour:02d}:{self.minute % 60:02d}'

    def log(self, text, who=None):
        self.events.append({'day': self.day, 'time': self.clock, 'text': text})
        if who:
            who.memory.append(text)

    def door(self, name):
        return BUILDINGS[name][4]

    def near_door(self, name, spread=2):
        dx, dy = self.door(name)
        for _ in range(20):
            t = (dx + self.rng.randint(-spread, spread), dy + self.rng.randint(0, spread))
            if 0 <= t[0] < GW and 0 <= t[1] < GH and WALKABLE[t[1], t[0]]:
                return t
        return (dx, dy)

    # -- the brain: pick the most pressing thing to do
    def decide(self, a):
        h = self.hour
        night = h >= 22 or h < 5 or (h < 7 and self.minute % 1440 < a.wake)
        if a.is_robot:
            return self.decide_robot(a)
        if night:
            a.action = 'sleep'
            a.go(self.door(a.home))
            return
        if a.hunger > 65:
            a.action = 'eat'
            a.go(self.door('bakery') if self.stock['bread'] > 0 else self.door('tavern'))
            return
        if 18 <= h < 22 and a.role != 'innkeeper' and a.social > 25:
            a.action = 'socialize'
            a.go(self.near_door('tavern', 3))
            a.timer = self.rng.randint(40, 90)
            return
        a.action = 'work'
        a.timer = self.rng.randint(30, 70)
        if a.role == 'farmer':
            ripe = [p for p in self.plots if p.ripe]
            near = lambda p: abs(p.tx - a.tile[0]) + abs(p.ty - a.tile[1])  # noqa: E731
            a.target_plot = min(ripe, key=near) if ripe else self.rng.choice(self.plots)
            a.harvest = 0
            a.go(a.target_plot.tile)
        elif a.role == 'fisher':
            a.go(FISH_SPOT)
        elif a.role == 'bard':
            awake = [b for b in self.agents if b is not a and not b.hidden]
            if awake:
                a.go(self.rng.choice(awake).tile)
            a.timer = 20
        else:
            site = {'baker': 'bakery', 'smith': 'smithy', 'scribe': 'library', 'innkeeper': 'tavern'}[a.role]
            a.go(self.near_door(site, 1))

    def decide_robot(self, a):
        if a.battery < 25:
            a.action = 'charge'
            a.go(self.door('dock'))
            return
        a.action = 'water'
        dry = sorted(self.plots, key=lambda p: (p.water > 0, self.rng.random()))
        a.target_plot = dry[0]
        a.go(a.target_plot.tile)
        a.timer = 12

    # -- what happens when an agent spends a minute on its action
    def act(self, a):
        if a.path:
            return
        if a.action == 'sleep':
            if not a.hidden:
                a.hidden = True
                self.log(f'{a.name} went to bed.', a)
            a.energy = min(100, a.energy + 0.25)
            if not (self.hour >= 22 or self.hour < 5 or self.minute % 1440 < a.wake):
                a.hidden = False
                self.log(f'{a.name} woke up.', a)
                self.decide(a)
            return
        a.hidden = False
        if a.action == 'eat':
            if self.stock['bread'] > 0 and a.tile == self.door('bakery'):
                self.stock['bread'] -= 1
                a.hunger -= 60
                a.say('Mm, warm bread.')
                self.log(f'{a.name} ate bread.', a)
            elif self.stock['fish'] > 0:
                self.stock['fish'] -= 1
                a.hunger -= 55
                a.say('Fish stew!')
                self.log(f'{a.name} had fish stew at the Tavern.', a)
            else:
                a.hunger -= 25
                a.say('Thin soup again.')
                self.log(f'{a.name} made do with soup.', a)
            a.hunger = max(0, a.hunger)
            a.timer = 0
        elif a.action == 'water':
            p = a.target_plot
            p.water = 180
            if p.growth <= 0:
                p.growth = 0.02
            if self.rng.random() < 0.04:
                a.say(self.rng.choice(TOPIC['robot']))
        elif a.action == 'charge':
            a.hidden = True
            a.battery = min(100, a.battery + 1.2)
            if a.battery >= 100:
                a.hidden = False
                a.action, a.timer = 'idle', 0
                self.log('SEED-1 finished charging.', a)
            return
        elif a.action == 'work':
            self.work(a)
        if a.timer > 0:
            a.timer -= 1

    def work(self, a):
        m, r = self.minute, a.role
        if r == 'farmer' and a.target_plot:
            p = a.target_plot
            if p.ripe and a.harvest < HARVEST_MINUTES:
                a.harvest += 1  # picking takes a while
            elif p.ripe:
                a.harvest = 0
                if p.giant:
                    self.stock['crops'] += 3
                    a.say('By the stars, look at it!', 40)
                    self.log(f'{a.name} harvested a giant pumpkin.', a)
                    self.discover('pumpkin', a)
                else:
                    self.stock['crops'] += 1
                    a.say('Another one ripe.')
                    self.log(f'{a.name} harvested a crop.', a)
                p.growth = 0.0
                a.timer = 0
            else:
                p.growth = min(RIPE, p.growth + 0.002)
                if any(q.ripe for q in self.plots):
                    a.timer = 0  # something is ripe: go pick it instead
        elif r == 'baker' and m % 30 == 0:
            if self.stock['crops'] > 0:
                self.stock['crops'] -= 1
                self.stock['bread'] += 2
                self.log(f'{a.name} baked two loaves.', a)
            elif m % 60 == 0:
                self.stock['bread'] += 1
        elif r == 'smith' and m % 90 == 0:
            self.stock['tools'] += 1
            self.log(f'{a.name} forged a tool.', a)
        elif r == 'scribe' and m % 45 == 0:
            self.stock['pages'] += 1
        elif r == 'fisher' and m % 20 == 0 and self.rng.random() < 0.45:
            self.stock['fish'] += 1
            if self.rng.random() < GOLDEN_CARP_CHANCE:
                a.say('It is... GOLDEN?!', 40)
                self.log(f'{a.name} caught a golden carp.', a)
                self.discover('carp', a)
            else:
                a.say('Got one!')
                self.log(f'{a.name} caught a fish.', a)

    def discover(self, key, witness):
        '''The first remarkable event becomes the rumor; its witness is the first to know.'''
        if self.rumor is None:
            self.rumor = (key, RUMORS[key][0])
            witness.knows = True
            self.log(f'New rumor: {self.rumor[1]}.', witness)

    # -- two agents close together may talk, and gossip spreads
    def try_chats(self):
        if self.hour >= 22 or self.hour < 5:
            return
        awake = [a for a in self.agents if not a.hidden and a.chat is None and a.action != 'sleep']
        for i, a in enumerate(awake):
            for b in awake[i + 1:]:
                if a.chat or b.chat or math.hypot(a.x - b.x, a.y - b.y) > 30:
                    continue
                key = tuple(sorted((a.name, b.name)))
                if self.minute < self.pair_cd.get(key, -1):
                    continue
                if max(a.social, b.social) < 30 and not (a.knows != b.knows):
                    continue
                self.start_chat(a, b, key)

    def start_chat(self, a, b, key):
        if b.knows and not a.knows:
            a, b = b, a
        tells = a.knows and not b.knows
        rumor = self.rumor[1] if tells else None
        lines = self.writer.exchange(a, b, self.clock, rumor) if self.writer else None
        if not lines:
            if tells:
                lines = (self.rng.choice(TELL).format(r=rumor), self.rng.choice(RUMORS[self.rumor[0]][1]))
            elif self.rng.random() < 0.5:
                lines = (self.rng.choice(GREET[time_of_day(self.hour)]).format(b=b.name), self.rng.choice(REPLY))
            else:
                lines = (self.rng.choice(TOPIC[a.role]), self.rng.choice(REPLY))
        a.say(lines[0], 30)
        b.pending_line = lines[1]
        a.chat, b.chat = (b, 60), (a, 60)
        a.path, b.path = [], []
        a.social = max(0, a.social - 45)
        b.social = max(0, b.social - 45)
        self.pair_cd[key] = self.minute + 150
        if tells:
            b.knows = True
            self.log(f'{a.name} told {b.name} the rumor.', b)
        else:
            self.log(f'{a.name} and {b.name} chatted.', a)

    # -- one simulated minute
    def tick(self):
        self.minute += 1
        for p in self.plots:
            if 0 < p.growth < RIPE:
                p.growth = min(RIPE, p.growth + (0.0022 if p.water > 0 else 0.0008))
            elif p.ripe and p.water > 0:
                p.growth = min(MAX_GROWTH, p.growth + 0.0022)  # overgrowth, only while watered
            p.water = max(0, p.water - 1)
        for a in self.agents:
            if a.bubble:
                text, left = a.bubble
                a.bubble = (text, left - 1) if left > 1 else None
            if a.chat:
                partner, left = a.chat
                if left == 30 and getattr(a, 'pending_line', None):
                    a.say(a.pending_line, 30)
                    a.pending_line = None
                a.chat = (partner, left - 1) if left > 1 else None
                a.moving = False
                if a.chat is None and a.goal and a.tile != a.goal:
                    a.go(a.goal)  # pick the errand back up
                continue
            if not a.is_robot and a.action != 'sleep':
                a.hunger += 0.18
                a.social += 0.14
                a.energy -= 0.05
            if a.is_robot:
                a.battery -= 0.09
            a.step(2.6 if a.is_robot else 4.6)
            bedtime = not a.is_robot and a.action != 'sleep' and (self.hour >= 22 or self.hour < 5)
            if bedtime or (not a.path and a.timer <= 0 and a.action not in ('sleep', 'charge')):
                self.decide(a)
            self.act(a)
        self.try_chats()


# ---------------------------------------------------------------- renderer

def load_font(size):
    for name in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', os.path.join(HERE, 'cour.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_background(seed):
    rng = random.Random(seed + 1)
    img = Image.new('RGB', (W, H), '#5f9e4a')
    d = ImageDraw.Draw(img)
    for ty in range(GH):
        for tx in range(GW):
            g = rng.randint(-10, 10)
            d.rectangle([tx * TILE, ty * TILE, tx * TILE + 15, ty * TILE + 15], fill=(95 + g, 158 + g, 74 + g // 2))
            for _ in range(2):
                px, py = tx * TILE + rng.randint(1, 14), ty * TILE + rng.randint(1, 14)
                d.line([px, py, px, py - 2], fill=(70, 128, 55))
    # roads
    d.rectangle([0, ROAD_ROW * TILE, W, ROAD_ROW * TILE + 15], fill='#c8b27a')
    d.rectangle([ROAD_COL * TILE, 11 * TILE, ROAD_COL * TILE + 15, H], fill='#c8b27a')
    for x in range(0, W, 7):
        d.point([x, ROAD_ROW * TILE + rng.randint(2, 13)], fill='#a8925e')
    # field
    fx, fy, fw, fh = FIELD
    d.rectangle([fx * TILE - 3, fy * TILE - 3, (fx + fw) * TILE + 2, (fy + fh) * TILE + 2], outline='#6b4a2b', width=2)
    for px in range(fx * TILE - 3, (fx + fw) * TILE + 3, 12):
        d.rectangle([px, fy * TILE - 6, px + 2, fy * TILE], fill='#6b4a2b')
    # pond
    cx, cy, rx, ry = POND
    d.ellipse([(cx - rx - 0.4) * TILE, (cy - ry - 0.4) * TILE, (cx + rx + 0.4) * TILE, (cy + ry + 0.4) * TILE], fill='#d9c48f')
    d.ellipse([(cx - rx) * TILE, (cy - ry) * TILE, (cx + rx) * TILE, (cy + ry) * TILE], fill='#3b7bbf')
    for _ in range(14):
        x, y = (cx + rng.uniform(-rx * 0.7, rx * 0.7)) * TILE, (cy + rng.uniform(-ry * 0.6, ry * 0.6)) * TILE
        d.line([x, y, x + 8, y], fill='#6fa6dc')
    # well
    wx, wy = WELL[0] * TILE, WELL[1] * TILE
    d.ellipse([wx + 1, wy + 3, wx + 15, wy + 15], fill='#7d7d7d', outline='#4a4a4a')
    d.ellipse([wx + 4, wy + 6, wx + 12, wy + 12], fill='#23405e')
    d.polygon([(wx - 1, wy + 4), (wx + 8, wy - 4), (wx + 17, wy + 4)], fill='#7b3f2a')
    # trees
    trees = []
    for _ in range(70):
        tx, ty = rng.randint(0, GW - 1), rng.randint(0, GH - 1)
        if WALKABLE[ty, tx] and ty not in (ROAD_ROW, ROAD_ROW - 1, ROAD_ROW + 1) and tx != ROAD_COL \
                and not (FIELD[0] - 1 <= tx <= FIELD[0] + FIELD[2] and FIELD[1] - 1 <= ty <= FIELD[1] + FIELD[3]) \
                and all(abs(tx - b[4][0]) > 1 or abs(ty - b[4][1]) > 1 for b in BUILDINGS.values()) \
                and (ty < 8 or ty > 24 or tx < 2 or tx > 45):
            trees.append((tx, ty))
    for tx, ty in trees:
        x, y = tx * TILE + 8, ty * TILE + 8
        d.rectangle([x - 1, y + 2, x + 1, y + 8], fill='#5a3b22')
        d.ellipse([x - 8, y - 9, x + 8, y + 5], fill='#2f6b32')
        d.ellipse([x - 5, y - 7, x + 3, y - 1], fill='#3f8a42')
    # buildings
    label_font = load_font(9)
    windows = {}
    for name, (x, y, w, h, door, wall, roof, label) in BUILDINGS.items():
        X, Y, X2, Y2 = x * TILE, y * TILE, (x + w) * TILE - 1, (y + h) * TILE - 1
        roof_h = max(10, h * TILE // 2)
        d.rectangle([X + 2, Y + roof_h, X2 - 2, Y2], fill=wall, outline='#3a2a1a')
        d.polygon([(X - 2, Y + roof_h + 2), (X + 8, Y), (X2 - 8, Y), (X2 + 2, Y + roof_h + 2)], fill=roof, outline='#2a1a10')
        dx = door[0] * TILE
        door_top = Y2 - 12 if door[1] > y else Y + roof_h
        d.rectangle([dx + 4, door_top, dx + 11, door_top + 12], fill='#3d2616')
        wins = []
        for wxp in (X + 8, X2 - 16):
            if abs(wxp - dx) > 10:
                wy = Y + roof_h + 5
                d.rectangle([wxp, wy, wxp + 7, wy + 6], fill='#2c3b4a', outline='#3a2a1a')
                wins.append((wxp, wy))
        windows[name] = wins
        if name == 'bakery':
            d.rectangle([X2 - 14, Y - 6, X2 - 8, Y + 6], fill='#6b4a3b')
        if label:
            tw = d.textlength(label, font=label_font)
            lx, ly = (X + X2) / 2 - tw / 2, Y + roof_h // 2 - 4
            d.rectangle([lx - 2, ly, lx + tw + 2, ly + 11], fill='#f2e6c9', outline='#3a2a1a')
            d.text((lx, ly), label, fill='#2a1a10', font=label_font)
    return img, windows


def daylight(minute):
    '''RGB multiplier for the time of day.'''
    h = (minute % 1440) / 60
    night, dawn, day, dusk = (0.32, 0.38, 0.62), (0.95, 0.72, 0.62), (1.0, 1.0, 1.0), (1.0, 0.68, 0.5)

    def mix(a, b, t):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    if h < 5 or h >= 21.5:
        return night
    if h < 6:
        return mix(night, dawn, h - 5)
    if h < 7.5:
        return mix(dawn, day, (h - 6) / 1.5)
    if h < 18:
        return day
    if h < 19.5:
        return mix(day, dusk, (h - 18) / 1.5)
    return mix(dusk, night, (h - 19.5) / 2)


def glow_sprite(radius, color):
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    fall = np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / radius, 0, 1) ** 2
    return fall[..., None] * np.array(color, dtype=np.float32)[None, None, :]


class Renderer:
    def __init__(self, village, seed):
        self.v = village
        self.bg, self.windows = draw_background(seed)
        self.font = load_font(10)
        self.hud_font = load_font(11)
        self.glow = glow_sprite(26, (255, 190, 90))
        self.small_glow = glow_sprite(12, (255, 210, 120))

    def plot(self, d, p):
        X, Y = p.tx * TILE, p.ty * TILE
        d.rectangle([X + 1, Y + 1, X + 30, Y + 30], fill='#5a3a20' if p.water else '#7a5230')
        for r in range(3):
            d.line([X + 3, Y + 6 + r * 10, X + 28, Y + 6 + r * 10], fill='#6b4526')
        g = p.growth
        if g <= 0.02:
            return
        for i in range(3):
            for j in range(3):
                cx, cy = X + 6 + i * 10, Y + 6 + j * 10
                if g < 0.35:
                    d.point([cx, cy - 1], fill='#8fd16a')
                    d.point([cx + 1, cy - 2], fill='#8fd16a')
                elif g < 0.7:
                    d.ellipse([cx - 2, cy - 4, cx + 2, cy], fill='#5fb34a')
                elif g < 1:
                    d.ellipse([cx - 3, cy - 5, cx + 3, cy + 1], fill='#4a9a3a')
                    d.point([cx, cy - 2], fill='#e0a030')
                elif g < RIPE + 0.05 or (i, j) != (1, 1):
                    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 2], fill='#e8892a', outline='#9a5518')
        if g >= RIPE + 0.05:
            # the centre pumpkin swells as the plot overgrows
            s = 4 + int(11 * min(1, (g - RIPE) / (GIANT - RIPE)))
            d.ellipse([X + 16 - s, Y + 18 - s, X + 16 + s, Y + 16 + s // 2], fill='#f08a24', outline='#8a4a10', width=2)
            d.line([X + 16, Y + 18 - s, X + 17, Y + 13 - s], fill='#3f7a2a', width=2)

    def agent(self, d, a, frame):
        x, y = int(a.x), int(a.y)
        d.ellipse([x - 6, y - 2, x + 6, y + 2], fill=(40, 60, 30))
        bob = (frame // 3) % 2 if a.moving else 0
        if a.is_robot:
            d.rectangle([x - 6, y - 4, x - 3, y], fill='#2a2a2a')
            d.rectangle([x + 3, y - 4, x + 6, y], fill='#2a2a2a')
            d.rectangle([x - 7, y - 14 - bob, x + 7, y - 3 - bob], fill=a.color, outline='#44525a')
            d.rectangle([x - 5, y - 11 - bob, x + 5, y - 8 - bob], fill='#1d2b33')
            eye = '#6ff7c8' if (frame // 12) % 4 else '#1d2b33'
            d.rectangle([x - 3, y - 10 - bob, x - 1, y - 9 - bob], fill=eye)
            d.rectangle([x + 1, y - 10 - bob, x + 3, y - 9 - bob], fill=eye)
            d.line([x, y - 14 - bob, x, y - 19 - bob], fill='#44525a')
            d.point([x, y - 20 - bob], fill='#ff5a5a' if (frame // 8) % 2 else '#ffd0d0')
            if a.action == 'water' and not a.path:
                for k in range(3):
                    d.point([x + 9 + k, y - 6 + ((frame + k * 3) % 6)], fill='#8fc8ff')
        else:
            step = (frame // 4) % 2 if a.moving else 0
            d.rectangle([x - 3, y - 4, x - 1, y - step], fill='#3a2a1e')
            d.rectangle([x + 1, y - 4, x + 3, y - (1 - step) if a.moving else y], fill='#3a2a1e')
            d.rectangle([x - 4, y - 12 - bob, x + 4, y - 4 - bob], fill=a.color, outline='#2a1a10')
            d.ellipse([x - 4, y - 19 - bob, x + 4, y - 11 - bob], fill=a.skin, outline='#2a1a10')
            d.rectangle([x - 4, y - 19 - bob, x + 4, y - 16 - bob], fill='#3a2a1e')
            if a.action == 'work' and not a.path and a.role == 'smith' and frame % 10 < 3:
                d.point([x + 7, y - 10], fill='#ffd24a')
                d.point([x + 9, y - 13], fill='#ff8a2a')
        if a.knows:
            sx, sy = x, y - 26 - bob
            d.polygon([(sx, sy - 4), (sx + 1, sy - 1), (sx + 4, sy - 1), (sx + 2, sy + 1),
                       (sx + 3, sy + 4), (sx, sy + 2), (sx - 3, sy + 4), (sx - 2, sy + 1),
                       (sx - 4, sy - 1), (sx - 1, sy - 1)], fill='#ffd84a', outline='#8a6a10')

    def bubble(self, d, a, placed):
        '''Draws a speech bubble, lifted above any bubble it would overlap. Returns its box.'''
        text = f'{a.name}: {a.bubble[0]}'
        tw = d.textlength(text, font=self.font)
        x = min(max(4, a.x - tw / 2 - 4), W - tw - 12)
        y = max(24, a.y - 46)
        for bx0, by0, bx1, by1 in sorted(placed, key=lambda b: -b[1]):
            if x < bx1 and x + tw + 8 > bx0 and y < by1 + 6 and y + 15 > by0:
                y = by0 - 21
        y = max(20, y)
        d.rounded_rectangle([x, y, x + tw + 8, y + 15], radius=5, fill='#fffdf5', outline='#2a1a10')
        tail_x = min(max(x + 6, a.x), x + tw + 2)
        d.polygon([(tail_x - 3, y + 15), (tail_x + 3, y + 15), (tail_x, y + 20)], fill='#fffdf5', outline='#2a1a10')
        d.line([tail_x - 2, y + 15, tail_x + 2, y + 15], fill='#fffdf5')
        d.text((x + 4, y + 2), text, fill='#1a1410', font=self.font)
        return (x, y, x + tw + 8, y + 15)

    def lights(self, v):
        spots = []
        occupied = {a.home for a in v.agents if a.hidden and a.action == 'sleep'}
        late = v.hour >= 18 or v.hour < 7
        for name, wins in self.windows.items():
            lit = name in ('tavern', 'library', 'bakery') and 17 <= v.hour < 23
            lit = lit or (name in occupied and late and v.hour < 23 and v.hour >= 18) or (name in occupied and 5 <= v.hour < 7)
            for wx, wy in wins:
                spots.append((wx + 4, wy + 3, lit, True))
        spots.append((WELL[0] * TILE + 8, WELL[1] * TILE + 2, True, False))
        return spots

    def frame(self, frame_no):
        v = self.v
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        for p in v.plots:
            self.plot(d, p)
        if v.stock['bread'] > 0 and 6 <= v.hour < 20:
            bx, by = BUILDINGS['bakery'][0] * TILE + BUILDINGS['bakery'][2] * TILE - 11, BUILDINGS['bakery'][1] * TILE - 10
            for k in range(3):
                off = (frame_no + k * 9) % 27
                d.ellipse([bx - 3 + k, by - off, bx + 3 + k, by - off + 5], fill=(225, 225, 225))
        visible = sorted([a for a in v.agents if not a.hidden], key=lambda a: a.y)
        for a in visible:
            self.agent(d, a, frame_no)
        # lighting
        arr = np.asarray(img, dtype=np.float32) * np.array(daylight(v.minute), dtype=np.float32)
        darkness = 1 - daylight(v.minute)[0]
        if darkness > 0.05:
            for x, y, lit, window in self.lights(v):
                if not lit:
                    continue
                if window:
                    arr[y - 3:y + 4, x - 4:x + 4] = (255, 214, 120)
                g = self.small_glow if window else self.glow
                r = g.shape[0] // 2
                y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
                arr[y0:y1, x0:x1] += g[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)] * darkness * 0.9
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        placed = []
        for a in sorted((a for a in visible if a.bubble), key=lambda a: -a.y):
            placed.append(self.bubble(d, a, placed))
        clean = np.array(img)  # no HUD: feeds the ASCII and vertical renders
        self.hud(d)
        return clean, np.asarray(img)

    def hud(self, d):
        v = self.v
        s = v.stock
        knowers = sum(a.knows for a in v.agents)
        left = f'SEEDHOLLOW   Day {v.day}  {v.clock}'
        right = f'bread {s["bread"]}  crops {s["crops"]}  fish {s["fish"]}  tools {s["tools"]}   rumor {knowers}/{len(v.agents)}'
        d.rectangle([0, 0, W, 17], fill=(20, 24, 22))
        d.text((6, 2), left, fill='#f2e6c9', font=self.hud_font)
        tw = d.textlength(right, font=self.hud_font)
        d.text((W - tw - 6, 2), right, fill='#ffd84a' if knowers > 1 else '#cfd8cf', font=self.hud_font)
        recent = v.events[-2:]
        if recent:
            h = 14 * len(recent) + 6
            d.rectangle([0, H - h, W, H], fill=(20, 24, 22))
            for i, e in enumerate(recent):
                d.text((6, H - h + 3 + i * 14), f'{e["time"]}  {e["text"]}', fill='#cfd8cf', font=self.font)


# ---------------------------------------------------------------- output

def ascii_frames_writer(path, fps, fontsize, background):
    '''
    Returns (convert, close) built on ascii.py's NumPy renderer, or None if unavailable.
    convert(frame) writes the ASCII frame to `path` and returns it.
    '''
    sys.path.insert(0, HERE)
    try:
        import ascii as ascii_video
    except ImportError as e:
        print(f'[ascii] ascii.py not importable, skipping: {e}', file=sys.stderr)
        return None
    chars = np.array(list(' .:-=+*#%@'))
    bitmaps = ascii_video.get_font_bitmaps(fontsize, max(1, fontsize // 10), False, background, chars,
                                           os.path.join(HERE, 'cour.ttf'))
    empty = np.array([], dtype=np.uint16)
    state = {}

    def convert(frame):
        out = ascii_video.draw_ascii(frame, chars, background, True, empty, bitmaps)
        out = out[: out.shape[0] // 2 * 2, : out.shape[1] // 2 * 2]
        if 'writer' not in state:
            h, w = out.shape[:2]
            state['writer'] = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, quality=7, macro_block_size=2)
            state['writer'].send(None)
        out = np.ascontiguousarray(out)
        state['writer'].send(out)
        return out

    def close():
        if 'writer' in state:
            state['writer'].close()
    return convert, close


class Vertical:
    '''
    Lays out a 1080x1920 frame for Shorts: a caption band, the pixel village,
    the same moment through ascii.py, then clock, rumor meter and recent events.
    '''
    VW, VH = 1080, 1920
    BAND = 200                       # caption band; treasuretavern draws here with --shorts
    CROP = (24, 0, W - 24, H)        # 720x432 of the map, scaled 1.5x to 1080x648
    PANEL_H = 648

    def __init__(self, village, caption=None):
        self.v = village
        self.caption = caption       # None when treasuretavern will burn the caption in
        self.big = load_font(58)
        self.mid = load_font(34)
        self.small = load_font(27)
        self.tag = load_font(22)
        self.chip = load_font(24)

    def panel(self, arr):
        img = Image.fromarray(arr).crop(self.CROP) if arr.shape[1] >= W else Image.fromarray(arr)
        return img.resize((self.VW, self.PANEL_H), Image.NEAREST)

    def compose(self, clean, ascii_frame):
        v = self.v
        img = Image.new('RGB', (self.VW, self.VH), (18, 22, 20))
        d = ImageDraw.Draw(img)
        if self.caption:
            tw = d.textlength(self.caption, font=self.big)
            d.text(((self.VW - tw) / 2, 30), self.caption, fill='#f2e6c9', font=self.big)
        sub = 'Seedhollow, an AI agent village'
        tw = d.textlength(sub, font=self.small)
        d.text(((self.VW - tw) / 2, 140), sub, fill='#9fb09a', font=self.small)

        y = self.BAND
        for arr, label in ((clean, 'PIXEL'), (ascii_frame, 'ASCII  via ascii.py')):
            img.paste(self.panel(arr), (0, y))
            lw = d.textlength(label, font=self.tag)
            d.rectangle([16, y + 14, 16 + lw + 20, y + 48], fill=(18, 22, 20))
            d.text((26, y + 17), label, fill='#cfd8cf', font=self.tag)
            y += self.PANEL_H

        # info panel
        top = y + 24
        d.text((40, top), f'Day {v.day}   {v.clock}', fill='#f2e6c9', font=self.big)
        knowers = sum(a.knows for a in v.agents)
        rumor = v.rumor[1] if v.rumor else 'No rumor yet. Something will happen.'
        d.text((40, top + 78), f'RUMOR {knowers}/{len(v.agents)}', fill='#ffd84a' if knowers else '#8a948a', font=self.mid)
        d.text((260, top + 82), rumor[:44], fill='#e6dcc0' if v.rumor else '#8a948a', font=self.small)
        # one chip per villager, ringed in gold once they have heard the rumor
        for i, a in enumerate(v.agents):
            cx, cy = 40 + (i % 4) * 255, top + 140 + (i // 4) * 58
            ring = '#ffd84a' if a.knows else '#3a443c'
            d.ellipse([cx, cy, cx + 38, cy + 38], fill=a.color, outline=ring, width=4)
            d.text((cx + 50, cy + 5), a.name, fill='#f2e6c9' if a.knows else '#8a948a', font=self.chip)
        ey = top + 262
        for e in v.events[-3:]:
            d.text((40, ey), f'{e["time"]}  {e["text"]}'[:62], fill='#cfd8cf', font=self.tag)
            ey += 34
        return np.asarray(img)


def clean_caption(text):
    # treasuretavern's drawtext escaping breaks on quotes, colons, commas and brackets.
    return ''.join(c for c in text if c.isalnum() or c in ' .!?-').strip()


def make_short(tt_dir, src, out_dir, caption, seconds):
    service = os.path.join(os.path.abspath(tt_dir), 'src', 'services', 'VideoMontageService')
    ts_node = os.path.join(os.path.abspath(tt_dir), 'node_modules', '.bin', 'ts-node')
    if not os.path.exists(service + '.ts') or not os.path.exists(ts_node):
        print(f'[shorts] treasuretavern not found or not installed at {tt_dir}', file=sys.stderr)
        return None
    runner = os.path.join(out_dir, '_short_runner.ts')
    with open(runner, 'w') as f:
        f.write(
            f"import {{ VideoMontageService }} from {json.dumps(service)};\n"
            f"new VideoMontageService({json.dumps(os.path.abspath(out_dir))})\n"
            f"  .processVerticalFormat({json.dumps(os.path.abspath(src))}, {{\n"
            f"    text: {json.dumps(clean_caption(caption))}, fontSize: 64, textPosition: 'top',\n"
            f"    outputFileName: 'village_short.mp4', textDuration: {{ start: 0, end: {seconds} }},\n"
            f"    verification: {{ requireDimensionCheck: true, requireManualCheck: false }},\n"
            f"  }}).then(p => console.log('SHORT ' + p)).catch(e => {{ console.error(e); process.exit(1); }});\n"
        )
    env = dict(os.environ, TS_NODE_COMPILER_OPTIONS=json.dumps(
        {'module': 'commonjs', 'moduleResolution': 'node', 'esModuleInterop': True}))
    res = subprocess.run([ts_node, '--transpile-only', runner], capture_output=True, text=True, env=env, cwd=tt_dir)
    os.remove(runner)
    for line in res.stdout.splitlines():
        if line.startswith('SHORT '):
            return line[6:]
    print(f'[shorts] failed:\n{res.stderr[-2000:]}', file=sys.stderr)
    return None


def parse_args():
    p = argparse.ArgumentParser(description='Simulate an AI agent village and render it to video.')
    p.add_argument('-o', '--out', default='village_out', help='Output directory.')
    p.add_argument('-s', '--seconds', type=float, default=45, help='Video length in seconds.')
    p.add_argument('--fps', type=int, default=24, help='Frames per second.')
    p.add_argument('--speed', type=int, default=1, help='Village minutes per frame. At 24 fps, --speed 2 fits two days into 60 s.')
    p.add_argument('--start', default='05:30', help='Clock time the video starts at (HH:MM).')
    p.add_argument('--seed', type=int, default=3,
                   help='Random seed. Same seed, same day. Seed 3 finds its rumor on day 1; many seeds take two days.')
    p.add_argument('--no-ascii', action='store_true', help='Skip the ASCII render (also skips the vertical render).')
    p.add_argument('--no-vertical', action='store_true', help='Skip the 1080x1920 render.')
    p.add_argument('--ascii-font', type=int, default=10, help='ASCII glyph size in pixels.')
    p.add_argument('--ascii-bg', type=int, default=0, choices=[0, 255], help='ASCII background, 0 black or 255 white.')
    p.add_argument('--shorts', metavar='TREASURETAVERN_DIR',
                   help='Burn the caption into the vertical render with treasuretavern instead of in Python.')
    p.add_argument('--caption', default='A day in an AI village', help='Caption for the vertical render.')
    p.add_argument('--llm', action='store_true', help='Let Claude write the dialogue (needs anthropic + API credentials).')
    p.add_argument('--model', default='claude-opus-5', help='Claude model for --llm.')
    p.add_argument('--llm-max', type=int, default=30, help='Maximum Claude calls per run. Each call pauses the render.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    hh, mm = map(int, args.start.split(':'))
    writer = ClaudeWriter(args.model, args.llm_max) if args.llm else None
    village = Village(args.seed, hh * 60 + mm, writer)
    renderer = Renderer(village, args.seed)

    paths = {k: os.path.join(args.out, f) for k, f in
             (('pixel', 'village.mp4'), ('ascii', 'village_ascii.mp4'), ('vertical', 'village_vertical.mp4'))}
    pixel = imageio_ffmpeg.write_frames(paths['pixel'], (W, H), fps=args.fps, quality=8, macro_block_size=16)
    pixel.send(None)
    ascii_out = None if args.no_ascii else ascii_frames_writer(paths['ascii'], args.fps, args.ascii_font, args.ascii_bg)
    vertical = vwriter = None
    if ascii_out and not args.no_vertical:
        vertical = Vertical(village, None if args.shorts else args.caption)
        vwriter = imageio_ffmpeg.write_frames(paths['vertical'], (Vertical.VW, Vertical.VH), fps=args.fps,
                                              quality=7, macro_block_size=8)
        vwriter.send(None)

    frames = int(args.seconds * args.fps)
    for i in range(frames):
        for _ in range(args.speed):
            village.tick()
        clean, framed = renderer.frame(i)
        pixel.send(np.ascontiguousarray(framed))
        if ascii_out:
            art = ascii_out[0](clean)
            if vwriter:
                vwriter.send(np.ascontiguousarray(vertical.compose(clean, art)))
        if i % args.fps == 0:
            print(f'\rDay {village.day} {village.clock}  frame {i}/{frames}', end='', flush=True)
    print()
    pixel.close()
    if ascii_out:
        ascii_out[1]()
    if vwriter:
        vwriter.close()

    with open(os.path.join(args.out, 'village_log.json'), 'w') as f:
        json.dump({
            'seed': args.seed,
            'rumor': village.rumor[1] if village.rumor else None,
            'rumor_known_by': [a.name for a in village.agents if a.knows],
            'stock': village.stock,
            'llm_calls': writer.calls if writer else 0,
            'events': village.events,
            'memories': {a.name: a.memory for a in village.agents},
        }, f, indent=2)

    print(f'pixel     {paths["pixel"]}')
    if ascii_out:
        print(f'ascii     {paths["ascii"]}')
    if vwriter:
        print(f'vertical  {paths["vertical"]}')
        if args.shorts:
            short = make_short(args.shorts, paths['vertical'], args.out, args.caption, args.seconds)
            if short:
                print(f'short     {short}')
    print(f'log       {os.path.join(args.out, "village_log.json")}  ({len(village.events)} events, rumor: '
          f'{village.rumor[1] if village.rumor else "none"}, known by {sum(a.knows for a in village.agents)}/8)')


if __name__ == '__main__':
    main()
