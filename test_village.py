'''
Tests for village.py. Run with: python -m unittest test_village

Behavioural tests sweep seeds and assert rates or invariants, not the outcome of
hand-picked seeds. Thresholds come from a measured baseline over seeds 1 to 50
(two days each): a rumor in 46 of 50 (31 pumpkin, 15 carp), latest bedtime 00:43.
'''

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import imageio_ffmpeg

import village as V

START = 5 * 60 + 30
SEEDS = range(1, 51)
TWO_DAYS = 2 * 1440


def run(seed, minutes):
    v = V.Village(seed, START)
    for _ in range(minutes):
        v.tick()
    return v


class FakeClient:
    '''Stands in for anthropic.Anthropic(): returns a canned reply or raises.'''

    def __init__(self, text=None, error=None, api_key='test-key'):
        self.api_key, self.auth_token, self.credentials = api_key, None, None
        self.calls = 0
        self.prompts = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.text, self.error = text, error

    def create(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs['messages'][0]['content'])
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason='end_turn', content=[SimpleNamespace(type='text', text=self.text)])


def chat(cid, a='Ada', b='Bram', rumor=None):
    return {'id': cid, 'day': 1, 'time': '09:00', 'rumor': rumor, 'lines': ['x', 'y'],
            'a': a, 'a_role': 'farmer', 'a_last': None, 'b': b, 'b_role': 'baker'}


class TestWorld(unittest.TestCase):
    def test_bfs_path_is_walkable_and_connected(self):
        start, goal = V.BUILDINGS['house1'][4], V.FISH_SPOT
        path = V.bfs(start, goal)
        self.assertEqual(path[-1], goal)
        prev = start
        for x, y in path:
            self.assertTrue(V.WALKABLE[y, x])
            self.assertEqual(abs(x - prev[0]) + abs(y - prev[1]), 1)
            prev = (x, y)

    def test_every_door_reachable_from_every_door(self):
        doors = [b[4] for b in V.BUILDINGS.values()]
        for a in doors:
            for b in doors:
                self.assertEqual(V.bfs(a, b)[-1], b)


class TestSimulationSweep(unittest.TestCase):
    '''One pass over 50 seeds x 2 days, checking invariants on every tick.'''

    @classmethod
    def setUpClass(cls):
        cls.villages, cls.violations = {}, []
        for seed in SEEDS:
            v = V.Village(seed, START)
            knew = set()
            for _ in range(TWO_DAYS):
                n_events = len(v.events)
                v.tick()
                fresh = [e['text'] for e in v.events[n_events:]]
                for a in v.agents:
                    if not V.WALKABLE[a.tile[1], a.tile[0]]:
                        cls.violations.append(f'seed {seed} {v.clock}: {a.name} on a wall at {a.tile}')
                if any(n < 0 for n in v.stock.values()):
                    cls.violations.append(f'seed {seed} {v.clock}: negative stock {v.stock}')
                if v.hour == 1 and v.minute % 60 == 30:
                    for a in v.agents:
                        if not a.is_robot and not (a.hidden and a.action == 'sleep'):
                            cls.violations.append(f'seed {seed} day {v.day} 01:30: {a.name} still up')
                # anyone who newly knows the rumor must have been told, or seen it, this very tick
                for a in v.agents:
                    if a.knows and a.name not in knew:
                        told = any(f' told {a.name} the rumor' in t for t in fresh)
                        saw = any(t.startswith('New rumor') for t in a.memory[-3:])
                        if not (told or saw):
                            cls.violations.append(f'seed {seed} {v.clock}: {a.name} learned the rumor unseen')
                        knew.add(a.name)
            cls.villages[seed] = v

    def test_no_invariant_violations(self):
        self.assertEqual(self.violations[:10], [])

    def test_rumor_emerges_in_most_seeds(self):
        found = sum(v.rumor is not None for v in self.villages.values())
        self.assertGreaterEqual(found, 40, f'rumor in only {found}/50 seeds (baseline 46)')

    def test_both_rumor_mechanisms_occur(self):
        kinds = {v.rumor[0] for v in self.villages.values() if v.rumor}
        self.assertEqual(kinds, {'pumpkin', 'carp'})

    def test_exactly_one_witness_per_rumor(self):
        for seed, v in self.villages.items():
            if v.rumor:
                witnesses = [a for a in v.agents if any(m.startswith('New rumor') for m in a.memory)]
                self.assertEqual(len(witnesses), 1, f'seed {seed}')

    def test_day_counter_reaches_day_three(self):
        v = self.villages[1]
        self.assertEqual((v.day, v.clock), (3, '05:30'))
        days = [e['day'] for e in v.events]
        self.assertEqual(days, sorted(days))       # days only move forward
        self.assertEqual(set(days) - {1, 2, 3}, set())
        self.assertIn(2, days)                     # the rollover really happened

    def test_same_seed_same_story(self):
        again = run(7, TWO_DAYS)
        self.assertEqual(again.events, self.villages[7].events)
        self.assertEqual(again.chats, self.villages[7].chats)

    def test_nothing_is_scripted_at_start(self):
        for seed in SEEDS:
            v = V.Village(seed, START)
            self.assertIsNone(v.rumor)
            self.assertFalse(any(a.knows for a in v.agents))
            self.assertFalse(any(p.ripe for p in v.plots))


class TestDialogue(unittest.TestCase):
    def test_parse_exchange_ignores_colons_inside_lines(self):
        text = 'Ada: Meet me at 07:30 by the well.\nBram: Fine, bring bread.'
        self.assertEqual(V.parse_exchange(text, 'Ada', 'Bram'),
                         ('Meet me at 07:30 by the well.', 'Fine, bring bread.'))

    def test_parse_exchange_handles_markdown_and_preamble(self):
        text = 'Here you go:\n**Ada:** Hello there.\n**Bram:** Morning.'
        self.assertEqual(V.parse_exchange(text, 'Ada', 'Bram'), ('Hello there.', 'Morning.'))

    def test_parse_exchange_rejects_half_answers(self):
        self.assertIsNone(V.parse_exchange('Ada: Hello.', 'Ada', 'Bram'))

    def test_fill_rewrites_chats_and_renderer_uses_them(self):
        client = FakeClient(text='Ada: The rows look fine.\nBram: Bread by noon.')
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        self.assertEqual(writer.fill([chat(0)]), {(0, 0): 'The rows look fine.', (0, 1): 'Bread by noon.'})
        # a real chat from the sim, drawn with rewritten lines
        v = V.Village(3, START)
        ref = None
        while ref is None and v.minute < START + 1440:
            v.tick()
            ref = next((a.bubble[2] for a in v.agents if a.bubble and a.bubble[2] and not a.hidden), None)
        self.assertIsNotNone(ref, 'no chat all day')
        cid, idx = ref
        scene = V.snapshot(v)
        plain = V.Renderer(3).frame(scene, 0)[0]
        rewritten = V.Renderer(3, {(cid, idx): 'A LINE CLAUDE WROTE'}).frame(scene, 0)[0]
        self.assertFalse(np.array_equal(plain, rewritten))

    def test_rumor_chats_are_rewritten_first(self):
        client = FakeClient(text='Ada: Hi.\nBram: Hi.')
        writer = V.ClaudeWriter('claude-opus-5', 2, client=client)
        chats = [chat(0), chat(1), chat(2, rumor='a golden carp'), chat(3), chat(4, rumor='a golden carp')]
        got = writer.fill(chats)
        self.assertEqual(client.calls, 2)
        self.assertEqual({k[0] for k in got}, {2, 4})

    def test_missing_credentials_disable_the_writer_without_calling(self):
        client = FakeClient(text='Ada: Hi.\nBram: Hi.', api_key=None)
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        self.assertEqual(writer.fill([chat(0)]), {})
        self.assertEqual(client.calls, 0)

    def test_api_errors_keep_template_lines(self):
        client = FakeClient(error=ConnectionError('network down'))
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        self.assertEqual(writer.fill([chat(0), chat(1)]), {})
        self.assertEqual(writer.failures, 2)

    def test_bugs_are_not_swallowed(self):
        client = FakeClient(error=TypeError("create() got an unexpected keyword argument 'fallbacks'"))
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        with self.assertRaises(TypeError):
            writer.fill([chat(0)])


class TestRender(unittest.TestCase):
    def test_frames_have_expected_shapes(self):
        scene = V.snapshot(run(3, 400))
        clean, framed = V.Renderer(3).frame(scene, 0)
        self.assertEqual(clean.shape, (V.H, V.W, 3))
        self.assertFalse(np.array_equal(clean[:17], framed[:17]))  # HUD only on the framed copy
        vert = V.Vertical('caption').compose(scene, clean, V.AsciiStage(10, 0)(clean))
        self.assertEqual(vert.shape, (V.Vertical.VH, V.Vertical.VW, 3))

    def test_rendering_is_deterministic(self):
        scene = V.snapshot(run(5, 900))
        a = V.Renderer(5).frame(scene, 42)[1]
        b = V.Renderer(5).frame(scene, 42)[1]
        self.assertTrue(np.array_equal(a, b))

    def test_caption_cleaning_strips_what_breaks_treasuretavern(self):
        self.assertEqual(V.clean_caption("Tip: it's [easy], 100%"), 'Tip its easy 100')


class TestEncodedFiles(unittest.TestCase):
    '''Checks the files actually written, including parallel segments joined back together.'''

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        args = V.parse_args(['-o', cls.tmp, '-s', '2', '--fps', '12', '-j', '3', '--start', '19:00'])
        cls.village, cls.paths, cls.log = V.run(args)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def probe(self, path):
        reader = imageio_ffmpeg.read_frames(path)
        meta = next(reader)
        frames = sum(1 for _ in reader)
        reader.close()
        return meta['size'], frames

    def test_every_output_has_every_frame_at_its_exact_size(self):
        expected = {'pixel': (V.W, V.H), 'vertical': (V.Vertical.VW, V.Vertical.VH)}
        for kind in ('pixel', 'ascii', 'vertical'):
            size, frames = self.probe(self.paths[kind])
            self.assertEqual(frames, 24, kind)
            if kind in expected:
                self.assertEqual(tuple(size), expected[kind], kind)
            self.assertEqual((size[0] % 2, size[1] % 2), (0, 0), kind)

    def test_segments_are_cleaned_up(self):
        self.assertFalse(os.path.exists(os.path.join(self.tmp, '.segments')))

    def test_log_records_chats_and_timings(self):
        self.assertIn('render', self.log['timings_s'])
        self.assertEqual(len(self.log['chats']), len(self.village.chats))


if __name__ == '__main__':
    unittest.main()
