'''Tests for village.py. Run with: python -m unittest test_village'''

import unittest
from types import SimpleNamespace

import numpy as np

import village as V

START = 5 * 60 + 30


def run(seed, minutes, writer=None):
    v = V.Village(seed, START, writer)
    for _ in range(minutes):
        v.tick()
    return v


class FakeClient:
    '''Stands in for anthropic.Anthropic(): returns a canned reply or raises.'''

    def __init__(self, text=None, error=None):
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.text, self.error = text, error

    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason='end_turn', content=[SimpleNamespace(type='text', text=self.text)])


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


class TestSimulation(unittest.TestCase):
    def test_same_seed_same_day(self):
        a, b = run(3, 900), run(3, 900)
        self.assertEqual(a.events, b.events)
        self.assertEqual(a.stock, b.stock)

    def test_day_counter_rolls_over_at_midnight(self):
        v = run(1, 24 * 60 - START + 5)  # to 00:05 on day 2
        self.assertEqual(v.day, 2)
        self.assertEqual(v.clock, '00:05')
        self.assertEqual({e['day'] for e in v.events} - {1, 2}, set())

    def test_everyone_asleep_at_2am(self):
        for seed in range(1, 6):
            v = run(seed, 24 * 60 - START + 2 * 60)  # 02:00, day 2
            for a in v.agents:
                if not a.is_robot:
                    self.assertEqual(a.action, 'sleep', f'seed {seed}: {a.name} is {a.action}')
                    self.assertTrue(a.hidden, f'seed {seed}: {a.name} is not in bed')

    def test_nothing_is_scripted_at_start(self):
        v = V.Village(3, START)
        self.assertIsNone(v.rumor)
        self.assertFalse(any(a.knows for a in v.agents))
        self.assertFalse(any(p.giant or p.ripe for p in v.plots))

    def test_rumor_spreads_only_by_being_told(self):
        for seed in (3, 9, 14):
            v = run(seed, 2 * 1440)
            self.assertIsNotNone(v.rumor, f'seed {seed} found no rumor')
            told = {e['text'].split(' told ')[1].split(' the rumor')[0]
                    for e in v.events if ' told ' in e['text']}
            untold = [a for a in v.agents if a.knows and a.name not in told]
            # exactly one villager learns it without being told: the one who saw it happen
            self.assertEqual(len(untold), 1, f'seed {seed}: {[a.name for a in untold]}')
            self.assertTrue(any(m.startswith('New rumor') for m in untold[0].memory))

    def test_giant_pumpkin_can_emerge(self):
        # Across seeds, overgrowth produces a pumpkin rumor without any plot being preset.
        kinds = {run(seed, 2 * 1440).rumor[0] for seed in (1, 2, 9)}
        self.assertIn('pumpkin', kinds)


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

    def test_claude_lines_are_used_when_available(self):
        client = FakeClient(text='Ada: The rows look fine.\nBram: Bread by noon.')
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        v = V.Village(3, START, writer)
        ada, bram = v.by_name['Ada'], v.by_name['Bram']
        ada.hidden = bram.hidden = False
        v.start_chat(ada, bram, ('Ada', 'Bram'))
        self.assertEqual(ada.bubble[0], 'The rows look fine.')
        self.assertEqual(bram.pending_line, 'Bread by noon.')
        self.assertEqual(client.calls, 1)

    def test_missing_credentials_disable_the_writer(self):
        err = TypeError('Could not resolve authentication method. Expected one of api_key...')
        client = FakeClient(error=err)
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        a, b = SimpleNamespace(name='Ada', role='farmer', memory=[]), SimpleNamespace(name='Bram', role='baker')
        self.assertIsNone(writer.exchange(a, b, '09:00'))
        self.assertIsNone(writer.client)
        self.assertIsNone(writer.exchange(a, b, '09:01'))
        self.assertEqual(client.calls, 1)

    def test_other_type_errors_are_not_swallowed(self):
        client = FakeClient(error=TypeError("create() got an unexpected keyword argument 'fallbacks'"))
        writer = V.ClaudeWriter('claude-opus-5', 5, client=client)
        a, b = SimpleNamespace(name='Ada', role='farmer', memory=[]), SimpleNamespace(name='Bram', role='baker')
        with self.assertRaises(TypeError):
            writer.exchange(a, b, '09:00')

    def test_call_budget_is_respected(self):
        client = FakeClient(text='Ada: Hi.\nBram: Hi.')
        writer = V.ClaudeWriter('claude-opus-5', 2, client=client)
        a, b = SimpleNamespace(name='Ada', role='farmer', memory=[]), SimpleNamespace(name='Bram', role='baker')
        for _ in range(5):
            writer.exchange(a, b, '09:00')
        self.assertEqual(client.calls, 2)


class TestRender(unittest.TestCase):
    def test_frames_have_expected_shapes(self):
        v = run(3, 400)
        clean, framed = V.Renderer(v, 3).frame(0)
        self.assertEqual(clean.shape, (V.H, V.W, 3))
        self.assertEqual(framed.shape, (V.H, V.W, 3))
        self.assertFalse(np.array_equal(clean[:17], framed[:17]))  # HUD only on the framed copy
        vert = V.Vertical(v, 'caption').compose(clean, clean)
        self.assertEqual(vert.shape, (V.Vertical.VH, V.Vertical.VW, 3))

    def test_caption_cleaning_strips_what_breaks_treasuretavern(self):
        self.assertEqual(V.clean_caption("Tip: it's [easy], 100%"), 'Tip its easy 100')


if __name__ == '__main__':
    unittest.main()
