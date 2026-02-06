import functools
import time as pytime

from pyscript import when
from pyscript.web import *
import numpy as np
from .position_optimization import PermutationFinder
from .state import share_url


class ScorePlotter:
    def __init__(self, element_id):
        self.svg = page.find(element_id)[0]
        self.polyline = self.svg.find('polyline')[0]
        self.width = 1000
        self.height = 100

        # Reset display and clear points
        self.svg.style['display'] = 'block'
        self.polyline.setAttribute('points', '')

    def update(self, vals):
        vals = -np.array(vals)
        vals -= np.min(vals)
        vals /= max(1e-9, np.max(np.abs(vals)))

        points = []
        xs = np.exp(np.log1p(np.linspace(0, self.width, len(vals))))
        xs = np.log1p(np.linspace(0, self.width, len(vals)))
        xs *= self.width / xs.max()
        vals = (self.height - 5) * vals + 5 * (1 - vals)

        for x, y in zip(xs, vals):
            points.append(f'{x:.1f},{y:.1f}')

        self.polyline.setAttribute('points', ' '.join(points))


class PlayerTable:
    def __init__(self):
        self.table = self.find_table()

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def find_table():
        result = page.find('#player_table')[0]
        result.append(
            tr(
                th('Name'),
                th('Setting'),
                th('Blocking'),
                th('Hitting'),
                th('Hit L/R'),
                th('Set from back'),
            )
        )
        return result

    def add_row(self, name, setter, middle, outside, pin_pref, back_row_set):
        make_check = lambda x, cls: input_(
            type='range',
            classes=[cls, 'value_slider', 'autosave'],
            min=0,
            max=1,
            step='0.001',
            value=x,
        )
        name_elt = input_(value=str(name), classes=['player_name', 'autosave'])
        when('input', name_elt, handler=share_url)

        row = tr(
            td(name_elt),
            td(make_check(setter, 'player_setter')),
            td(make_check(middle, 'player_middle')),
            td(make_check(outside, 'player_outside')),
            td(make_check(pin_pref, 'player_pin_side')),
            td(make_check(back_row_set, 'player_back_row_set')),
            td(button('\u274c', on_click=self.remove_row_callback)),
            classes=['player_row'],
        )
        when('change', row['.autosave'], handler=share_url)

        self.table.append(row)

    @staticmethod
    def remove_row_callback(event):
        button = event.target
        row = button.closest('.player_row')
        row.remove()

    def read(self):
        names = list(e.value for e in self.table.find('.player_name'))
        checks = []
        for name in ('setter', 'middle', 'outside', 'pin_side', 'back_row_set'):
            checks.append(
                [float(e.value) for e in self.table.find('.player_{}'.format(name))]
            )

        prefs = []
        for s, m, o, p, b in zip(*checks):
            prefs.append((s, m, o, p, b))

        return names, prefs


class OutputTable:
    def __init__(self):
        self.table = self.find_table()

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def find_table():
        result = page.find('#output_table')[0]
        result.append(
            tr(
                th('Names'),
                th('Scores', style=dict(width='35%')),
            )
        )
        return result

    def clear(self):
        self.table.innerHTML = ''
        self.find_table.cache_clear()
        self.table = self.find_table()

    def add_row(self, names, scores):
        self.table.append(
            tr(
                td(names),
                td(scores),
            )
        )


class AddPlayerState:
    COUNT = 1

    def __call__(self, event):
        name = 'Player {}'.format(self.COUNT)
        self.COUNT += 1
        PlayerTable().add_row(name, 1, 1, 1, 0.5, 1.0)


when('click', page['#add_player_button'])(AddPlayerState())


@when('click', page['#calculate_button'])
def calculate(evt):
    t = PlayerTable()
    names, positions = t.read()
    names = np.array(names, dtype=str)
    positions = np.array(positions, dtype=float)

    population = int(page['#settings_population'][0].value)
    iterations = int(page['#settings_iterations'][0].value)
    rows = int(page['#settings_rows'][0].value)
    swap_cost = float(page['#settings_swap_cost'][0].value)
    flex_power = float(page['#settings_flex_power'][0].value)
    p = PermutationFinder(positions, swap_cost=swap_cost, flex_power=flex_power)
    seed = int(pytime.time() * 100) % 2**32
    pop = p.optimize(population=population, rounds=iterations, seed=seed)

    plot = ScorePlotter('#score_plot')
    plot.update(p.last_scores)

    to = OutputTable()
    to.clear()
    handled = set()
    position_remap = {'S': 'B'}
    for k, perm, assignments in pop:
        pnames = names[np.array(perm)]
        roll = -pnames.tolist().index(names[0])
        pnames = np.roll(pnames, roll).tolist()
        assignments = np.roll(assignments, roll)
        pnames = tuple(pnames)
        pnames = tuple(name.title() for name in pnames)
        if pnames not in handled:
            stats = dict(zip(pnames, p.get_assignment_statistics(assignments)))
            augmented_names = []
            for name in pnames:
                player_stats = stats[name]
                details = []
                for pos in 'smopSldr':
                    pos_name = position_remap.get(pos, pos).upper()
                    if not player_stats[pos]:
                        continue
                    elif player_stats[pos] == 1:
                        details.append(pos_name)
                    else:
                        details.append(
                            '{}<sup>{}</sup>'.format(pos_name, player_stats[pos])
                        )
                details = ''.join(details)
                augmented_names.append('{} ({})'.format(name, details))

            score_str = ', '.join(['{:.02f}'.format(-v) for v in k])
            to.add_row(', '.join(augmented_names), score_str)
        handled.add(pnames)
        if len(handled) >= rows:
            break
