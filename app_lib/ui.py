import functools
import time as pytime

from pyscript import when
from pyscript.web import *
import numpy as np
from .position_optimization import PermutationFinder


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
                th('Setter'),
                th('Middle'),
                th('Outside'),
            )
        )
        return result

    def add_row(self, name, setter, middle, outside):
        make_check = lambda x, cls: input_(
            type='range',
            classes=[cls, 'value_slider'],
            min=0,
            max=1,
            step='0.001',
            value=x,
        )
        name_elt = input_(value=str(name), classes=['player_name'])
        self.table.append(
            tr(
                td(name_elt),
                td(make_check(setter, 'player_setter')),
                td(make_check(middle, 'player_middle')),
                td(make_check(outside, 'player_outside')),
                td(button('\u274c', on_click=self.remove_row_callback)),
                classes=['player_row'],
            )
        )

    @staticmethod
    def remove_row_callback(event):
        button = event.target
        row = button.closest('.player_row')
        row.remove()

    def read(self):
        names = list(e.value for e in self.table.find('.player_name'))
        checks = []
        for name in ('setter', 'middle', 'outside'):
            checks.append(
                [float(e.value) for e in self.table.find('.player_{}'.format(name))]
            )

        prefs = []
        for s, m, o in zip(*checks):
            prefs.append((s, m, o))

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
        PlayerTable().add_row(name, 1, 1, 1)


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
    p = PermutationFinder(positions)
    seed = int(pytime.time() * 100) % 2**32
    pop = p.optimize(population=population, rounds=iterations, seed=seed)

    to = OutputTable()
    to.clear()
    handled = set()
    for k, perm in pop:
        pnames = names[np.array(perm)]
        pnames = np.roll(pnames, -pnames.tolist().index(names[0])).tolist()
        pnames = tuple(pnames)
        pnames = tuple(name.title() for name in pnames)
        if pnames not in handled:
            to.add_row(
                ', '.join(pnames), (', '.join(['{:.02f}'.format(-v) for v in k]))
            )
        handled.add(pnames)
        if len(handled) >= rows:
            break
