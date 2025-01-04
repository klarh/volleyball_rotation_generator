import urllib

import numpy as np
import js
from pyscript import when
from pyscript import web
from . import ui

QUERY_ARGS = dict(urllib.parse.parse_qsl(js.location.search[1:]))

positions = names = None
if 'players' in QUERY_ARGS:
    players = QUERY_ARGS['players'].split(',')
    positions = []
    names = []
    for i, elt in enumerate(players):
        if i % 4 == 0:
            names.append(elt)
            positions.append([])
        else:
            try:
                positions[-1].append(float(elt))
            except ValueError:
                positions[-1].append(1.0)

    if len(positions[-1]) != 3:
        positions.pop()

if not positions or not names:
    # setter, middle, outside
    positions = np.random.random((7, 3))
    names = np.array(
        ['Player {}'.format(chr(ord('a') + i)) for i in range(len(positions))]
    )


@when('click', web.page['#share_button'])
def share_url():
    query = {}

    for name in ('population', 'iterations', 'rows', 'swap_cost'):
        query[name] = web.page['#settings_{}'.format(name)][0].value

    table = ui.PlayerTable()
    names, positions = table.read()
    player_bits = []
    for name, pos in zip(names, np.round(positions, 3)):
        player_bits.append(name)
        for p in pos:
            player_bits.append('{:.03f}'.format(p))
    query['players'] = ','.join(player_bits)

    parsed_url = urllib.parse.urlparse(str(js.location))
    parsed_url = parsed_url._replace(
        query='&'.join(['{}={}'.format(k, v) for (k, v) in query.items()])
    )
    js.history.pushState('', '', parsed_url.geturl())
