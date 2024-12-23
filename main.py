from pyscript import document
import js
import urllib

QUERY_ARGS = dict(urllib.parse.parse_qsl(js.location.search[1:]))

import numpy as np
import app_lib

# setter, middle, outside
positions = np.array([
    (1., .1, .2), # matthew
    (.1, 0, 1), # enrique
    (0, .3, 1), # zack
    (0, 0, 1), # elvis
    (0, 1, 1e-2), # sam
    (0, 1, 1e-2), # johnny
    (1, 0, 1e-2), # ivan
    (1, 0, .5), # byron
])

names = np.array(['matthew', 'enrique', 'zack', 'elvis', 'sam', 'johnny', 'ivan', 'byron'])

t = app_lib.ui.PlayerTable()
for (name, pos) in zip(names, positions):
    t.add_row(name, *pos)
