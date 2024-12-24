from pyscript import document
import js
import urllib

QUERY_ARGS = dict(urllib.parse.parse_qsl(js.location.search[1:]))

import numpy as np
import app_lib

# setter, middle, outside
positions = np.random.random((7, 3))
names = np.array(['Player {}'.format(chr(ord('a') + i)) for i in range(len(positions))])

t = app_lib.ui.PlayerTable()
for (name, pos) in zip(names, positions):
    t.add_row(name, *pos)
