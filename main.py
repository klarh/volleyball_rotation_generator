import numpy as np
import app_lib

from pyscript import web

for name in ('population', 'iterations', 'rows'):
    if name in app_lib.state.QUERY_ARGS:
        web.page['#settings_{}'.format(name)].value = app_lib.state.QUERY_ARGS[name]

names = app_lib.state.names
positions = app_lib.state.positions

t = app_lib.ui.PlayerTable()
for (name, pos) in zip(names, positions):
    t.add_row(name, *pos)
