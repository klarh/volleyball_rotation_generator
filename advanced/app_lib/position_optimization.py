import numpy as np
import functools
import itertools
import collections

base_systems = [
    'smoldr',  # 4-2 basic
    'SmoldS',  # 6-2 (setter hitting when in front row)
    'pmolds',  # 5-1 setter back row
    'smoldp',  # 5-1 setter front row
]

position_names = list(sorted(set(''.join(base_systems))))

swap_count_dict = {
    (0, 1, 2): 0,
    (0, 2, 1): 1,
    (1, 0, 2): 1,
    (1, 2, 0): 2,
    (2, 0, 1): 2,
    (2, 1, 0): 3,
}

system_swap_counts = {}
for base in base_systems:
    base_front, base_back = base[:3], base[3:]
    for front_perm, back_perm in list(
        itertools.product(
            itertools.permutations(base_front), itertools.permutations(base_back)
        )
    ):
        front_idx = tuple(base_front.index(c) for c in front_perm)
        back_idx = tuple(base_back.index(c) for c in back_perm)
        total_swaps = swap_count_dict[front_idx] + swap_count_dict[back_idx]

        key = ''.join(front_perm) + ''.join(back_perm)
        system_swap_counts[key] = total_swaps

all_systems = np.array(list(sorted(system_swap_counts)))
n_systems = len(all_systems)
back_row_setter_systems = np.array(
    ['s' not in name[:3] or 'S' in name for name in all_systems], dtype=np.float32
)
system_swap_array = np.array(
    [system_swap_counts[key] for key in all_systems], dtype=np.float32
)
system_permutation_array = np.array(
    [[position_names.index(c) for c in name] for name in all_systems]
)

# Create indices for the 3 dimensions: (System, Position, Role)
sys_indices = np.arange(n_systems)[:, None]  # Shape (N, 1)
pos_indices = np.arange(6)[None, :]  # Shape (1, 6)
# Use fancy indexing to set the ones
system_permutation_onehot = np.zeros((n_systems, 6, len(position_names)))
system_permutation_onehot[sys_indices, pos_indices, system_permutation_array] = 1

position_preference = np.dtype(
    [
        ('ternary', (np.float32, 3)),
        ('pin_preference', np.float32),
        ('back_row_set', np.float32),
    ]
)


def get_scores(prefs, flex_power=0.5, swap_cost=0.1):
    N = len(prefs)

    ternary = np.array(prefs['ternary'])
    ternary /= np.sum(ternary, axis=-1, keepdims=True)
    ternary = ternary**flex_power

    player_position_scores = dict(
        # front row setter
        s=ternary[:, 0],
        # middle
        m=ternary[:, 1],
        # outside
        o=ternary[:, 2] * (1 - prefs['pin_preference']),
        # oppo
        p=ternary[:, 2] * prefs['pin_preference'],
        # setter (back row)/oppo (front row)
        S=0.5 * (ternary[:, 0] + ternary[:, 2] * prefs['pin_preference']),
    )
    player_position_scores['l'] = player_position_scores['o']
    player_position_scores['d'] = player_position_scores['m']
    player_position_scores['r'] = player_position_scores['s']

    player_position_array = np.array(
        [player_position_scores[c] for c in position_names]
    ).T

    all_rotations = np.arange(N)[:, None] + np.arange(6)[None, :]
    all_rotations %= N

    rotation_prefs = player_position_array[all_rotations]

    scores = (
        np.einsum('r p c, y p c -> r y', rotation_prefs, system_permutation_onehot) / 6
    )

    back_set_comfort = prefs['back_row_set'][all_rotations]  # Look up comfort
    avg_comfort = np.exp(
        np.mean(np.log(back_set_comfort + 1e-9), axis=-1, keepdims=True)
    )

    system_comfort_factor = (1 - back_row_setter_systems) + (
        back_row_setter_systems * avg_comfort
    )

    penalty_term = system_swap_array * swap_cost

    scores = (scores * system_comfort_factor) - penalty_term
    return scores


def mix_permutation(p1, p2, rng):
    direct = rng.integers(1, len(p1) - 1)
    left = p1[:direct]
    right = tuple([v for v in p2 if v not in set(left)])
    return left + right


class PermutationFinder:
    def __init__(self, preferences, cache_size=256000, flex_power=0.5, swap_cost=0.1):
        basic_preferences = np.asarray(preferences)

        preferences = np.zeros(len(preferences), dtype=position_preference)
        preferences['ternary'][:] = basic_preferences[:, :3]
        preferences['pin_preference'] = basic_preferences[:, 3]
        preferences['back_row_set'] = basic_preferences[:, 4]
        self.preferences = preferences

        self.evaluate_permutation = functools.lru_cache(maxsize=cache_size)(
            self._evaluate_permutation
        )
        self.flex_power = flex_power
        self.swap_cost = swap_cost
        self.N = len(self.preferences)

    def _evaluate_permutation(self, p):
        p = np.array(p)
        scores = get_scores(
            self.preferences[p], flex_power=self.flex_power, swap_cost=self.swap_cost
        )
        score_summary = scores.max(axis=-1).mean()
        return (np.array([-score_summary]), scores)

    def perm_to_key(self, perm):
        # perm = tuple(perm.tolist())
        (scores, assignments) = self.evaluate_permutation(perm)
        return (tuple(scores.tolist()), perm, assignments)

    def optimize(self, population=128, seed=13, rounds=128):
        rng = np.random.default_rng(seed)

        pop = []
        for _ in range(population):
            perm = tuple(rng.permutation(self.N).tolist())
            pop.append(self.perm_to_key(perm))

        # permutation -> index
        live_perms = collections.defaultdict(set)
        for i, (scores, perm, _) in enumerate(pop):
            live_perms[perm].add(i)

        best_score = max(v[0][0] for v in pop)
        best_scores = [best_score]
        for i, j in rng.choice(population, size=(rounds, 2)):
            if i == j:
                j = (i + 1) % population
            mi, mj = pop[i], pop[j]
            mix = mix_permutation(mi[1], mj[1], rng)
            mix = self.perm_to_key(mix)
            if mix < mi and mix[1] not in live_perms:
                live_perms[mi[1]].remove(i)
                if not live_perms[mi[1]]:
                    del live_perms[mi[1]]
                pop[i] = mix
                live_perms[mix[1]].add(i)

            best_score = max(best_score, mix[0][0])
            best_scores.append(best_score)

        pop.sort()
        self.last_scores = best_scores

        return pop

    @staticmethod
    def get_assignment_statistics(scores):
        best_system = np.argmax(scores, axis=-1)
        chosen_systems = all_systems[best_system]

        N = scores.shape[0]
        position_counts = [collections.Counter() for _ in range(N)]
        for offset, system_name in enumerate(chosen_systems):
            for i, c in enumerate(system_name):
                i = (i + offset) % N
                position_counts[i][c] += 1
        return position_counts


# ---- net ----
assignments = np.array(
    [
        # o m s
        [(0, 0), (1, 1), (2, 2)],
        # m o s
        [(0, 0), (1, 2), (2, 1)],
        # o s m
        [(0, 1), (1, 0), (2, 2)],
        # m s o
        [(0, 1), (1, 2), (2, 0)],
        # s m o
        [(0, 2), (1, 1), (2, 0)],
        # s o m
        [(0, 2), (1, 0), (2, 1)],
    ]
)
assignment_penalties = np.array([0.0, 1, 1, 2, 3, 2])
assignments = 3 * assignments[..., 0] + assignments[..., 1]
