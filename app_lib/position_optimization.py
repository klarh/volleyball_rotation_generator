import collections
import functools

import numpy as np

ScoreResult = collections.namedtuple('ScoreResult', ['scores', 'assignments'])


@functools.lru_cache
def get_triplets(N):
    triplets = np.arange(N)[:, None] + np.arange(3)
    triplets %= N
    return triplets


@functools.lru_cache
def get_keyfuns(swap_cost=0.1):
    # sorted order
    return [
        lambda x: -np.sum(x > 0, axis=-1),
        lambda x: -np.sum(x - assignment_penalties[:, None] * swap_cost, axis=-1),
    ]


def get_scores(positions, **keyfun_kwargs):
    keyfuns = get_keyfuns(**keyfun_kwargs)
    # (N_player, 3)
    positions = np.array(positions)
    positions /= np.maximum(1e-6, np.sum(positions, axis=-1, keepdims=True))
    triplets = get_triplets(len(positions))

    # (N_player, 9) -> (N_player, N_assignments, 3)
    assigned_position_values = positions[triplets].reshape((len(positions), 9))[
        :, assignments
    ]
    # (N_keys, N_player, N_assignments)
    keys = np.array([keyfun(assigned_position_values) for keyfun in keyfuns])
    # (N_player, N_assignments) -> N_player
    best_assignments = np.lexsort(keys[::-1])[:, 0]
    # (N_player, N_assignments, N_keys) -> (N_player, N_keys)
    best_scores = keys.transpose([1, 2, 0])[np.arange(len(positions)), best_assignments]
    # -> N_keys
    return ScoreResult(best_scores.mean(axis=0), best_assignments)


def mix_permutation(p1, p2, rng):
    direct = rng.integers(1, len(p1) - 1)
    left = p1[:direct]
    right = tuple([v for v in p2 if v not in set(left)])
    return left + right


class PermutationFinder:
    def __init__(self, positions, cache_size=256000, swap_cost=0.1):
        self.positions = np.asarray(positions)
        self.evaluate_permutation = functools.lru_cache(maxsize=cache_size)(
            self._evaluate_permutation
        )
        self.swap_cost = swap_cost
        self.N = len(self.positions)

    def _evaluate_permutation(self, p):
        p = np.array(p)
        return get_scores(self.positions[p], swap_cost=self.swap_cost)

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

        pop.sort()

        return pop

    @staticmethod
    def get_assignment_statistics(assignments):
        assign_names = np.array(list('smo' 'som' 'mso' 'mos' 'oms' 'osm')).reshape(
            (6, 3)
        )
        position_assignments = [collections.Counter() for _ in range(len(assignments))]
        N = len(assignments)

        for i, j in enumerate(assignments):
            a, b, c = assign_names[j]
            position_assignments[(i + 0) % N][a] += 1
            position_assignments[(i + 1) % N][b] += 1
            position_assignments[(i + 2) % N][c] += 1
        return position_assignments


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
