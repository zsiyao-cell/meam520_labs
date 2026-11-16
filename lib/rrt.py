import numpy as np
from lib.calculateFK import FK
from lib.detectCollision import detectCollision

fk = FK()

# Panda joint limits (rad)
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

LINK_MARGIN = 0.02            # obstacles are inflated by this much to account for link thickness
HAND_HALF_WIDTH = 0.10        # gripper approximated by a segment of this half-width ...
HAND_BACK = 0.105             # ... located this far behind the end-effector frame


def inflate(box, m=LINK_MARGIN):
    box = np.asarray(box, dtype=float)
    return np.concatenate([box[:3] - m, box[3:] + m])


def get_obstacles(map_struct):
    return np.array(map_struct.obstacles, dtype=float).reshape(-1, 6)


def inJointLimits(q):
    return bool(np.all(q >= LOWER) and np.all(q <= UPPER))


def robotSegments(q):
    """Line-segment model of the arm + gripper. Returns (starts, ends), each N x 3."""
    pos, T0e = fk.forward(q)
    R, p = T0e[:3, :3], T0e[:3, 3]
    v1 = p + R @ np.array([0, HAND_HALF_WIDTH, -HAND_BACK])
    v2 = p + R @ np.array([0, -HAND_HALF_WIDTH, -HAND_BACK])
    mid = p + R @ np.array([0, 0, -HAND_BACK])
    starts = np.vstack([pos[:-1], v1[None], pos[7][None]])
    ends = np.vstack([pos[1:], v2[None], mid[None]])
    return starts, ends


def isRobotCollided(q, obstacles):
    """True if configuration q is in collision with any obstacle or violates joint limits."""
    if not inJointLimits(q):
        return True
    starts, ends = robotSegments(q)
    for box in obstacles:
        if np.any(detectCollision(starts, ends, inflate(box))):
            return True
    return False


def isEdgeCollided(q1, q2, obstacles, res=0.01):
    """Check the straight joint-space segment q1->q2 by sampling every `res` rad (max joint diff)."""
    n = max(2, int(np.ceil(np.max(np.abs(q2 - q1)) / res)) + 1)
    for t in np.linspace(0, 1, n):
        if isRobotCollided(q1 + t * (q2 - q1), obstacles):
            return True
    return False


class _Tree:
    def __init__(self, root, cap=20000):
        self.nodes = np.zeros((cap, 7))
        self.nodes[0] = root
        self.parent = [-1]
        self.n = 1

    def nearest(self, q):
        return int(np.argmin(np.linalg.norm(self.nodes[:self.n] - q, axis=1)))

    def add(self, q, parent):
        if self.n >= len(self.nodes):
            self.nodes = np.vstack([self.nodes, np.zeros_like(self.nodes)])
        self.nodes[self.n] = q
        self.parent.append(parent)
        self.n += 1
        return self.n - 1

    def trace(self, idx):      # node -> root
        out = []
        while idx != -1:
            out.append(self.nodes[idx].copy())
            idx = self.parent[idx]
        return out


TRAPPED, ADVANCED, REACHED = 0, 1, 2


def _extend(tree, q_target, obstacles, step):
    i = tree.nearest(q_target)
    q_near = tree.nodes[i]
    d = q_target - q_near
    dist = np.linalg.norm(d)
    if dist <= step:
        q_new, status = q_target.copy(), REACHED
    else:
        q_new, status = q_near + step * d / dist, ADVANCED
    if isEdgeCollided(q_near, q_new, obstacles):
        return TRAPPED, -1
    return status, tree.add(q_new, i)


def _connect(tree, q_target, obstacles, step):
    status, idx = ADVANCED, -1
    while status == ADVANCED:
        status, idx = _extend(tree, q_target, obstacles, step)
    return status, idx


def _shortcut(path, obstacles, iters=150):
    path = [p for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))
        if not isEdgeCollided(path[i], path[j], obstacles):
            path = path[:i + 1] + path[j:]
    return path


def rrt(map_struct, start, goal, step=0.3, max_iter=6000, smooth=True, verbose=False):
    """
    RRT-Connect (bidirectional RRT) in the 7-D joint space.
    Returns q_path (N x 7), first row = start, last row = goal.
    Returns an empty (0 x 7) array if start/goal are invalid or no path is found.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    obstacles = get_obstacles(map_struct)
    empty = np.array([]).reshape(0, 7)

    if isRobotCollided(start, obstacles):
        if verbose: print("RRT: start configuration is in collision / outside joint limits")
        return empty
    if isRobotCollided(goal, obstacles):
        if verbose: print("RRT: goal configuration is in collision / outside joint limits")
        return empty
    if not isEdgeCollided(start, goal, obstacles):
        return np.array([start, goal])

    Ta, Tb = _Tree(start), _Tree(goal)
    a_is_start = True

    for it in range(max_iter):
        q_rand = LOWER + np.random.rand(7) * (UPPER - LOWER)
        status, ia = _extend(Ta, q_rand, obstacles, step)
        if status != TRAPPED:
            status2, ib = _connect(Tb, Ta.nodes[ia], obstacles, step)
            if status2 == REACHED:
                path_a = Ta.trace(ia)[::-1]     # root_a -> meeting point
                path_b = Tb.trace(ib)           # meeting point -> root_b
                full = path_a + path_b[1:] if np.allclose(path_a[-1], path_b[0]) else path_a + path_b
                if not a_is_start:
                    full = full[::-1]
                full[0], full[-1] = start.copy(), goal.copy()
                if verbose: print("RRT: connected after {i} iterations".format(i=it + 1))
                if smooth:
                    full = _shortcut(full, obstacles)
                return np.array(full)
        Ta, Tb = Tb, Ta
        a_is_start = not a_is_start

    if verbose:
        print("RRT: gave up after {m} iterations (start tree {a} nodes, goal tree {b} nodes)".format(
            m=max_iter, a=(Ta if a_is_start else Tb).n, b=(Tb if a_is_start else Ta).n))
    return empty


if __name__ == "__main__":
    from collections import namedtuple
    M = namedtuple("map", ["obstacles"])
    m = M(np.array([[0.2, -0.5, 0.2, 0.4, 0.5, 0.5]]))
    s = np.array([0, -1, 0, -2, 0, 1.57, 0]); g = np.array([-1.2, 0.4, 0, -1.5, 0, 1.8, 0.7])
    p = rrt(m, s, g)
    print(p.shape)
