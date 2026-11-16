"""Lab 4 shared case preparation, validation, recording and optional Gazebo replay."""
import argparse
import ast
import csv
import hashlib
import inspect
import io
import json
import sys
import textwrap
import types
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lib.calculateFK import FK
from lib.loadmap import loadmap
from lib.rrt import rrt, LOWER, UPPER, get_obstacles, isRobotCollided, isEdgeCollided

HERE = Path(__file__).resolve().parent
CASES = HERE / 'lab4_cases.json'


def pf_instance(seed, escapes):
    """Instrument a private in-memory copy of plan; original source stays on disk.

    Capture final locals even when plan returns an empty path. Override only its
    local RNG seed and optional escape limit. Refuse an RRT fallback.
    """
    from lib.potentialFieldPlanner import PotentialFieldPlanner
    planner = PotentialFieldPlanner()
    original = planner.plan.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
            if name == 'rrt':
                raise RuntimeError('Remove RRT fallback from PF before comparing planners.')
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                    if target.id == 'rng':
                        node.value = ast.parse('np.random.default_rng(self._record_seed)', mode='eval').body
                    if target.id == 'max_escapes' and escapes is not None:
                        node.value = ast.Constant(escapes)
    if not {'rng', 'escape_count', 'max_escapes', 'q_path'}.issubset(names):
        raise RuntimeError('PF plan layout changed; expected rng/escape_count/max_escapes/q_path locals.')
    final = ast.parse('self._record_state = locals().copy()').body
    fn.body = [ast.Try(body=fn.body, handlers=[], orelse=[], finalbody=final)]
    fn.decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<recorded PF plan>', 'exec'), namespace)
    planner._record_seed = seed
    planner._record_state = {}
    planner.plan = types.MethodType(namespace[fn.name], planner)
    return planner


def validate(path, start, goal, obstacles):
    if path.size == 0:
        return 'no_path_returned'
    if path.ndim != 2 or path.shape[1] != 7 or not np.isfinite(path).all():
        return 'invalid_path_array'
    if not np.allclose(path[0], start, atol=1e-8, rtol=0):
        return 'wrong_start'
    if not np.allclose(path[-1], goal, atol=1e-8, rtol=0):
        return 'goal_not_reached'
    if any(isRobotCollided(q, obstacles) for q in path):
        return 'invalid_waypoint'
    if any(isEdgeCollided(a, b, obstacles, res=0.01) for a, b in zip(path[:-1], path[1:])):
        return 'edge_collision_fine_check'
    return ''


def run_plan(method, data, start, goal, seed, escapes=None):
    obstacles = get_obstacles(data)
    np.random.seed(seed)
    planner = pf_instance(seed, escapes) if method == 'pf' else None
    log = io.StringIO()
    state = {}
    path = np.empty((0, 7))
    failure = ''
    elapsed = 0.0
    if isRobotCollided(start, obstacles):
        failure = 'invalid_start'
    elif isRobotCollided(goal, obstacles):
        failure = 'invalid_goal'
    else:
        with redirect_stdout(log):
            begin = perf_counter()
            try:
                path = np.asarray(planner.plan(data, start.copy(), goal.copy()) if planner
                                  else rrt(data, start.copy(), goal.copy(), verbose=True), dtype=float)
            except Exception as error:
                failure = f'exception: {type(error).__name__}: {error}'
            finally:
                elapsed = perf_counter() - begin
        state = planner._record_state if planner else {}
        if not failure:
            failure = validate(path, start, goal, obstacles)
        if failure == 'no_path_returned':
            reasons = [line for line in log.getvalue().splitlines()
                       if 'PF stopped:' in line or 'gave up after' in line]
            if reasons:
                failure = reasons[-1]
            elif state.get('iteration', -1) >= state.get('field_budget', float('inf')):
                failure = 'iteration_budget_exhausted'
            elif state.get('stagnation', 0) >= 25 and state.get('escape_count', 0) >= state.get('max_escapes', float('inf')):
                failure = 'escape_limit_reached'
            else:
                failure = 'no_path_returned_reason_unreported'
    return path, failure, elapsed, state, log.getvalue()


def prepare():
    if CASES.exists():
        raise SystemExit(f'{CASES} already exists. Rename it before preparing a new test set.')
    s = [0, -1, 0, -2, 0, 1.57, 0]
    g = [1.5, 0.5, 0, -2, 0, 2.5, 0.707]
    tasks = {
        'A': dict(map='emptyMap', start=s, goal=g),
        'B': None,
        'C': dict(map='map1', start=s, goal=[-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7]),
        'D': dict(map='map2', start=[0, 0.2, 0, -2.5, 0, 2.7, 0.707], goal=[1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707]),
        'E': dict(map='map3', start=[-1.08312, .18518, .79219, -1.73972, .01961, 1.78752, .66078], goal=[1.02977, .70564, -.73863, -1.60708, .11197, 1.93308, .79836]),
        'F': None,
        'G': dict(map='map5', start=[1.5, -1, 0, -3, 0, 3.14, .707], goal=[-1.5, 0, 1, -1, 1, 3.14, .707]),
        'H': None,
    }
    rng = np.random.default_rng(2026)
    for letter, name, method in [('B', 'map1', 'pf'), ('F', 'map4', 'rrt')]:
        data = loadmap(str(ROOT / 'maps' / (name + '.txt')))
        obstacles = get_obstacles(data)
        old = np.array(g)
        blocked = np.array([1.58439, .85187, .02265, -1.83902, -.51254, 2.48782, .49619])
        for attempt in range(200):
            if letter == 'B':
                start = np.array(s, dtype=float)
                goal = old + (attempt / 30) * (blocked - old) if attempt <= 30 else old + rng.normal(0, .25, 7)
            else:
                start = np.array(s, dtype=float) if attempt < 100 else rng.uniform(LOWER, UPPER)
                goal = rng.uniform(LOWER, UPPER)
            start, goal = np.round(start, 5), np.round(goal, 5)
            if isRobotCollided(start, obstacles) or isRobotCollided(goal, obstacles):
                continue
            if not isEdgeCollided(start, goal, obstacles):
                continue
            print(f'Preparing {letter}: checking candidate {attempt + 1}/200', flush=True)
            path, failure, _, _, _ = run_plan(method, data, start, goal, 42)
            if not failure:
                tasks[letter] = dict(map=name, start=start.tolist(), goal=goal.tolist(),
                                     selection=f'{method} success using preparation seed 42')
                break
        if tasks[letter] is None:
            raise SystemExit(f'No verified candidate for {letter}. Send this output and your maps/{name}.txt plus current PF planner. No case file written.')
    tasks['H'] = dict(map='map1', start=tasks['B']['goal'], goal=tasks['B']['start'])
    for letter, task in tasks.items():
        data = loadmap(str(ROOT / 'maps' / (task['map'] + '.txt')))
        obstacles = get_obstacles(data)
        start, goal = np.array(task['start']), np.array(task['goal'])
        if isRobotCollided(start, obstacles) or isRobotCollided(goal, obstacles):
            raise SystemExit(f'{letter}: invalid endpoint. No case file written.')
        task['direct'] = not isEdgeCollided(start, goal, obstacles)
        if letter != 'A' and task['direct']:
            raise SystemExit(f'{letter}: unexpectedly direct. No case file written.')
        task['map_sha256'] = hashlib.sha256((ROOT / 'maps' / (task['map'] + '.txt')).read_bytes()).hexdigest()
    CASES.write_text(json.dumps(tasks, indent=2), encoding='utf-8')
    print(f'Prepared A-H: {CASES}')
    print(json.dumps(tasks, indent=2))


def save_path(filename, path):
    fk = FK()
    with filename.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['point'] + [f'q{i}' for i in range(7)] + ['ee_x_m', 'ee_y_m', 'ee_z_m'])
        if path.ndim == 2 and path.shape[1] == 7 and np.isfinite(path).all():
            for i, q in enumerate(path):
                _, pose = fk.forward(q)
                writer.writerow([i] + q.tolist() + pose[:3, 3].tolist())


def main(method='rrt'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?', choices=list('ABCDEFGH') + ['all'])
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--escapes', type=int, default=None, help='PF only; 0 disables random escape')
    parser.add_argument('--execute', action='store_true', help='Replay one successful path in running Gazebo')
    parser.add_argument('--output', type=Path, default=HERE / 'experiment_data')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if not args.case or args.runs < 1 or (args.escapes is not None and args.escapes < 0):
        parser.error('Choose A-H/all, positive runs, and nonnegative escapes.')
    if args.execute and (args.case == 'all' or args.runs != 1):
        parser.error('--execute requires one case and --runs 1')
    if method == 'rrt' and args.escapes is not None:
        parser.error('--escapes applies to PF only')
    if not CASES.exists():
        parser.error('Run python3 rrt_record.py --prepare first.')
    tasks = json.loads(CASES.read_text())
    folder = args.output / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + method)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'cases.json').write_text(json.dumps(tasks, indent=2))
    sources = {}
    for name in ['rrt.py', 'potentialFieldPlanner.py', 'calculateFK.py', 'calculateFKJac.py', 'detectCollision.py', 'loadmap.py']:
        file = ROOT / 'lib' / name
        if file.exists():
            sources[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (folder / 'settings.json').write_text(json.dumps(dict(method=method, seed=args.seed,
        runs=args.runs, escapes=args.escapes, source_sha256=sources, validation_res_rad=.01), indent=2))
    columns = ['case', 'map', 'method', 'seed', 'success', 'planning_time_s', 'joint_path_length_rad',
               'waypoints', 'failure_reason', 'escape_count', 'escape_limit', 'iterations', 'direct', 'execution_status']
    with (folder / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for letter in (list('ABCDEFGH') if args.case == 'all' else [args.case]):
            task = tasks[letter]
            map_file = ROOT / 'maps' / (task['map'] + '.txt')
            if hashlib.sha256(map_file.read_bytes()).hexdigest() != task['map_sha256']:
                raise RuntimeError('Map changed since preparation: ' + str(map_file))
            data = loadmap(str(map_file))
            start, goal = np.array(task['start']), np.array(task['goal'])
            for seed in range(args.seed, args.seed + args.runs):
                path, failure, elapsed, state, log = run_plan(method, data, start, goal, seed, args.escapes)
                stem = f'{letter}_seed{seed}'
                np.save(folder / (stem + '_path.npy'), path)
                save_path(folder / (stem + '_path.csv'), path)
                partial = np.asarray(state.get('q_path', np.empty((0, 7))))
                if failure and partial.size:
                    save_path(folder / (stem + '_partial.csv'), partial)
                (folder / (stem + '_log.txt')).write_text(log, encoding='utf-8')
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if not failure else ''
                execution = 'not_requested'
                if args.execute and not failure:
                    import rospy
                    from core.interfaces import ArmController
                    rospy.init_node(method + '_record')
                    arm = ArmController()
                    input(f'Confirm Gazebo map={task["map"]}; Enter to move to start: ')
                    arm.safe_move_to_position(start)
                    if not np.allclose(arm.get_positions(), start, atol=.03, rtol=0):
                        execution = 'start_not_reached'
                    else:
                        input('Start reached. Enter to execute path: ')
                        execution = 'completed'
                        for q in path:
                            arm.safe_move_to_position(q)
                            if not np.allclose(arm.get_positions(), q, atol=.03, rtol=0):
                                execution = 'waypoint_not_reached'
                                break
                writer.writerow(dict(case=letter, map=task['map'], method=method, seed=seed,
                    success=not bool(failure), planning_time_s=elapsed, joint_path_length_rad=length,
                    waypoints=len(path), failure_reason=failure, escape_count=state.get('escape_count', '') if method == 'pf' else '',
                    escape_limit=state.get('max_escapes', ''), iterations=state.get('iteration', ''),
                    direct=task['direct'], execution_status=execution))
                stream.flush()
                print(f'{letter} {method} seed={seed}: success={not bool(failure)}, {elapsed:.3f}s, {failure}', flush=True)
    print(f'Results saved: {folder}')


if __name__ == '__main__':
    main()
