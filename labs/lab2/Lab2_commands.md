# Lab 2 Commands

Start the simulation or connect the robot using the course instructions first.

## Open the code folder

```bash
cd ~/meam520_ws/src/meam520_labs/labs/lab2
```

## Position only

```bash
python follow_experiment.py --mode position_only --environment sim --ry 0.20 --rz 0.15 --f 1 --cycles 3
```

## Position + orientation

```bash
python follow_experiment.py --mode full_pose --environment sim --ry 0.20 --rz 0.15 --f 1 --cycles 3
```

Use the same parameters and starting configuration for both runs.

- `--ry`, `--rz`: ellipse radii in meters.
- `--f`: angular frequency in rad/s.
- `--cycles`: number of loops.
- Hardware: change `sim` to `hw` after simulation checks. This only changes the file label, not the robot connection.

## Find saved data

```bash
ls -lt experiment_data/
```

CSV = recorded data. TXT = settings and summary.

Press `Ctrl+C` to interrupt the program; this does not replace the robot E-stop.
