import sys
import rospy
import numpy as np
from math import pi
from time import perf_counter

from core.interfaces import ArmController
from lib.rrt import rrt

from lib.loadmap import loadmap
from copy import deepcopy
import os


starts = [np.array([0, -1, 0, -2, 0, 1.57, 0]),
          np.array([0, 0.2, 0, -2.5, 0, 2.7, 0.707]), 
          np.array([-1.08312, 0.18518, 0.79219, -1.73972, 0.01961, 1.78752, 0.66078]),
          np.array([0, -1, 0, -2, 0, 1.57, 0]),
          np.array([1.5, -1, 0, -3, 0, 3.14, 0.707])]
goals = [np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7]),
         np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707]),
         np.array([1.02977, 0.70564, -0.73863, -1.60708, 0.11197, 1.93308, 0.79836]),
         np.array([1.58439, 0.85187, 0.02265, -1.83902, -0.51254, 2.48782, 0.49619]),
         np.array([-1.5, 0.0, 1.0, -1.0, 1.0, 3.14, 0.707])]
mapNames = ["map1",
            "map2",
            "map3",
            "map1",
            "map3"]

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython rrt_demo.py 1\n\tpython rrt_demo.py 2\n\tpython rrt_demo.py 3 ...")
        exit()

    rospy.init_node('RRT')

    arm = ArmController()
    index = int(sys.argv[1])-1
    print("Running test "+sys.argv[1])
    print("Moving to Start Position")
    arm.safe_move_to_position(starts[index])
    here = os.path.dirname(os.path.abspath(__file__))
    map_file = None
    for rel in ("../../maps", "../maps"):    
        cand = os.path.join(here, rel, mapNames[index] + ".txt")
        if os.path.exists(cand):
            map_file = cand
            break
    if map_file is None:
        print("Could not find " + mapNames[index] + ".txt in ../../maps or ../maps")
        exit()
    map_struct = loadmap(map_file)
    print("Map = "+ mapNames[index])

    print("Starting to plan")
    start = perf_counter()
    path = rrt(deepcopy(map_struct), deepcopy(starts[index]), deepcopy(goals[index]), verbose=True)
    stop = perf_counter()
    dt = stop - start

    if len(path) == 0:
        print("RRT took {time:2.2f} sec. No path found (invalid start/goal or max iterations reached).".format(time=dt))
        exit()

    print("RRT took {time:2.2f} sec. Path has {n} waypoints. Path is.".format(time=dt, n=len(path)))
    print(np.round(path, 4))
    input("Press Enter to Send Path to Arm")

    for joint_set in path:
        arm.safe_move_to_position(joint_set)
    print("Trajectory Complete!")
