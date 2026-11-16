import sys
import rospy
import numpy as np
from math import pi
from time import perf_counter
import os

from core.interfaces import ArmController
from lib.potentialFieldPlanner import PotentialFieldPlanner

from lib.loadmap import loadmap
from copy import deepcopy


starts = [np.array([0.0, -1.0, 0.0, -2.0, 0.0, 1.57, 0.0]),
          np.array([0, 0.2, 0, -2.5, 0, 2.7, 0.707]), 
          np.array([-1.08312, 0.18518, 0.79219, -1.73972, 0.01961, 1.78752, 0.66078]),
          np.array([0, -1, 0, -2, 0, 1.57, 0]),
          np.array([1.5, -1, 0, -3, 0, 3.14, 0.707])]
goals = [np.array([1.55063, 0.71112, 0.01359, -1.90341, -0.30752, 2.49269, 0.58051]),
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
        print("usage:\n\tpython potentialField_demo.py 1\n\tpython potentialField_demo.py 2\n\tpython potentialField_demo.py 3 ...")
        exit()

    rospy.init_node('potentialField')

    arm = ArmController()
    index = int(sys.argv[1])-1
    print("Running test "+sys.argv[1])
    print("Moving to Start Position")
    arm.safe_move_to_position(starts[index])
    maps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "maps"))
    map_struct = loadmap(os.path.join(maps_dir, mapNames[index] + ".txt"))
    print("Map = "+ mapNames[index])

    print("Starting to plan")
    start = perf_counter()
    planner = PotentialFieldPlanner()
    path = planner.plan(deepcopy(map_struct), deepcopy(starts[index]), deepcopy(goals[index]))
    stop = perf_counter()
    dt = stop - start
    print("Potential Field took {time:2.2f} sec. Path is.".format(time=dt))
    print(np.round(path,4))
    if path.size == 0:
        print("No path found. Arm will not move.")
        exit()
    input("Press Enter to Send Path to Arm")

    for joint_set in path:
        arm.safe_move_to_position(joint_set)
    print("Trajectory Complete!")
