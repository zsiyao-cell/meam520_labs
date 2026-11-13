import argparse, csv, os
from datetime import datetime
from lib.calculateFK import FK
import sys
from math import pi, sin, cos
import numpy as np
from time import perf_counter

import rospy
import roslib
import tf
import geometry_msgs.msg

from core.interfaces import ArmController

from lib.IK_position_null import IK


rospy.init_node("visualizer")

# Using your solution code
ik = IK()

#########################
##  RViz Communication ##
#########################

tf_broad  = tf.TransformBroadcaster()

# Broadcasts a frame using the transform from given frame to world frame
def show_pose(H,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(H),
        tf.transformations.quaternion_from_matrix(H),
        rospy.Time.now(),
        frame,
        "world"
    )

#############################
##  Transformation Helpers ##
#############################

def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])

#################
##  IK Targets ##
#################

# TODO: Try testing your own targets!

# Note: below we are using some helper functions which make it easier to generate
# valid transformation matrices from a translation vector and Euler angles, or a
# sequence of successive rotations around z, y, and x. You are free to use these
# to generate your own tests, or directly write out transforms you wish to test.

targets = [
    transform( np.array([-.2, -.3, .5]), np.array([0,pi,pi])            ),
    transform( np.array([-.2, .3, .5]),  np.array([pi/6,5/6*pi,7/6*pi]) ),
    transform( np.array([.5, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.7, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi])            ),
    transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi-pi/2])       ),
    transform( np.array([.2, -.6, 0.5]), np.array([0,pi-pi/2,pi])       ),
    transform( np.array([.2, -.6, 0.5]), np.array([pi/4,pi-pi/2,pi])    ),
    transform( np.array([.5, 0, 0.2]),   np.array([0,pi-pi/2,pi])       ),
    transform( np.array([.4, 0, 0.2]),   np.array([pi/2,pi-pi/2,pi])    ),
]

####################
## Test Execution ##
####################

np.set_printoptions(suppress=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['sim','hw'], required=True)
    parser.add_argument('--method', choices=['both','J_pseudo','J_trans'], default='both')
    parser.add_argument('--targets', choices=['original','new'], default='new')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=float, nargs=7)
    parser.add_argument('--warm-start', action='store_true')
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--move', action='store_true', help='Execute successful solutions; otherwise solve only')
    parser.add_argument('--output-dir', default='experiment_data')
    args = parser.parse_args(rospy.myargv()[1:])
    if not 1 <= args.count <= 10: parser.error('count must be 1..10')
    arm = ArmController()
    neutral = np.asarray(arm.neutral_position()).copy()
    base_seed = np.array(args.seed) if args.seed else neutral.copy()
    if np.any(base_seed < ik.lower) or np.any(base_seed > ik.upper): parser.error('seed outside joint limits')
    fk = FK()
    if args.targets == 'new':
        offsets = np.array([[.2,0,0,0,0,0,0],[-.2,0,0,0,0,0,0],
            [0,.15,0,0,0,0,0],[0,-.15,0,0,0,0,0],[0,0,.2,0,0,0,0],
            [0,0,-.2,0,0,0,0],[0,0,0,.15,0,0,0],[0,0,0,-.15,0,0,0],
            [0,0,0,0,.2,.1,0],[0,0,0,0,-.2,-.1,.15]])
        targets = [fk.forward(neutral+d)[1] for d in offsets]
    os.makedirs(args.output_dir,exist_ok=True)
    stem = os.path.join(args.output_dir,'ik_'+args.environment+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    def errors(q,target):
        T = fk.forward(q)[1]
        return [np.linalg.norm(T[:3,3]-target[:3,3]), np.arccos(np.clip((np.trace(target[:3,:3].T @ T[:3,:3])-1)/2,-1,1))]
    methods = ['J_pseudo','J_trans'] if args.method == 'both' else [args.method]
    with open(stem+'.csv','w',newline='') as f, open(stem+'_rollout.csv','w',newline='') as h:
        w,c = csv.writer(f),csv.writer(h)
        w.writerow(['environment','target_set','target_id','method','warm_start','alpha','runtime_s','rollout_count','success','message','position_error_m','orientation_error_rad','executed']+['seed'+str(i) for i in range(7)]+['q'+str(i) for i in range(7)]+['target'+str(i) for i in range(16)]+['q_measured'+str(i) for i in range(7)]+['measured_position_error_m','measured_orientation_error_rad'])
        c.writerow(['target_id','method','sample','kind','position_error_m','orientation_error_rad']+['q'+str(i) for i in range(7)])
        for method in methods:
            seed = base_seed.copy()
            for i,target in enumerate(targets[:args.count]):
                used = seed.copy() if args.warm_start else base_seed.copy()
                start = perf_counter()
                q,rollout,success,message = ik.inverse(target,used.copy(),method=method,alpha=args.alpha)
                elapsed = perf_counter()-start
                measured = np.full(7,np.nan); executed=False
                print(method,i+1,success,elapsed,len(rollout),message)
                show_pose(target,'target')
                if args.move and success:
                    input('Check target in simulation first. Enter to execute target %d...' % (i+1))
                    arm.safe_move_to_position(q)
                    measured=np.asarray(arm.get_positions()); executed=True
                w.writerow([args.environment,args.targets,i+1,method,args.warm_start,args.alpha,elapsed,len(rollout),success,message]+errors(q,target)+[executed]+list(used)+list(q)+list(target.flatten())+list(measured)+(errors(measured,target) if executed else [np.nan,np.nan]))
                f.flush()
                for j,qr in enumerate(rollout): c.writerow([i+1,method,j,'rollout']+errors(qr,target)+list(qr))
                c.writerow([i+1,method,len(rollout),'final']+errors(q,target)+list(q)); h.flush()
                if args.warm_start: seed=q.copy() if success else base_seed.copy()
    print('Saved:',stem+'.csv',stem+'_rollout.csv')
