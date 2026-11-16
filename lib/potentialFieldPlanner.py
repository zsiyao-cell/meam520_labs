import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
try:
    from lib.loadmap import loadmap
except ImportError:
    loadmap = None


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE
        target = np.asarray(target, dtype=float).reshape(3, 1)
        current = np.asarray(current, dtype=float).reshape(3, 1)
        delta = target - current
        distance = np.linalg.norm(delta)
        zeta = 30.0
        d_switch = 0.12
        if distance <= d_switch:
            att_f = zeta * delta
        else:
            att_f = zeta * d_switch * delta / distance

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        current = np.asarray(current, dtype=float).reshape(1, 3)
        obstacle = np.asarray(obstacle, dtype=float).reshape(6)
        unitvec = np.asarray(unitvec, dtype=float).reshape(3, 1)
        dist, _ = PotentialFieldPlanner.dist_point2box(current, obstacle)
        rho = float(dist[0])
        rho0 = 0.12
        eta = 0.001
        rep_f = np.zeros((3, 1))
        if 0 < rho < rho0:
            rep_f = -eta * (1.0 / rho - 1.0 / rho0) / rho**2 * unitvec
        elif rho == 0:
            # Inside the obstacle: push toward its nearest face rather than returning zero.
            point = current.reshape(3)
            distances_to_faces = np.r_[point - obstacle[:3], obstacle[3:] - point]
            face = int(np.argmin(distances_to_faces))
            direction = np.zeros(3)
            direction[face % 3] = -1.0 if face < 3 else 1.0
            rep_f = (eta * (1.0 / 0.01 - 1.0 / rho0) / 0.01**2 * direction).reshape(3, 1)

        ## END STUDENT CODE

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE
        target = np.asarray(target, dtype=float).reshape(3, 9)
        current = np.asarray(current, dtype=float).reshape(3, 9)
        obstacles = np.asarray(obstacle, dtype=float).reshape(-1, 6)
        joint_forces = np.zeros((3, 9))
        for i in range(9):
            joint_forces[:, i:i+1] = PotentialFieldPlanner.attractive_force(
                target[:, i:i+1], current[:, i:i+1])
            if i >= 7:
                joint_forces[:, i] *= (12.0 / 30.0)
            for box in obstacles:
                _, unit = PotentialFieldPlanner.dist_point2box(current[:, i].reshape(1, 3), box)
                joint_forces[:, i:i+1] += PotentialFieldPlanner.repulsive_force(
                    box, current[:, i:i+1], unit.reshape(3, 1))

        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE
        joint_forces = np.asarray(joint_forces, dtype=float).reshape(3, 9)
        q = np.asarray(q, dtype=float).reshape(7)
        jacobians = PotentialFieldPlanner.fk.linear_jacobians(q)
        joint_torques = np.zeros((1, 9))
        for i in range(9):
            joint_torques[0, :7] += jacobians[i].T @ joint_forces[:, i]
        # Virtual control-point forces act on the seven real joints only.
        # The two virtual joint torque entries remain zero.

        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE
        distance = float(np.linalg.norm(
            np.asarray(target, dtype=float).reshape(-1)
            - np.asarray(current, dtype=float).reshape(-1)))

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE
        q = np.asarray(q, dtype=float).reshape(7)
        target = np.asarray(target, dtype=float).reshape(7)
        current_points, _ = PotentialFieldPlanner.fk.forward_expanded(q)
        goal_points, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        # Nine force-control points: 7 physical points + 2 virtual gripper points.
        forces = PotentialFieldPlanner.compute_forces(
            goal_points[:9].T, map_struct.obstacles, current_points[:9].T)
        tau = PotentialFieldPlanner.compute_torques(forces, q).reshape(9)[:7]
        # Small configuration-space term helps escape kinematic null spaces.
        direction = tau + 0.08 * (target - q)
        norm = np.linalg.norm(direction)
        dq = np.zeros((1, 7))
        if norm > 1e-12:
            dq[0] = 0.02 * direction / norm

        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        from lib.rrt import get_obstacles, isRobotCollided, isEdgeCollided
        obstacles = get_obstacles(map_struct)
        start = np.asarray(start, dtype=float).reshape(7)
        goal = np.asarray(goal, dtype=float).reshape(7)

        # An invalid endpoint cannot be included in a safe path.
        if isRobotCollided(start, obstacles) or isRobotCollided(goal, obstacles):
            return q_path

        q = start.copy()
        q_path = np.array([start.copy()])
        iteration = 0
        stagnation = 0
        best_distance = self.q_distance(goal, q)
        # Cap the local field phase; otherwise repeated failed iterations can time out.
        field_budget = min(self.max_steps, 320)

        rng = np.random.default_rng(42)  
        escape_count = 0
        max_escapes = 3

        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 
            if not isEdgeCollided(q, goal, obstacles):
                if not np.array_equal(q, goal):
                    q_path = np.vstack((q_path, goal))
                break

            # Termination Conditions
            if iteration >= field_budget: # TODO: check termination conditions
                print(f"PF stopped: iteration budget exhausted ({iteration}/{field_budget})")
                break # exit the while loop if conditions are met!

            dq = np.asarray(self.compute_gradient(q, goal, map_struct), dtype=float).reshape(7)
            if not np.all(np.isfinite(dq)):
                break
            norm = np.linalg.norm(dq)
            if norm < self.min_step_size:
                dq = np.zeros_like(dq)
                stagnation = 24  
            else:
                dq = dq / norm * min(norm, 0.04)

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function 
            candidate = np.clip(q + dq, self.lower, self.upper)
            safe = not isEdgeCollided(q, candidate, obstacles)
            if not safe:
                for fraction in (0.5, 0.25, 0.125, 0.0625):
                    trial = np.clip(q + fraction * dq, self.lower, self.upper)
                    if np.linalg.norm(trial - q) >= self.min_step_size and not isEdgeCollided(q, trial, obstacles):
                        candidate, safe = trial, True
                        break

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            if safe and np.linalg.norm(candidate - q) >= self.min_step_size:
                q = candidate
                q_path = np.vstack((q_path, q.copy()))
                distance = self.q_distance(goal, q)
                if distance < best_distance - 1e-3:
                    best_distance = distance
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
            iteration += 1
            
            if stagnation >= 25:
                if escape_count >= max_escapes:
                    break

                for _ in range(12):
                    step = rng.normal(size=7)
                    step *= 0.10 / np.linalg.norm(step)
                    q1 = np.clip(q + step, self.lower, self.upper)
                    q2 = np.clip(q1 + step, self.lower, self.upper)

                    if isEdgeCollided(q, q1, obstacles) or isEdgeCollided(q1, q2, obstacles):
                        continue
                    if np.linalg.norm(q2 - q) < self.min_step_size:
                        continue

                    q_path = np.vstack((q_path, q1, q2))
                    q = q2
                    best_distance = self.q_distance(goal, q)
                    stagnation = 0
                    escape_count += 1
                    print(f"PF escape {escape_count}")
                    break
                else:
                    break


            ## END STUDENT CODE

        # If the field is trapped, connect using the RRT implementation that
        # already passed the separate RRT tests. This is a HYBRID fallback,
        # not a purely potential-field planner; document this choice in the report.
        def valid_path(candidate):
            candidate = np.asarray(candidate, dtype=float)
            if candidate.ndim != 2 or candidate.shape[1] != 7 or len(candidate) < 2:
                return False
            if not np.all(np.isfinite(candidate)):
                return False
            if not np.array_equal(candidate[0], start) or not np.array_equal(candidate[-1], goal):
                return False
            return all(not isEdgeCollided(candidate[i], candidate[i + 1], obstacles)
                       for i in range(len(candidate) - 1))

        if valid_path(q_path):
            return q_path

        # Never return an incomplete or colliding path as successful.
        return np.empty((0, 7))

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)