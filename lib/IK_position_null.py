import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
from lib.IK_velocity import IK_velocity


class IK:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = (
        lower + (upper - lower) / 2
    )  # compute middle of range of motion of each joint
    fk = FK()

    def __init__(
        self, linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5
    ):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        displacement = np.zeros(3)
        axis = np.zeros(3)

        p_des = target[:3, 3]
        p_curr = current[:3, 3]
        displacement = p_des - p_curr

        R_des = target[:3, :3]
        R_curr = current[:3, :3]
        axis = calcAngDiff(R_des, R_curr)

        ## END STUDENT CODE
        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H
        """

        ## STUDENT CODE STARTS HERE
        distance = 0
        angle = 0

        distance = np.linalg.norm(G[:3, 3] - H[:3, 3])

        R_G = G[:3, :3]
        R_H = H[:3, :3]
        R_rel = R_G.T @ R_H

        cos_theta = (np.trace(R_rel) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle = acos(cos_theta)

        ## END STUDENT CODE
        return distance, angle

    def is_valid_solution(self, q, target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        ## STUDENT CODE STARTS HERE
        success = False
        message = "Solution found/not found + reason"

        within_limits = np.all((q >= self.lower) & (q <= self.upper))

        _, current = self.fk.forward(q)

        distance, angle = self.distance_and_angle(target, current)

        position_valid = distance < self.linear_tol
        orientation_valid = angle < self.angular_tol

        success = within_limits and position_valid and orientation_valid

        if success:
            message = "Solution found"
        elif not within_limits:
            message = "Solution violates joint limits"
        elif not position_valid:
            message = "Position error exceeds tolerance"
        else:
            message = "Orientation error exceeds tolerance"

        ## END STUDENT CODE
        return success, message

    ####################
    ## Task Functions ##
    ####################


    @staticmethod
    def end_effector_task(q, target, method="J_pseudo"):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        _, current = IK.fk.forward(q)

        displacement, axis = IK.displacement_and_axis(target, current)

        if method == "J_pseudo":
            dq = IK_velocity(q, displacement, axis).reshape(7)

        elif method == "J_trans":
            J = calcJacobian(q)
            error = np.concatenate((displacement, axis))
            dq = J.T @ error

        else:
            raise ValueError("Unknown IK method: " + str(method))

        return dq


    @staticmethod
    def joint_centering_task(q, rate=5e-1):
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset  # proportional term (implied quadratic cost)

        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, alpha, method="J_pseudo"):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = np.asarray(seed, dtype=float).reshape(7).copy()
        rollout = []

        ## STUDENT CODE STARTS HERE

        ## gradient descent:
        for step in range(self.max_steps):
            rollout.append(q.copy())

            # Primary Task - Achieve End Effector Pose
            dq_ik = IK.end_effector_task(q, target, method=method)

            # Secondary Task - Center Joints
            dq_center = IK.joint_centering_task(q)

            ## Task Prioritization
            J = calcJacobian(q)
            J_pinv = np.linalg.pinv(J)

            N = np.eye(7) - J_pinv @ J

            dq = dq_ik + N @ dq_center

            # Check termination conditions
            if np.linalg.norm(dq) < self.min_step_size:
                break

            q = q + alpha * dq

            # update q

        ## END STUDENT CODE

        success, message = self.is_valid_solution(q, target)
        return q, rollout, success, message


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    target = np.array(
        [
            [0, -1, 0, -0.2],
            [-1, 0, 0, 0],
            [0, 0, -1, 0.5],
            [0, 0, 0, 1],
        ]
    )

    # Using pseudo-inverse
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(
        target, seed, alpha=0.5, method="J_trans"
    )

    for i, q_pseudo in enumerate(rollout_pseudo):
        joints, pose = ik.fk.forward(q_pseudo)
        d, ang = IK.distance_and_angle(target, pose)
        print(
            "iteration:",
            i,
            " q =",
            q_pseudo,
            " d={d:3.4f}  ang={ang:3.3f}".format(d=d, ang=ang),
        )

    # compare
    print("\n   Success: ", success_pseudo, ":  ", message_pseudo)
    print("   Solution: ", q_pseudo)
    print("   #Iterations : ", len(rollout_pseudo))
