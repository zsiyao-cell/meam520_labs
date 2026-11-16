
import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        from lib.calculateFK import FK
        self.fk = FK()

        pass

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))

        # Use the same FK convention as calculateFK.py from Lab 1.
        physical_positions, end_effector_T = self.fk.forward(q)
        Ai = self.compute_Ai(q)

        # The planner uses seven physical joint points and two virtual gripper
        # points as its nine control points. The end effector is stored last.
        jointPositions[:7, :] = physical_positions[:7, :]
        for i in range(7):
            T0e[i] = Ai[i]

        # Two virtual control points on opposite sides of the gripper.
        # Coordinates below are in the end-effector frame (meters).
        virtual_offsets = (
            np.array([0.0, 0.04, 0.105]),
            np.array([0.0, -0.04, 0.105]),
        )
        for i, offset in enumerate(virtual_offsets):
            T_virtual = end_effector_T.copy()
            T_virtual[:3, 3] = end_effector_T[:3, 3] + end_effector_T[:3, :3] @ offset
            T0e[7 + i] = T_virtual
            jointPositions[7 + i] = T_virtual[:3, 3]

        jointPositions[9, :] = physical_positions[7, :]
        T0e[9] = end_effector_T

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def linear_jacobians(self, q):
        """
        Compute linear velocity Jacobians of the 10 expanded control points.

        INPUT:
        q - 1x7 numpy array of joint angles

        OUTPUT:
        Jv - 10 x 3 x 7 numpy array, where Jv[k] maps joint velocities
             to the linear velocity of control point k in the world frame.
        """
        q = np.asarray(q, dtype=float).reshape(7)
        points, _ = self.forward_expanded(q)
        axes = self.fk.get_axis_of_rotation(q)
        Ai = self.compute_Ai(q)

        # Joint j rotates about z_{j-1} through the previous frame origin.
        origins = np.zeros((7, 3))
        for j in range(1, 7):
            origins[j] = Ai[j - 1][:3, 3]

        Jv = np.zeros((10, 3, 7))
        for k in range(10):
            # Physical control point k is upstream of joints k,...,6.
            # Both virtual gripper points and the end effector depend on all 7 joints.
            count = k if k < 7 else 7
            for j in range(count):
                Jv[k, :, j] = np.cross(
                    axes[:, j], points[k] - origins[j]
                )
        return Jv

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return self.fk.compute_Ai(q)[1:]

    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",joint_positions)
