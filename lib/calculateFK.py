import numpy as np
from math import pi


class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        self.a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
        self.alpha = [-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0]
        self.d = [0.333, 0, 0.316, 0, 0.384, 0, 0.210]

        self.joint_offsets = [
            np.array([0,0,0.141]),
            np.array([0,0,0]),
            np.array([0,0,0.195]),
            np.array([0,0,0]),
            np.array([0,0,0.125]),
            np.array([0,0,-0.015]),
            np.array([0,0,0.051]),
            np.array([0,0,0])
        ]


        pass

    def dh_transform(self, a, alpha, d, theta):
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)

        T = np.array([
            [c_theta, -s_theta * c_alpha,  s_theta * s_alpha,  a*c_theta],
            [s_theta, c_theta * c_alpha,  -c_theta * s_alpha,  a*s_theta],
            [0,       s_alpha,             c_alpha,            d],
            [0,       0,                   0,                  1]
        ])

        return T


    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)


        theta = [
            q[0],
            q[1],
            q[2],
            q[3],
            q[4],
            q[5],
            q[6] - pi/4,
        ]

        A = []
        T = []
        for i in range(7):
            A.append(
                self.dh_transform(self.a[i], self.alpha[i], self.d[i], theta[i])
                ) 
            if i==0:
                T.append(A[i]) 
            else:
                T.append(T[i-1] @ A[i])
        
        T0e = T[6]

        for j in range(8):
            if j == 0:
                jointPositions[j, :] = self.joint_offsets[j]
            else:
                jointPositions[j, :] = T[j-1][:3, 3] + T[j-1][:3,:3] @ self.joint_offsets[j]
            
 

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        Ai = self.compute_Ai(q)

        axis_of_rotation_list = np.zeros((3, 7))

        z0 = np.array([[0],[0],[1]])

        for i in range(7):
            z_i = Ai[i][:3,2]
            axis_of_rotation_list[:,i] = z_i

        return axis_of_rotation_list

    
    def compute_Ai(self, q):

        theta = [
            q[0],
            q[1],
            q[2],
            q[3],
            q[4],
            q[5],
            q[6] - np.pi/4
        ]

        Ai = [np.eye(4)]

        for i in range(7):

            A = self.dh_transform(
                self.a[i],
                self.alpha[i],
                self.d[i],
                theta[i]
            )

            T = Ai[-1] @ A

            Ai.append(T)

        return Ai

    
if __name__ == "__main__":

    fk = FK()

    np.set_printoptions(precision=4, suppress=True)

    # matches figure in the handout

    q = np.array([0,0,0,0,0,0,0])

    Ai = fk.compute_Ai(q)

    axes = fk.get_axis_of_rotation(q)

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)

    print("Ai length:", len(Ai))
    print("Ai matches FK:", np.allclose(Ai[-1], T0e))
    print("Axes shape:", axes.shape)
    print("Axes:", axes)
    print("Axis norms:", np.linalg.norm(axes, axis=0))
