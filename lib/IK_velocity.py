import numpy as np
from lib.calcJacobian import calcJacobian
from lib.FK_velocity import FK_velocity


def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))
    desired_velocity = np.zeros((6,1))

    J = calcJacobian(q_in)

    v_in = v_in.reshape((3, 1))
    omega_in = omega_in.reshape((3, 1))

    desired_velocity = np.concatenate((
        np.asarray(v_in).reshape(3,),
        np.asarray(omega_in).reshape(3,)
    ))

    mask = ~np.isnan(desired_velocity)

    J_reduced = J[mask, :]
    desired_reduced = desired_velocity[mask]

    if not np.any(mask):
        dq = np.zeros((1, 7))
    else:
        dq_solution = np.linalg.lstsq(
            J_reduced, desired_reduced, rcond=None
        )[0]
     
        dq = dq_solution.reshape(1, 7) 

    return dq



if __name__ == "__main__":

    q = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])

    # Desired velocity
    v = np.array([0.1, 0.0, 0.0])
    omega = np.array([np.nan, np.nan, np.nan])

    # Step 1: Inverse velocity kinematics
    dq = IK_velocity(q, v, omega)

    # Step 2: Forward velocity kinematics
    v_actual, omega_actual = FK_velocity(q, dq.reshape(7,))

    # Step 3: Compare results
    print("Desired linear velocity:", v)
    print("Actual linear velocity:", v_actual)
    print("Linear velocity error:",
          np.linalg.norm(v - np.asarray(v_actual).reshape(3,)))
