import argparse
import csv
from datetime import datetime
from pathlib import Path
from threading import Lock
import numpy as np
import rospy
from math import cos, sin, pi
import matplotlib.pyplot as plt
import geometry_msgs
import visualization_msgs
from tf.transformations import quaternion_from_matrix

from core.interfaces import ArmController
from core.utils import time_in_seconds

from lib.IK_velocity import IK_velocity
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff

######################
## Rotation Helpers ##
######################


def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis.
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


## NOTE: calcAngDiff() MOVED TO calcAngDiff.py


##################
## Follow class ##
##################


class JacobianDemo:
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function
    """

    active = False  # When to stop commanding arm
    start_time = 0  # start time
    dt = 0.03  # constant for how to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher(
        "/vis/trace", geometry_msgs.msg.PointStamped, queue_size=10
    )
    ellipsoid_pub = rospy.Publisher(
        "/vis/ellip", visualization_msgs.msg.Marker, queue_size=10
    )
    counter = 0
    x0 = np.array([0.307, 0, 0.487])  # corresponds to neutral position
    last_iteration_time = None

    def __init__(self, mode="full_pose"):
        self.mode = mode
        self.active = False
        self.start_time = 0.0
        self.last_iteration_time = None
        self.counter = 0
        self.rows = []
        self.lock = Lock()
        self.last_measured_q = None

    @staticmethod
    def log_header():
        columns = ["time_s", "mode"]
        columns += ["q%d_rad" % (i + 1) for i in range(7)]
        columns += ["qdot_cmd%d_rad_s" % (i + 1) for i in range(7)]
        for prefix in ("pd", "pa", "vd", "v_cmd", "omega_cmd"):
            columns += ["%s_%s" % (prefix, axis) for axis in "xyz"]
        for prefix in ("Rd", "Ra"):
            columns += ["%s_%d%d" % (prefix, i + 1, j + 1)
                        for i in range(3) for j in range(3)]
        return columns

    def save_log(self, args, duration):
        """Save samples and experiment parameters after commanding has stopped."""
        folder = Path(args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = folder / ("%s_ellipse_%s_%s.csv" % (args.environment, self.mode, stamp))
        with self.lock:
            samples = list(self.rows)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.log_header())
            writer.writerows(samples)

        # Sidecar keeps the CSV strictly rectangular and easy to load with pandas.
        metadata_path = path.with_suffix(".txt")
        with metadata_path.open("w") as stream:
            stream.write("data_file: %s\n" % path.name)
            stream.write("environment_label: %s (entered by operator)\n" % args.environment)
            stream.write("mode: %s\n" % self.mode)
            stream.write("f_rad_s: %s\nry_m: %s\nrz_m: %s\n" % (args.f, args.ry, args.rz))
            stream.write("cycles: %s\nramp_s: %s\nplanned_duration_s: %.9g\n" %
                         (args.cycles, args.ramp, duration))
            stream.write("x0_m: %s\n" % self.x0.tolist())
            stream.write("samples: %d\n" % len(samples))
            if samples:
                stream.write("q_first_rad: %s\n" % samples[0][2:9])
                stream.write("q_last_rad: %s\n" % samples[-1][2:9])
                stream.write("joint_displacement_norm_rad: %.9g\n" %
                             np.linalg.norm(np.array(samples[-1][2:9]) -
                                            np.array(samples[0][2:9])))
        print("Saved %d samples to %s" % (len(samples), path.resolve()))
        print("Saved parameters to %s" % metadata_path.resolve())

    @staticmethod
    def experiment_ellipse(t, f, ry, rz, duration, ramp):
        """Ellipse with smooth start AND finish at x0; f is angular frequency."""
        t = np.clip(t, 0.0, duration)
        u = min(t / ramp, 1.0)
        w = min((duration - t) / ramp, 1.0)
        def smooth(z):
            return 3 * z**2 - 2 * z**3
        def smooth_derivative(z):
            return 6 * z * (1 - z) / ramp if z < 1.0 else 0.0
        a, b = smooth(u), smooth(w)
        a_dot = smooth_derivative(u)
        b_dot = -smooth_derivative(w)
        s = a * b
        s_dot = a_dot * b + a * b_dot
        phase = f * t
        offset = np.array([0.0, ry * cos(phase), rz * sin(phase)])
        offset_dot = np.array([0.0, -ry * f * sin(phase), rz * f * cos(phase)])
        xdes = JacobianDemo.x0 + s * offset
        vdes = s * offset_dot + s_dot * offset
        angle = -pi + 0.1 * s * sin(phase)
        Rdes = rotvec_to_matrix(np.array([angle, 0.0, 0.0]))
        ang_vdes = np.array([0.1 * (s_dot * sin(phase) + s * f * cos(phase)),
                             0.0, 0.0])
        return Rdes, ang_vdes, xdes, vdes

    ##################
    ## TRAJECTORIES ##
    ##################

    def line(t, f=0.5, L=0.35):
        """
        Calculate the position and velocity of the line trajector

        Inputs:
        t - time in sec since start
        f - frequency in rad/s of the line trajectory
        L - length of the line in meters

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        Rdes = 3x3 np array of target end effector orientation in the world frame
        ang_vdes = 0x3 np array of target end effector orientation velocity in the
        rotation vector representation in the world frame
        """
        # Linear trajectory along z-axis
        xdes = JacobianDemo.x0 + np.array([
            0,
            0,
            (L / 2) * sin(f * t)
        ])

        vdes = np.array([
            0,
            0,
            (L / 2) * f * cos(f * t)
        ])

        # Orientation trajectory: rotate around x-axis
        ang = -np.pi + (np.pi / 4.0) * sin(f * t)

        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)

        ang_v = (np.pi / 4.0) * f * cos(f * t)

        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

        return Rdes, ang_vdes, xdes, vdes

    def eight(t, fx=0.5, fy=1.0, rx=0.15, ry=0.1):
        """
        Calculate the position and velocity of the figure 8 trajectory

        Inputs:
        t - time in sec since start
        fx - frequency in rad/s of the x portion
        fy - frequency in rad/s of the y portion
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        Rdes = 3x3 np array of target end effector orientation in the world frame
        ang_vdes = 0x3 np array of target end effector orientation velocity in the rotation vector representation in the world frame
        """
        # TODO: Implement linear and orientation components of this trajectory
        # Test extensively in simulation before your robot lab!

        # Linear position trajectory
        xdes = JacobianDemo.x0 + np.array([
            rx * sin(fx * t),
            ry * sin(fy * t),
            0
        ])

        # Linear velocity trajectory
        vdes = np.array([
            rx * fx * cos(fx * t),
            ry * fy * cos(fy * t),
            0
        ])

        # Orientation trajectory: small oscillation about x-axis
        ang = -np.pi + 0.1 * sin(fx * t)

        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)

        # Desired angular velocity
        ang_v = 0.1 * fx * cos(fx * t)

        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

        ## END STUDENT CODE
        return Rdes, ang_vdes, xdes, vdes

    def ellipse(t, f=1, ry=.2, rz=.15):
        """
        Calculate the position and velocity of the ellipse trajectory, in the world y and z

        Inputs:
        t - time in sec since start
        f - frequency in rad/s of the ellipse
        ry - y-axis length
        rz - z-axis length

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """
        # TODO: Implement ellipse trajectory

        # Ellipse in the y-z plane
        phase = f * t

        offset = np.array([
            0,
            ry * cos(phase),
            rz * sin(phase)
        ])

        offset_dot = np.array([
            0,
            -ry * f * sin(phase),
            rz * f * cos(phase)
        ])

        # Smooth start over first 3 seconds
        ramp_time = 3.0
        u = np.clip(t / ramp_time, 0.0, 1.0)

        s = 3 * u**2 - 2 * u**3

        if t < ramp_time:
            s_dot = (6 * u - 6 * u**2) / ramp_time
        else:
            s_dot = 0.0

        # Desired linear position and velocity
        xdes = JacobianDemo.x0 + s * offset
        vdes = s * offset_dot + s_dot * offset

        # Orientation trajectory
        ang = -np.pi + 0.1 * s * sin(phase)

        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)

        # Angular velocity, including ramp derivative
        ang_v = 0.1 * (
            s_dot * sin(phase) + s * f * cos(phase)
        )

        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

        ## END STUDENT CODE
        return Rdes, ang_vdes, xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "endeffector"
        msg.point.x = 0
        msg.point.y = 0
        msg.point.z = 0
        self.point_pub.publish(msg)

    ################
    ## CONTROLLER ##
    ################

    def follow_trajectory(self, state, trajectory):

        if self.active:
            try:
              with self.lock:
                if not self.active:
                    return
                t = time_in_seconds() - self.start_time

                # get desired trajectory position and velocity
                Rdes, ang_vdes, xdes, vdes = trajectory(t)

                # get current end effector position
                q = np.asarray(state["position"], dtype=float).reshape(7)
                self.last_measured_q = q.copy()

                joints, T0e = self.fk.forward(q)

                R = T0e[:3, :3]
                x = T0e[0:3, 3]
                curr_x = np.copy(x.flatten())

                # First Order Integrator, Proportional Control with Feed Forward
                kp = 0.01
                v = vdes + kp * (xdes - curr_x)

                # Rotation
                kr = 0.01
                if self.mode == "position_only":
                    # IK_velocity uses NaN angular components to remove constraints.
                    omega = np.full(3, np.nan)
                else:
                    omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()

                # Velocity Inverse Kinematics
                dq = IK_velocity(q, v, omega).flatten()

                # Get the correct timing to update with the robot
                if self.last_iteration_time == None:
                    self.last_iteration_time = time_in_seconds()

                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()

                new_q = q + self.dt * dq

                arm.safe_set_joint_positions_velocities(new_q, dq)

                # Sample the measured state, not the integrated position command.
                row = [float(t), self.mode]
                row += q.tolist() + dq.tolist()
                row += xdes.tolist() + curr_x.tolist()
                row += vdes.tolist() + v.tolist() + omega.tolist()
                row += Rdes.reshape(-1).tolist() + R.reshape(-1).tolist()
                self.rows.append(row)

                # Downsample visualization to reduce rendering overhead
                self.counter = self.counter + 1
                if self.counter == 10:
                    self.show_ee_position()
                    self.counter = 0

            except rospy.exceptions.ROSException:
                pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite ellipse tracking with CSV logging")
    parser.add_argument("--mode", choices=("position_only", "full_pose"), required=True)
    parser.add_argument("--environment", choices=("sim", "hw"), required=True,
                        help="Label for data files; does not change the ROS connection")
    parser.add_argument("--f", type=float, default=1.0, help="Angular frequency in rad/s")
    parser.add_argument("--ry", type=float, default=0.15, help="Y radius in meters")
    parser.add_argument("--rz", type=float, default=0.15, help="Z radius in meters")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--ramp", type=float, default=3.0, help="Start/end taper in seconds")
    parser.add_argument("--output-dir", default="experiment_data")
    args = parser.parse_args()
    if args.f <= 0 or args.ry <= 0 or args.rz <= 0 or args.cycles <= 0 or args.ramp <= 0:
        parser.error("f, ry, rz, cycles, and ramp must be positive")
    duration = args.cycles * 2 * pi / args.f
    if 2 * args.ramp >= duration:
        parser.error("two ramps must be shorter than total cycle duration")

    rospy.init_node("follower")
    JD = JacobianDemo(mode=args.mode)
    trajectory = lambda t: JD.experiment_ellipse(
        t, args.f, args.ry, args.rz, duration, args.ramp)
    callback = lambda state: JD.follow_trajectory(state, trajectory)
    arm = ArmController(on_state_callback=callback)

    try:
        print("Resetting arm to neutral position...")
        arm.safe_move_to_position(arm.neutral_position())
        with JD.lock:
            JD.start_time = time_in_seconds()
            JD.active = True
        print("Running %s for %.2f s; Ctrl+C stops early and saves data." %
              (args.mode, duration))
        while not rospy.is_shutdown() and time_in_seconds() - JD.start_time < duration:
            rospy.sleep(0.05)
    except (KeyboardInterrupt, rospy.exceptions.ROSInterruptException):
        print("Interrupted; saving the available samples.")
    finally:
        with JD.lock:
            JD.active = False
            # Send a zero-velocity command when leaving the trajectory loop.
            if JD.last_measured_q is not None and not rospy.is_shutdown():
                try:
                    arm.safe_set_joint_positions_velocities(
                        JD.last_measured_q, np.zeros(7))
                except rospy.exceptions.ROSException as exc:
                    print("Could not send stop command: %s" % exc)
        JD.save_log(args, duration)

