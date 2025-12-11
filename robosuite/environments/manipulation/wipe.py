import multiprocessing
from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import WipeArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from scipy.fftpack import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Default Wipe environment configuration
DEFAULT_WIPE_CONFIG = {
    # settings for reward
    "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision (except the wiping tool) with the table (originally -10.0)
    "wipe_contact_reward": 0.01,  # reward for contacting something with the wiping tool
    "unit_wiped_reward": 50.0,  # reward per peg wiped
    "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.05,  # penalty for each step that the force is over the safety threshold (0.05 originally)
    "distance_multiplier": 5.0,  # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
    "distance_th_multiplier": 5.0,  # multiplier in the tanh function for the aforementioned reward
    # settings for table top
    "table_full_size": [0.5, 0.8, 0.05],  # Size of tabletop
    "table_offset": [0.15, 0, 0.9],  # Offset of table (z dimension defines max height of table)
    "table_friction": [1.0, 0.1, 0.01],  # Friction parameters for the table (originally [0.03, 0.005, 0.0001])
    "table_friction_std": 0,  # Standard deviation to sample different friction parameters for the table each episode
    "table_height": 0.0,  # Additional height of the table over the default location
    "table_height_std": 0.0,  # Standard deviation to sample different heigths of the table each episode
    "line_width": 0.04,  # Width of the line to wipe (diameter of the pegs)
    "two_clusters": False,  # if the dirt to wipe is one continuous line or two
    "coverage_factor": 0.6,  # how much of the table surface we cover
    "num_markers": 100,  # How many particles of dirt to generate in the environment (originally 100)
    # settings for thresholds
    "contact_threshold": 1.0,  # Minimum eef force to qualify as contact [N]
    "pressure_threshold": 0.5,  # force threshold (N) to overcome to get increased contact wiping reward
    "pressure_threshold_max": 60.0,  # maximum force allowed (N)
    # misc settings
    "print_results": True,  # Whether to print results or not
    "get_info": True,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,  # Whether we allow for early terminations or not
    "use_condensed_obj_obs": True,  # Whether to use condensed object observation representation (only applicable if obj obs is active)
}


class Wipe(ManipulationEnv):
    """
    This class corresponds to the Wiping task for a single robot arm

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory.
            For this environment, setting a value other than the default ("WipingGripper") will raise an
            AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
            parameters, see the default configuration dict at the top of this file.
            If None is specified, the default configuration will be used.

        Raises:
            AssertionError: [Gripper specified]
            AssertionError: [Bad reward specification]
            AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="WipingGripper",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        task_config=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        # Assert that the gripper type is None
        # assert (
        #     gripper_types == "WipingGripper"
        # ), "Tried to specify gripper other than WipingGripper in Wipe environment!"

        # Get config
        self.task_config = task_config if task_config is not None else DEFAULT_WIPE_CONFIG

        # Set task-specific parameters

        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config["arm_limit_collision_penalty"]
        self.wipe_contact_reward = self.task_config["wipe_contact_reward"]
        self.unit_wiped_reward = self.task_config["unit_wiped_reward"]
        self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        self.excess_force_penalty_mul = self.task_config["excess_force_penalty_mul"]
        self.distance_multiplier = self.task_config["distance_multiplier"]
        self.distance_th_multiplier = self.task_config["distance_th_multiplier"]
        # Final reward computation
        # So that is better to finish that to stay touching the table for 100 steps
        # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
        self.task_complete_reward = self.unit_wiped_reward * (self.wipe_contact_reward + 0.5)
        # Verify that the distance multiplier is not greater than the task complete reward
        assert (
            self.task_complete_reward > self.distance_multiplier
        ), "Distance multiplier cannot be greater than task complete reward!"

        # settings for table top
        self.table_full_size = self.task_config["table_full_size"]
        self.table_height = self.task_config["table_height"]
        self.table_height_std = self.task_config["table_height_std"]
        delta_height = min(0, np.random.normal(self.table_height, self.table_height_std))  # sample variation in height
        self.table_offset = np.array(self.task_config["table_offset"]) + np.array((0, 0, delta_height))
        self.table_friction = self.task_config["table_friction"]
        self.table_friction_std = self.task_config["table_friction_std"]
        self.line_width = self.task_config["line_width"]
        self.two_clusters = self.task_config["two_clusters"]
        self.coverage_factor = self.task_config["coverage_factor"]
        self.num_markers = self.task_config["num_markers"]

        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold = self.task_config["pressure_threshold"]
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]

        # misc settings
        self.print_results = self.task_config["print_results"]
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.use_contact_obs = self.task_config["use_contact_obs"]
        self.early_terminations = self.task_config["early_terminations"]
        self.use_condensed_obj_obs = self.task_config["use_condensed_obj_obs"]

        # settings added for tuli project
        self.force_history_horizon = 30
        self.extended_action_step_size = 30
        self.force_history = []
        self.all_peak_freqs = []
        self.contact_history = []
        self.ideal_peak_freq = 0.4
        self.peak_freq_tolerance = 0.1

        # Scale reward if desired (see reward method for details)
        self.reward_normalization_factor = horizon / (
            self.num_markers * self.unit_wiped_reward + horizon * (self.wipe_contact_reward + self.task_complete_reward)
        )

        # set other wipe-specific attributes
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0
        self.metadata = []
        self.spec = "spec"

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

        # Parameters for wiping skill
        self.contact_force_threshold = 1.0  # N
        self.lateral_step_size = 0.5  # 1cm per step #0.05
        self.vertical_step_size = 3.0  # 0.5cm per step
        self.max_steps_without_contact = 500  # Maximum steps to try reaching contact


        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        # set after init to ensure self.robots is set
        self.ee_force_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}
        self.ee_torque_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}

    # TODO: Implement this method
    def _get_active_markers(self, c_geoms):
        return self.model.mujoco_arena.markers
    
    # def _get_active_markers(self, c_geoms):
    #     """
    #     Get the markers that are currently being wiped by the tool

    #     Args:
    #         c_geoms (list): List of corner geoms for the tool

    #     Returns:
    #         list: List of active markers
    #     """
    #     active_markers = []
    #     corner1_id = self.sim.model.geom_name2id(c_geoms[0])
    #     corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
    #     corner2_id = self.sim.model.geom_name2id(c_geoms[1])
    #     corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
    #     corner3_id = self.sim.model.geom_name2id(c_geoms[2])
    #     corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
    #     corner4_id = self.sim.model.geom_name2id(c_geoms[3])
    #     corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])

    #     # Unit vectors on my plane
    #     v1 = corner1_pos - corner2_pos
    #     v1 /= np.linalg.norm(v1)
    #     v2 = corner4_pos - corner2_pos
    #     v2 /= np.linalg.norm(v2)

    #     # Corners of the tool in the coordinate frame of the plane
    #     t1 = np.array([np.dot(corner1_pos - corner2_pos, v1), np.dot(corner1_pos - corner2_pos, v2)])
    #     t2 = np.array([np.dot(corner2_pos - corner2_pos, v1), np.dot(corner2_pos - corner2_pos, v2)])
    #     t3 = np.array([np.dot(corner3_pos - corner2_pos, v1), np.dot(corner3_pos - corner2_pos, v2)])
    #     t4 = np.array([np.dot(corner4_pos - corner2_pos, v1), np.dot(corner4_pos - corner2_pos, v2)])

    #     pp = [t1, t2, t4, t3]

    #     # Normal of the plane defined by v1 and v2
    #     n = np.cross(v1, v2)
    #     n /= np.linalg.norm(n)

    #     def isLeft(P0, P1, P2):
    #         return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

    #     def PointInRectangle(X, Y, Z, W, P):
    #         return isLeft(X, Y, P) < 0 and isLeft(Y, Z, P) < 0 and isLeft(Z, W, P) < 0 and isLeft(W, X, P) < 0

    #     # Only go into this computation if there are contact points
    #     if self.sim.data.ncon != 0:

    #         # Check each marker that is still active
    #         for marker in self.model.mujoco_arena.markers:

    #             # Current marker 3D location in world frame
    #             marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

    #             # We use the second tool corner as point on the plane and define the vector connecting
    #             # the marker position to that point
    #             v = marker_pos - corner2_pos

    #             # Shortest distance between the center of the marker and the plane
    #             dist = np.dot(v, n)

    #             # Projection of the center of the marker onto the plane
    #             projected_point = np.array(marker_pos) - dist * n

    #             # Positive distances means the center of the marker is over the plane
    #             # The plane is aligned with the bottom of the wiper and pointing up, so the marker would be over it
    #             if dist > 0.0:
    #                 # Distance smaller than this threshold means we are close to the plane on the upper part
    #                 if dist < 0.02:
    #                     # Write touching points and projected point in coordinates of the plane
    #                     pp_2 = np.array(
    #                         [np.dot(projected_point - corner2_pos, v1), np.dot(projected_point - corner2_pos, v2)]
    #                     )
    #                     # Check if marker is within the tool center:
    #                     if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
    #                         active_markers.append(marker)
    #     return active_markers

    def reward(self, action=None):
        reward = 0
        ee_force = np.linalg.norm(self.robots[0].ee_force["right"])
        
        self.force_history.append(ee_force)
        gripper_contact = self._has_gripper_contact
        any_contact = self._check_contact()
        self.contact_history.append(any_contact)
        print("any_contact: ", any_contact)
        
        start_idx = self.timestep - self.force_history_horizon
        peaks = []
        peak_freqs = []
        # only compute FFT if we have enough force history
        # if start_idx >= 0 and self.timestep % self.extended_action_step_size == 0:
        if start_idx >= 0:
            # return 0
            force_values = np.array(self.force_history[start_idx:])

            # Compute FFT
            N = self.force_history_horizon
            freqs = np.fft.fftfreq(N, d=1)  # Frequency axis
            
            # # ===== General FFT method =====
            # fft_values = np.abs(fft(force_values))  # Magnitude of FFT
            # # Find peaks in the frequency domain
            # peaks, _ = find_peaks(fft_values, height=0.1 * max(fft_values))
            # # ==============================

            # ===== Welch's method =====
            # Welch's method
            from scipy.signal import welch, find_peaks
            dt = 1 / self.control_freq
            fs = 1.0 / dt
            # use nperseg about the window length, maybe N or N//2
            f, Pxx = welch(force_values, fs=fs, window='hann', nperseg=min(N, 256))

            # detect peaks in PSD (Pxx)
            noise_floor = np.median(Pxx) + 3 * np.std(Pxx)  # tune factor
            peaks, props = find_peaks(Pxx, height=noise_floor, prominence=noise_floor*0.5)
            # ==============================
            
            peak_freqs = freqs[peaks]
            self.all_peak_freqs.append(peak_freqs)
            # print("peak_freqs: ", peak_freqs)
            # breakpoint()
        else:
            self.all_peak_freqs.append([])

        # # Plot time-domain signal
        # plt.figure(figsize=(12,5))
        # plt.subplot(2,1,1)
        # plt.plot(force_values)
        # plt.title("Force Data (Time Domain)")
        # plt.xlabel("Sample Index")
        # plt.ylabel("Force")

        # # Plot frequency spectrum
        # plt.subplot(2,1,2)
        # plt.plot(freqs[:N//2], fft_values[:N//2])  # Only positive frequencies
        # plt.scatter(freqs[peaks], fft_values[peaks], color='red', label="Peaks")
        # plt.title("Frequency Spectrum")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Magnitude")
        # plt.legend()
        # plt.show()

        reward = 0.0
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        # Penalty for excessive force with the end-effector
        elif ee_force > 500:
                # reward = -(self.excess_force_penalty_mul * ee_force)
                reward = self.arm_limit_collision_penalty
                self.f_excess += 1
        elif len(peak_freqs) > 0:
            for peak_freq in peak_freqs:
                print("peak_freq: ", peak_freq)
                if abs(peak_freq - self.ideal_peak_freq) < self.peak_freq_tolerance:
                    # breakpoint()
                    reward = 5.0
                    break

        # If the arm is not colliding or in joint limits, we check if we are wiping
        # (we don't want to reward wiping if there are unsafe situations)
        active_markers = []

        # Current 3D location of the corners of the wiping tool in world frame
        for arm in self.robots[0].arms:
            c_geoms = self.robots[0].gripper[arm].important_geoms["corners"]
            active_markers += self._get_active_markers(c_geoms)

        # Obtain the list of currently active (wiped) markers that where not wiped before
        # These are the markers we are wiping at this step
        lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
        new_active_markers = np.array(active_markers)[lall]

        # Loop through all new markers we are wiping at this step
        for new_active_marker in new_active_markers:
            # Grab relevant marker id info
            new_active_marker_geom_id = self.sim.model.geom_name2id(new_active_marker.visual_geoms[0])
            # Make this marker transparent since we wiped it (alpha = 0)
            self.sim.model.geom_rgba[new_active_marker_geom_id][3] = 0
            # Add this marker the wiped list
            self.wiped_markers.append(new_active_marker)

        return reward
    
    def reward_force_and_motion(self, action=None):
        """
        Sensory Reward function for the task.
        """
        reward = 0

        ee_vel = np.linalg.norm(self.robots[0]._hand_total_velocity["right"][:3])
        ee_force = np.linalg.norm(self.robots[0].ee_force["right"])

        ee_vel_th = 0.1
        ee_force_min_th = 2.0

        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        # Penalty for excessive force with the end-effector
        elif ee_force > 300:
                reward = -(self.excess_force_penalty_mul * ee_force)
                self.f_excess += 1
        elif ee_vel > ee_vel_th and ee_force > ee_force_min_th:
                reward = 5.0

        # If the arm is not colliding or in joint limits, we check if we are wiping
        # (we don't want to reward wiping if there are unsafe situations)
        active_markers = []

        # Current 3D location of the corners of the wiping tool in world frame
        for arm in self.robots[0].arms:
            c_geoms = self.robots[0].gripper[arm].important_geoms["corners"]
            active_markers += self._get_active_markers(c_geoms)

        # Obtain the list of currently active (wiped) markers that where not wiped before
        # These are the markers we are wiping at this step
        lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
        new_active_markers = np.array(active_markers)[lall]

        # Loop through all new markers we are wiping at this step
        for new_active_marker in new_active_markers:
            # Grab relevant marker id info
            new_active_marker_geom_id = self.sim.model.geom_name2id(new_active_marker.visual_geoms[0])
            # Make this marker transparent since we wiped it (alpha = 0)
            self.sim.model.geom_rgba[new_active_marker_geom_id][3] = 0
            # Add this marker the wiped list
            self.wiped_markers.append(new_active_marker)
        
        return reward

    
    def old_reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of self.unit_wiped_reward is provided per single dirt (peg) wiped during this step
            - a discrete reward of self.task_complete_reward is provided if all dirt is wiped

        Note that if the arm is either colliding or near its joint limit, a reward of 0 will be automatically given

        Un-normalized summed components if using reward shaping (individual components can be set to 0:

            - Reaching: in [0, self.distance_multiplier], proportional to distance between wiper and centroid of dirt
              and zero if the table has been fully wiped clean of all the dirt
            - Table Contact: in {0, self.wipe_contact_reward}, non-zero if wiper is in contact with table
            - Wiping: in {0, self.unit_wiped_reward}, non-zero for each dirt (peg) wiped during this step
            - Cleaned: in {0, self.task_complete_reward}, non-zero if no dirt remains on the table
            - Collision / Joint Limit Penalty: in {self.arm_limit_collision_penalty, 0}, nonzero if robot arm
              is colliding with an object
              - Note that if this value is nonzero, no other reward components can be added
            - Large Force Penalty: in [-inf, 0], scaled by wiper force and directly proportional to
              self.excess_force_penalty_mul if the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0], scaled by estimated wiper acceleration and directly
              proportional to self.ee_accel_penalty

        Note that the final per-step reward is normalized given the theoretical best episode return and then scaled:
        reward_scale * (horizon /
        (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        total_force_ee = max(
            [
                np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques[arm].current[:3]))
                for arm in self.robots[0].arms
            ]
        )

        # Neg Reward from collisions of the arm with the table
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        else:
            # If the arm is not colliding or in joint limits, we check if we are wiping
            # (we don't want to reward wiping if there are unsafe situations)
            active_markers = []

            # Current 3D location of the corners of the wiping tool in world frame
            for arm in self.robots[0].arms:
                c_geoms = self.robots[0].gripper[arm].important_geoms["corners"]
                active_markers += self._get_active_markers(c_geoms)

            # Obtain the list of currently active (wiped) markers that where not wiped before
            # These are the markers we are wiping at this step
            lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
            new_active_markers = np.array(active_markers)[lall]

            # Loop through all new markers we are wiping at this step
            for new_active_marker in new_active_markers:
                # Grab relevant marker id info
                new_active_marker_geom_id = self.sim.model.geom_name2id(new_active_marker.visual_geoms[0])
                # Make this marker transparent since we wiped it (alpha = 0)
                self.sim.model.geom_rgba[new_active_marker_geom_id][3] = 0
                # Add this marker the wiped list
                self.wiped_markers.append(new_active_marker)
                # Add reward if we're using the dense reward
                if self.reward_shaping:
                    reward += self.unit_wiped_reward

            # Additional reward components if using dense rewards
            if self.reward_shaping:
                # If we haven't wiped all the markers yet, add a smooth reward for getting closer
                # to the centroid of the dirt to wipe
                if len(self.wiped_markers) < self.num_markers:
                    _, _, mean_pos_to_things_to_wipe = self._get_wipe_information()
                    mean_distance_to_things_to_wipe = np.linalg.norm(mean_pos_to_things_to_wipe)
                    reward += self.distance_multiplier * (
                        1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe)
                    )

                # Reward for keeping contact
                if self.sim.data.ncon != 0 and self._has_gripper_contact:
                    reward += self.wipe_contact_reward

                # Penalty for excessive force with the end-effector
                if total_force_ee > self.pressure_threshold_max:
                    reward -= self.excess_force_penalty_mul * total_force_ee
                    self.f_excess += 1

                # Reward for pressing into table
                # TODO: Need to include this computation somehow in the scaled reward computation
                elif total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1:
                    reward += self.wipe_contact_reward + 0.01 * total_force_ee
                    if self.sim.data.ncon > 50:
                        reward += 10.0 * self.wipe_contact_reward

                # Penalize large accelerations
                reward -= self.ee_accel_penalty * max(
                    [np.mean(abs(self.robots[0].recent_ee_acc[arm].current)) for arm in self.robots[0].arms]
                )

            # Final reward if all wiped
            if len(self.wiped_markers) == self.num_markers:
                reward += self.task_complete_reward

        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}"
                "wiped markers: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    ws=len(self.wiped_markers),
                    sc=self.collisions,
                    fe=self.f_excess,
                )
            )
            print(string_to_print)

        unnormalized_reward = reward
        # If we're scaling our reward, we normalize the per-step rewards given the theoretical best episode return
        # This is equivalent to scaling the reward by:
        #   reward_scale * (horizon /
        #       (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))
        if self.reward_scale:
            reward *= self.reward_scale * self.reward_normalization_factor
        # if len(new_active_markers) > 0:
        #         breakpoint()
        # if reward > 0:
        #     breakpoint()
        return reward
    
    def get_amount_wiped(self):
        return len(self.wiped_markers) / self.num_markers

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        mujoco_arena = WipeArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        sensors = []
        names = []

        # Add binary contact observation
        if self.use_contact_obs:

            @sensor(modality=f"{pf}proprio")
            def gripper_contact(obs_cache):
                return self._has_gripper_contact

            sensors.append(gripper_contact)
            names.append(f"{pf}contact")

        # object information in the observation
        if self.use_object_obs:

            if self.use_condensed_obj_obs:
                # use implicit representation of wiping objects
                @sensor(modality=modality)
                def wipe_radius(obs_cache):
                    wipe_rad, wipe_cent, _ = self._get_wipe_information()
                    obs_cache["wipe_centroid"] = wipe_cent
                    return wipe_rad

                @sensor(modality=modality)
                def wipe_centroid(obs_cache):
                    return obs_cache["wipe_centroid"] if "wipe_centroid" in obs_cache else np.zeros(3)

                @sensor(modality=modality)
                def proportion_wiped(obs_cache):
                    return len(self.wiped_markers) / self.num_markers

                sensors += [proportion_wiped, wipe_radius, wipe_centroid]
                names += ["proportion_wiped", "wipe_radius", "wipe_centroid"]

                if self.use_robot_obs:
                    arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
                    full_prefixes = self._get_arm_prefixes(self.robots[0])

                    # also use ego-centric obs
                    robot_obs_sensors = [
                        self._get_obj_eef_sensor(
                            full_pf, "wipe_centroid", f"{arm_pf}gripper_to_wipe_centroid", modality
                        )
                        for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                    ]

                    sensors.extend(robot_obs_sensors)
                    names.extend([fn.__name__ for fn in robot_obs_sensors])

            else:
                # use explicit representation of wiping objects
                for i, marker in enumerate(self.model.mujoco_arena.markers):
                    marker_sensors, marker_sensor_names = self._create_marker_sensors(i, marker, modality)
                    sensors += marker_sensors
                    names += marker_sensor_names

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_marker_sensors(self, i, marker, modality="object"):
        """
        Helper function to create sensors for a given marker. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            i (int): ID number corresponding to the marker
            marker (MujocoObject): Marker to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given marker
                names (list): array of corresponding observable names
        """

        @sensor(modality=modality)
        def marker_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

        @sensor(modality=modality)
        def marker_wiped(obs_cache):
            return [0, 1][marker in self.wiped_markers]

        sensors = [marker_pos, marker_wiped]
        names = [f"marker{i}_pos", f"marker{i}_wiped"]

        if self.use_robot_obs:
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            grippers_to_marker_fns = [
                self._get_obj_eef_sensor(full_pf, f"marker{i}_pos", f"{arm_pf}gripper_to_marker{i}", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            sensors.extend(grippers_to_marker_fns)
            names.extend([fn.__name__ for fn in grippers_to_marker_fns])

        return sensors, names

    def _reset_internal(self):
        super()._reset_internal()

        # inherited class should reset positions of objects (only if we're not using a deterministic reset)
        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena(self.sim)

        # Reset all internal vars for this wipe task
        self.timestep = 0
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0

        # ee resets - bias at initial state
        self.ee_force_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}
        self.ee_torque_bias = {arm: np.zeros(3) for arm in self.robots[0].arms}

    def _check_success(self):
        """
        Checks if Task succeeds (all dirt wiped).

        Returns:
            bool: True if completed task
        """
        return True if len(self.wiped_markers) == self.num_markers else False

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion (wiping succeeded)
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            if self.print_results:
                print(40 * "-" + " COLLIDED " + 40 * "-")
            terminated = True

        # Prematurely terminate if task is success
        if self._check_success():
            if self.print_results:
                print(40 * "+" + " FINISHED WIPING " + 40 * "+")
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self.robots[0].check_q_limits():
            if self.print_results:
                print(40 * "-" + " JOINT LIMIT " + 40 * "-")
            terminated = True

        return terminated

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        amount_wiped = self.get_amount_wiped()
        info["amount_wiped"] = amount_wiped

        # Update force bias
        if all([np.linalg.norm(self.ee_force_bias[arm]) == 0 for arm in self.ee_force_bias]):
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        if self.get_info:
            info["add_vals"] = ["nwipedmarkers", "colls", "percent_viapoints_", "f_excess"]
            info["nwipedmarkers"] = len(self.wiped_markers)
            info["colls"] = self.collisions
            info["percent_viapoints_"] = len(self.wiped_markers) / self.num_markers
            info["f_excess"] = self.f_excess

        # allow episode to finish early if allowed
        if self.early_terminations:
            done = done or self._check_terminated()

        return reward, done, info

    def _get_wipe_information(self):
        """Returns set of wiping information"""
        mean_pos_to_things_to_wipe = np.zeros(3)
        wipe_centroid = np.zeros(3)
        marker_positions = []
        num_non_wiped_markers = 0
        if len(self.wiped_markers) < self.num_markers:
            for marker in self.model.mujoco_arena.markers:
                if marker not in self.wiped_markers:
                    marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])
                    wipe_centroid += marker_pos
                    marker_positions.append(marker_pos)
                    num_non_wiped_markers += 1
            wipe_centroid /= max(1, num_non_wiped_markers)

            # Mean position to things to wipe to the closest arm
            mean_pos_to_things_to_wipe_list = [wipe_centroid - self._get_eef_xpos(arm) for arm in self.robots[0].arms]
            mean_pos_to_things_to_wipe = mean_pos_to_things_to_wipe_list[
                np.argmin([np.linalg.norm(x) for x in mean_pos_to_things_to_wipe_list])
            ]
        # Radius of circle from centroid capturing all remaining wiping markers
        max_radius = 0
        if num_non_wiped_markers > 0:
            max_radius = np.max(np.linalg.norm(np.array(marker_positions) - wipe_centroid, axis=1))
        # Return all values
        return max_radius, wipe_centroid, mean_pos_to_things_to_wipe

    def _get_eef_xpos(self, arm):
        """
        Grabs End Effector position as specifed by the arm argument

        Args:
            arm (str): Arm name

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the any of the grippers are making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        arm = self.robots[0].arms[0]
        # print(f"ee_force: {np.linalg.norm(self.robots[0].ee_force[arm] - self.ee_force_bias[arm])}")
        return any(
            [   
                np.linalg.norm(self.robots[0].ee_force[arm] - self.ee_force_bias[arm]) > self.contact_threshold
                for arm in self.robots[0].arms
            ]
        )
    
    def reset(self):
        """
        Extends the superclass reset method to reset the force history list.
        """
        self.force_history = []
        return super().reset()

    def _check_contact(self):
        """Check if the robot's end effector is in contact with any surface"""
        # print(f"ncon: {self.sim.data.ncon}")
        return self.sim.data.ncon > 0

    def _get_current_ee_position(self):
        """Get the current end effector position"""
        return self.robots[0]._hand_pose["right"][:3, 3]  # Just position, not orientation

    def move_down_until_contact(self):
        """Move the end effector down until contact is detected"""
        steps_taken = 0
        while not self._check_contact() and steps_taken < self.max_steps_without_contact:
            # print("steps_taken, step, contact: ", steps_taken, self.timestep, self._check_contact())
            current_pos = self._get_current_ee_position()
            target_pos = current_pos.copy()
            target_pos[2] -= self.vertical_step_size  # Move down in z-axis
            
            # Create action to move down
            action = np.zeros(6)  # 6-DOF pose control + gripper
            action[:3] = target_pos - current_pos
            
            # Take the action
            # print(f"action: {action}")
            obs, reward, done, info = self.step(action)
            steps_taken += 1
        
        return self._check_contact()

    def move_laterally_with_contact(self, direction='right', distance=0.5):
        """
        Move the end effector laterally while maintaining contact
        direction: 'right' or 'left'
        distance: distance to move in meters
        """
        # if not self._check_contact():
        #     return False

        # steps_needed = int(distance / self.lateral_step_size)
        steps_needed = 80
        direction_multiplier = 1 if direction == 'right' else -1
        # print(f"steps_needed: {steps_needed}")
        is_contact = self._check_contact()
        print(f"is_contact: {is_contact}")
        
        init_pos = self._get_current_ee_position()
        for curr_step in range(steps_needed):
            current_pos = self._get_current_ee_position()
            target_pos = current_pos.copy()
            target_pos[1] += direction_multiplier * self.lateral_step_size  # Move in y-axis
            
            # # If we've lost contact, try to regain it by moving down slightly
            # if not self._check_contact():
            #     target_pos[2] -= self.vertical_step_size
            
            # Create action
            action = np.zeros(6)  # 6-DOF pose control + gripper
            action[:3] = target_pos - current_pos
            print(f"action: {np.linalg.norm(action)}")

            if curr_step % self.extended_action_step_size == 0:
                # Add some noise to the x and y axis movement
                xy_noise_scale = 0.0  # Standard deviation for x,y-axis noise in meters
                z_noise_scale = 0.0  # Standard deviation for z-axis noise in meters
                
                noise = np.zeros(6)
                x_noise = np.random.randn() * xy_noise_scale
                y_noise = np.random.randn() * xy_noise_scale
                # z noise should only be negative (or zero).
                # We take a standard normal random number, scale it, and then ensure it's non-positive.
                z_noise = min(0, np.random.randn() * z_noise_scale)
                
                # z_noise = -0.01
                # np.random.randn() * z_noise_scale

                noise[:3] = np.array([x_noise, y_noise, z_noise])

                # # Add some noise to the rotation axis
                # rotation_noise_scale = 0.1  # Standard deviation for rotational noise in radians
                # action[3:] += np.random.randn(3) * rotation_noise_scale
            
            print(f"noise: {noise}")
            action += noise

            # Take the action
            self.step(action)

            new_pos = self._get_current_ee_position()
            # print("action[:3], delta move: ", action[:3], np.linalg.norm(new_pos - current_pos))

            gripper_contact = self._has_gripper_contact
            # print(f"gripper_contact: {gripper_contact}")
            
            # print(f"is_contact: {self._check_contact()}")
            # If we still don't have contact, the movement failed
            # if not self._check_contact():
            #     return False
        final_pos = self._get_current_ee_position()
        print("total_delta_move: ", np.linalg.norm(final_pos - init_pos))

        return True

    def perform_wiping_skill(self, num_cycles=5):
        """
        Perform the complete wiping skill:
        1. Move down until contact
        2. Move right and left while maintaining contact
        3. Move up after completion
        """
        # First move down until contact
        if not self.move_down_until_contact():
            print("Failed to establish initial contact")
            return False

        # print("Starting lateral movement at step: ", self.timestep)
        # breakpoint()
        
        # Perform the specified number of cycles
        for cycle in range(num_cycles):
            # Move right
            if not self.move_laterally_with_contact('right', 40.0):
                print(f"Failed during rightward movement in cycle {cycle + 1}")
                return False
                        
            # # Move left
            # if not self.move_laterally_with_contact('left', 40.0):
            #     print(f"Failed during leftward movement in cycle {cycle + 1}")
            #     return False

        # breakpoint()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Plot 1: Force History on ax1
        ax1.plot(self.force_history)
        ax1.set_title("Force History")
        ax1.set_ylabel("End-effector Force (N)")
        ax1.grid(True)
        ax1.set_ylim(0, 10.0)

        # Plot 2: Peak Frequencies on ax2
        for t, freqs_at_t in enumerate(self.all_peak_freqs):
            if len(freqs_at_t) > 0:
                ax2.scatter([t] * len(freqs_at_t), freqs_at_t, c="b", marker="o")

        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Peak frequency (Hz)")
        ax2.set_title("Peak frequencies over time")

        # Plot 3: Contact History on ax3
        ax3.plot(self.contact_history)
        ax3.set_title("Contact History")
        ax3.set_ylabel("Contact (1) or No Contact (0)")
        ax3.grid(True)
        ax3.set_ylim(-0.1, 1.1)

        # Adjust layout and show the combined plot
        plt.tight_layout()
        plt.show()

        
        # Move up after completion
        current_pos = self._get_current_ee_position()
        target_pos = current_pos.copy()
        target_pos[2] += 0.1  # Move up 10cm
        
        action = np.zeros(6)
        action[:3] = target_pos - current_pos
        self.step(action)
        
        return True

