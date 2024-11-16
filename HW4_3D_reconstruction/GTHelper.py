import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot

class GTHelper():
    def __init__(self) -> None:
        self.PRIOR_NOISE = None
        self.ODOM_NOISE = None
        self.graph = None
        self.initial_estimates = None

    def create_non_linear_graph(self):
        self.graph = gtsam.NonlinearFactorGraph()

    def set_odom_noise(self, vars:np.ndarray):
        self.ODOM_NOISE = gtsam.noiseModel.Diagonal.Sigmas(vars)

    def set_prior_noise(self, vars:np.ndarray):
        self.PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(vars)

    def add_prior_factor(self, pose:tuple):
        self.graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(*pose), self.PRIOR_NOISE))
            
    def add_initial_estimates(self, poses:np.ndarray):
        self.initial_estimates = gtsam.Values()

        for idx, pose in enumerate(poses):
            self.initial_estimates.insert(idx+1, gtsam.Pose2(*pose))