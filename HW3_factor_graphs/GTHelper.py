import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot
from Pose2D import Pose2D

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
    
    def add_factors(self, match_dict:dict):
        for src_idx, des_idx in match_dict.keys():
            e = match_dict[(src_idx, des_idx)]["error"]
            H = match_dict[(src_idx, des_idx)]["H"]
            factor_pose = Pose2D.get_pose_from_homography(H)
            
            noise = gtsam.noiseModel.Diagonal.Sigmas([e, e, e])
            self.graph.add(gtsam.BetweenFactorPose2(src_idx + 1, 
                                                    des_idx + 1, 
                                                    gtsam.Pose2(*factor_pose), 
                                                    noise))
            
    def add_initial_estimates(self, poses:np.ndarray):
        self.initial_estimates = gtsam.Values()

        for idx, pose in enumerate(poses):
            self.initial_estimates.insert(idx+1, gtsam.Pose2(*pose))

    # def add_factors(self, match)