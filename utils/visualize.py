import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
import pandas as pd
from inference import config

class TactileVisualize():
    def __init__(self, skin_path,
                       init_positions, 
                       depth_range=[-5, 16]):
        self.skin_path = skin_path
        self.init_positions = init_positions
        self.deviations = np.zeros_like(self.init_positions) # running deviations
        self.depth_range =  depth_range
        self.plot_initialize()


    def plot_initialize(self):
        # set camera pososition
        self.cpos = np.array([[-57.34826013561122, -661.242740117575, 352.06298028880514],
                              [-0.056080635365844955, -1.209135079294247, 118.18100834810429],
                              [0.024125960714330284, 0.3320410642889663, 0.9429563455672065]])
        
        self.plotter = pvqt.BackgroundPlotter()
        self.plotter.set_background("white", top="white")
        pv.global_theme.font.color = 'black' 
        pv.global_theme.font.title_size = 16 
        pv.global_theme.font.label_size = 16  
        boring_cmap = plt.cm.get_cmap("bwr")  
        
        self.plotter.subplot(0, 0)
        self.plotter.camera_position = self.cpos
        self.plotter.show_axes()
        self.skin_est = pv.read(self.skin_path) # for pyvista visualization
        norm_deviations = np.linalg.norm(self.deviations, axis=1)
        self.skin_est['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.plotter.add_mesh(self.skin_est, cmap=boring_cmap, clim=self.depth_range)

    def updade_plot(self, deformation, positions):
        self.skin_est['contact depth (unit:mm)'] = deformation # for contact depth visualization
        self.skin_est.points = positions