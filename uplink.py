# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:39:23 2025

@author: tmal0697
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math
import hcipy as hc
import cmasher
import scipy.stats
import warnings
from datetime import date
import os
import imageio
from scipy import optimize


from lanternfiber import lanternfiber

time = date.today()
save_dir = 'simulations/' + str(time.month) + '-' + str(time.day) +'/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class uplink:
    def __init__(self, wavelength = 1.55e-6, grid_size = 128):
        self.wav = wavelength
        self.grid_size = grid_size
        self.refPSF = None
        
       
        
    def makeSystem(self, n_core = 1.444, n_cladding = 1.4385, core_radius = 7e-6, max_r = 2, n_modes = 3, matrix = 'unitary',
                   fiber_length = 10, transmitting_telescope_d = 1, recieving_telescope_d = 1):
        
        # Creates the uplink system.
        #
        #
        # Parameters
        # -------------------------
        # n_core 
        #       Refractive index of MMF core
        # n_cladding
        #       Refractive index of MMF cladding
        # core_radius
        #       Core radius of MMF fiber
        # max_r
        #       Maximum radius to calculate mode field, where max_r = 1 is the core radius
        # n_modes
        #       Number of modes supported by the MM core and the lantern
        # matrix
        #       transfer matrix of the lantern (from MM to SM end)
        # fiber_length
        #       length of MM fiber
        # transmitting_telescope_d
        #       diameter of the transmitting telescope in meters
        # recieving_telescope_d
        #       diameter of the recieving telescope in meters
        
        self.fiberCore = n_core
        self.fiberCladding = n_cladding
        self.fiberCoreRadius = core_radius
        self.max_r = max_r
        self.nModes = n_modes
        self.fiberLength = fiber_length
        self.trnsTelDia = transmitting_telescope_d
        self.rcvTelDia = recieving_telescope_d
        
        npix = int(self.grid_size/2)
        
        # Creating lantern using lanternfiber.py
        f = lanternfiber(self.fiberCore, self.fiberCladding, self.fiberCoreRadius,
                         self.wav, nmodes = self.nModes)
        f.find_fiber_modes(verbose = False)
        with warnings.catch_warnings(action = 'ignore'):
            f.make_fiber_modes(npix = npix, show_plots = False, max_r = self.max_r)
        modes_to_measure = np.arange(f.nmodes)
        self.LP_mf = f.allmodefields_rsoftorder # all mode fields found in the MM fiber end of the lantern
        
        # Transfer matrix
        if matrix == 'unitary':
            self.M = np.matrix(scipy.stats.unitary_group(self.nModes).rvs())
            self.M_t = self.M.getH()
        else:
            self.M = matrix
            self.M_t = matrix.getH()
        
        self.lantern = f
        
        # MM fiber
        self.fiberNA = np.sqrt(math.pow(self.fiberCore, 2) - math.pow(self.fiberCladding, 2))
        self.fiber = hc.StepIndexFiber(self.fiberCoreRadius, self.fiberNA, self.fiberLength)

        # Making grids for propagation
        D_focus = 4.1 * self.fiberCoreRadius
        self.MMfocalGrid = hc.make_pupil_grid(self.grid_size, D_focus)
        self.trnsPupilGrid = hc.make_pupil_grid(self.grid_size, self.trnsTelDia)
        self.rcvPupilGrid = hc.make_pupil_grid(self.grid_size, self.rcvTelDia)
        self.detectGrid = hc.make_pupil_grid(self.grid_size, 4.1*7e-6)
        
        # Propagators using hcipy
        trnsFocalLength = self.trnsTelDia/(2 * self.fiberNA)
        rcvFocalLength = self.rcvTelDia/(2 * self.fiberNA)
        self.propagatorTrns = hc.FraunhoferPropagator(self.trnsPupilGrid, self.MMfocalGrid, focal_length = trnsFocalLength)
        self.propagatorRcv = hc.FraunhoferPropagator(self.rcvPupilGrid, self.detectGrid, focal_length = trnsFocalLength)



    def makeAtmosphereLayer(self, seeing = 1.0, outer_scale = 40, tau0 = 0.005, show_plots = False):
        
        # Creates an atmospheric layer.
        #
        #
        # Parameters
        # -----------------
        # seeing
        #       parameter of turbulence in arcsec
        # outer_scale
        #       parameter of outer scale
        # tau0
        #       parameter of tau0
        # show_plots
        #       self-explanatory
        
        
        fried_parameter = hc.seeing_to_fried_parameter(seeing)
        Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter, self.wav)
        velocity = 0.314 * fried_parameter / tau0
        
        # layer creation using physical parameters
        with warnings.catch_warnings(action = 'ignore'):
            self.layer = hc.InfiniteAtmosphericLayer(self.rcvPupilGrid, Cn_squared, outer_scale, velocity)
        
        if show_plots == True:
            phase_screen_phase = self.layer.phase_for(self.wav) # in radian
            phase_screen_opd = phase_screen_phase * (self.wav / (2 * np.pi)) * 1e6

            hc.imshow_field(phase_screen_opd, vmin = -3, vmax = 3, 
                            cmap = 'Blues_r', interpolation = None)
            plt.title("Atmosphere Phase Screen")
            plt.xlabel("x position (m)")
            plt.ylabel("y position (m)")
            plt.colorbar()
            plt.show()
            
        return self.layer
       
        
        
    def setSMFParameters(self, parameter_array):
        
        # Setting the parameters of the SMF end of the photonic lantern
        #
        #
        # Parameters
        # ---------------------
        # parameter_array
        #       An array of the ampltitude and phase parameters of the SMF fibers
        
        
        self.parameters = parameter_array
        set_real = np.zeros(self.nModes)
        set_imag = np.zeros(self.nModes)
        
        # Intensities are the first set of parameters and phases are second set of parameters
        set_intensity = self.parameters[0:self.nModes]
        set_phase = self.parameters[self.nModes:self.nModes*2]
        
        for i in range(self.nModes):
            set_real[i] = set_intensity[i] * math.cos(set_phase[i])
            set_imag[i] = set_intensity[i] * math.sin(set_phase[i])
        
        # Complex representations of SMF parameters
        self.smf_complex = set_real + set_imag*1j
        


    def prop(self, show_plots = 'False'):
        
        # Propagation of light from the SMF end of the photonic lantern all the way to the receiver telescope
        #
        #
        # Parameters
        # -------------------------
        # show_plots
        #       self-explanatory
        
        # Complex coefficients of the LP modes in the MM fiber
        LP_complex_coefs = np.matmul(self.M_t, self.smf_complex).A1
        
        # Linear superposition of the LP modes
        combined_field = self.LP_mf[0]*LP_complex_coefs[0]
        for i in range(1, self.nModes):
            combined_field += self.LP_mf[i]*LP_complex_coefs[i]
        
        # Make Wavefront object for lantern output
        grid_diameter = self.fiberCoreRadius * 4
        griddy = hc.field.make_uniform_grid([self.grid_size, self.grid_size], [grid_diameter, grid_diameter])
        combinedField = hc.field.Field(combined_field.flatten(), griddy)
        lanternOutput = hc.optics.Wavefront(combinedField, wavelength = self.wav)
        
        # Propagation through system
        wf_foc_u = self.fiber.backward(lanternOutput)
        wf_u = self.propagatorTrns.backward(wf_foc_u)
        wf_atm = self.layer(wf_u)
        psf = self.propagatorRcv.forward(wf_atm)
        
        if show_plots:
            hc.imshow_field(wf_foc_u.power)
            circ = plt.Circle((0,0), self.fiberCoreRadius, 
                              edgecolor = 'white', 
                              fill = False,
                              linewidth = 2,
                              alpha = 0.5)
            plt.gca().add_artist(circ)
            plt.xlabel('x (um)')
            plt.ylabel('y (um)')
            plt.title('Photonic Lantern Output')
            plt.colorbar()
            plt.show()
            
            hc.imshow_field(psf.power)
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title('Best Scipy Output')
            plt.colorbar()
            plt.show()
                        
        return psf, wf_foc_u
            
    
        
    def makeRef(self, mode = 'LP01', show_plots = False):

        # Make reference psf of ideal propagations through a non-turbulent atmosphere
        #
        #
        # Parameters
        # ---------------------------
        # mode
        #       reference mode
        # show_plots
        #   self-explanatory
        
        # Set SMF Complex coefficients
        if mode == 'LP01':
            arr = [1 + 0j]
            for i in range(self.nModes - 1):
                arr.append(0 + 0j)
            smf_complex = np.matmul(self.M, arr)
            self.smf_complex = smf_complex.A1
            
        # Make non-turbulent atmosphere
        self.makeAtmosphereLayer(seeing = 0.0, show_plots = show_plots)
        
        # Propagation
        refPSF, wf_foc_u = self.prop(show_plots = show_plots)
        
        self.refPSF = refPSF
        
        if show_plots:
            hc.imshow_field(self.refPSF.power)
            plt.xlabel('Position (m)')
            plt.ylabel('Position (m)')
            plt.title('Ideal PSF')
            plt.colorbar()
            plt.show()
            

            
        return refPSF
        
    
    
    def strehl_ratio(self, psf, ref_psf):
        
        # Strehl ratio calculation as the ratio of the max between the two psfs
        #
        #
        # Parameters
        # --------------------------
        # psf
        #       measured/simulated psf of propagation through turbulent atmosphere
        # ref_psf
        #       reference psf of propagation through non-turbulent atmosphere
        
        return psf[np.argmax(ref_psf)] / ref_psf.max()
    
    
    
    def strehl_ratio_integral(self, psf, ref_psf):
        
        # Strehl ratio calculations as the overlap integral between two psfs
        #
        #
        # Parameters
        # --------------------------
        # psf
        #       measured/simulated psf of propagation through turbulent atmosphere
        # ref_psf
        #       reference psf of propagation through non-turbulent atmosphere
        
        overlap_int = np.sum(psf * ref_psf) / np.sqrt( np.sum(np.abs(ref_psf) ** 2) * np.sum(np.abs(psf) ** 2))
        return overlap_int
    


    def gradDescentWalkingVis(self, phases1, phases2, Js, phases1_gd, phases2_gd):
        
        # Visualization of gradient descent "walking" through a grid search ground truth
        #
        #
        # Parameters
        # --------------------------
        # phases1, phases2
        #       Grid Search phases
        # Js
        #       Grid Search Image Quality Metric Results
        # phases1_gd, phases2_gd
        #       Gradient Descent phase paths
        
        for i in range(len(phases1_gd)):
            fig, ax = plt.subplots()
            plt.scatter(phases1, phases2, c = Js)
            plt.title('Grid Search vs Gradient Descent Iteration %i' %(i+1))
            plt.ylabel('Channel 2 Phase (radians)')
            plt.xlabel('Channel 1 Phase (radians)')
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('J (Image Quality)')
            
            plt.plot(phases1_gd[0:i], phases2_gd[0:i], color = 'r')
            
            

    def GridSearch_2channel(self, num_iter = 20, show_plots = False, verbose = False):
        
        # Grid Search changing 2 phases, while keeping everything constant
        # 
        #
        # Parameters
        # --------------------------
        # num_iter
        #       The discreteness of the grid search
        # show_plots, verbose
        #       Self-explanatory
        
        set_intensity = [0.5, 0.5, 0.5]
        phases = np.linspace(0, 2*np.pi, num = num_iter)
        
        # Initializations
        parameter_array = np.concatenate((set_intensity, [0,0,0]))
        parameter_max_array = np.copy(parameter_array)
        
        phases1 = []
        phases2 = []
        Js = []
        
        maxJ = 0
        
        # Grid Search scanning in 2D space
        for j in range(num_iter):
            for k in range(num_iter):
                
                # parameter update
                parameter_array[3] = phases[j]
                parameter_array[4] = phases[k]
                
                # test propagation
                self.setSMFParameters(parameter_array)
                psf, wf_foc_u = self.prop(show_plots = False)
                
                # evaluating image quality
                with warnings.catch_warnings(action = 'ignore'):
                    J_new = self.strehl_ratio_integral(psf.power, self.refPSF.power)
                
                phases1.append(parameter_array[3])
                phases2.append(parameter_array[4])
                Js.append(J_new)
                
                if J_new > maxJ:
                    maxJ = J_new
                    parameter_max_array = np.copy(parameter_array)

                
        if verbose:
            print("----- Grid Search (2 Channel) ------")
            print('Max J: ', maxJ)
            print('Intensities: ', parameter_max_array[0:3])
            print('Phases: ', parameter_max_array[3:6])
               
        if show_plots:
            plt.scatter(phases1, phases2, c = Js)
            plt.title('Grid Search')
            plt.colorbar()
            plt.show()
            
            self.setSMFParameters(parameter_max_array)
            psf, wf_foc_u = self.prop(show_plots = True)
            
            hc.imshow_field(psf.power)
            plt.title('Best Result 2 Channel')
            plt.colorbar()
            plt.show()
            
        return phases1, phases2, Js


    
    def GridSearch_6channel(self, num_iter = 3, show_plots = False, verbose = False):

        # Grid Search changing all 6 parameters
        # 
        #
        # Parameters
        # --------------------------
        # num_iter
        #       The discreteness of the grid search
        # show_plots, verbose
        #       Self-explanatory
        
        # Initializations
        phases = np.linspace(0, 2*np.pi, num = num_iter)
        intensities = np.linspace(0, 1, num = num_iter)
        
        parameter_array = np.zeros(6)
        parameter_max_array = np.zeros(6)
        
        maxJ = 0
        
        # Grid Search scanning in 6-D space
        for j in range(num_iter):
            for k in range(num_iter):
                for m in range(num_iter):
                    for jj in range(num_iter):
                        for kk in range(num_iter):
                
                            # parameter update
                            parameter_array[0] = intensities[j]
                            parameter_array[1] = intensities[k]
                            parameter_array[2] = intensities[m]
                            parameter_array[3] = phases[jj]
                            parameter_array[4] = phases[kk]
                            
                            # test propagation
                            self.setSMFParameters(parameter_array)
                            psf, wf_foc_u = self.prop(show_plots = False)
                            
                            # evaluating image quality
                            with warnings.catch_warnings(action = 'ignore'):
                                J_new = self.strehl_ratio_integral(psf.power, self.refPSF.power)
                            
                            if J_new > maxJ:
                                maxJ = J_new
                                parameter_max_array = np.copy(parameter_array)
                    
               
        if verbose:
            print("----- Grid Search (6 Channel) ------")
            print('Max J: ', maxJ)
            print('Intensities: ', parameter_max_array[0:3])
            print('Phases: ', parameter_max_array[3:6])
               
        if show_plots:
 
            self.setSMFParameters(parameter_max_array)
            psf, wf_foc_u = self.prop(show_plots = True)
            
            hc.imshow_field(psf.power)
            plt.title('Best Result 6 Channel')
            plt.colorbar()
            plt.show()
            
        return maxJ
        


    def GradientDescent_2channel(self, learning_rate = 1, num_iter = 100, show_plots = False, verbose = False, trouble = False, save_fig = False):
        
        # Gradient Descent changing 2 phases, while keeping everything constant
        #
        #
        # Parameters
        # --------------------
        # learning_rate
        #       the weight of each change in parameter
        # num_iter
        #       number of iterations to converge
        # show_plots, verbose, save_fig
        #       self-explanatory
        # trouble
        #       trouble-shooting
        
        # Initializations
        parameter_array = [0.5, 0.5, 0.5, np.pi, np.pi, np.pi]
        
        del_phase = np.zeros(3)
        del_J = np.zeros(3)
        
        phases1 = []
        phases2 = []
        Js = []
        
        J = 0
        
        # Iterations
        for j in range(num_iter):
            
            # Test Propagation and Evaluating Image Quality
            self.setSMFParameters(parameter_array)
            psf, wf_foc_u = self.prop(show_plots = False)
            J_new = self.strehl_ratio_integral(psf.power, self.refPSF.power)
            
            
            if save_fig:
                fig, ax = plt.subplots()
                hc.imshow_field(psf.power)
                plt.title('Receiver (Iteration %i)' % j)
                plt.xlabel('x (um)')
                plt.ylabel('y (um)')
                plt.colorbar()
                path = save_dir + 'psf/'
                
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + 'iter' + str(j) + '.png')
                plt.close(fig)
                
                fig1, ax1 = plt.subplots()
                hc.imshow_field(wf_foc_u.power)
                circ = plt.Circle((0,0), self.fiberCoreRadius, 
                                  edgecolor = 'white', 
                                  fill = False,
                                  linewidth = 2,
                                  alpha = 0.5)
                plt.gca().add_artist(circ)
                plt.xlabel('x (um)')
                plt.ylabel('y (um)')
                plt.title('Lantern Output (Iteration %i)' % j)
                plt.colorbar()
                
                path = save_dir + 'lanternOut/'
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + 'iter' + str(j) + '.png')
                plt.close(fig1)

            
            if trouble:
                print("--------------- iteration ", j)
                print("Phase: ", parameter_array[3:6])
                
            # Finite Differences calculation
            if j == 0:
                
                # random assignment of parameter to change
                k = math.floor(np.random.rand() * 2)
                del_phase[k] = 2 * np.random.rand() - 1
                
            else:
                
                # Change in Image Quality, deltaJ
                del_J[k] = J_new - J
                
                # random assignment of parameter to change
                k = math.floor(np.random.rand() * 2)
                
                phases1.append(parameter_array[3])
                phases2.append(parameter_array[4])
                
                # Change-in-phase calculation
                if del_J[k] == 0:     
                    del_phase[k] = 2*np.random.rand() - 1
                else:  
                    del_phase[k] = math.fmod(learning_rate*del_J[k]/del_phase[k], 2*np.pi)
                    
            J = J_new
            Js.append(J_new)
            
            if j == num_iter - 1:
                if verbose:
                    print("-------Gradient Descent (2 Channel)---------")
                    print('Max J: ', J)
                    print('Intensities: ', parameter_array[0:3])
                    print('Phases: ', parameter_array[3:6])
                    
            
            if del_phase[k] == 0:
                del_phase[k] += 1e-5
                
            p = k + 3
            parameter_array[p] += del_phase[k]
            
            # Setting parameters
            if parameter_array[p] < 0:
                parameter_array[p] = 2*np.pi + parameter_array[p]
            elif parameter_array[p] > 2*np.pi:
                parameter_array[p] = parameter_array[p] - 2*np.pi  
                
            if trouble:
                print("image quality: ", J_new)
                print("Changing Paramater ", k)
                print("Change in Phase: ", del_phase)
                
        if show_plots:
            hc.imshow_field(psf.power)
            plt.title('Gradient Descent (2 Channel)')
            plt.colorbar()
            plt.show()
        
        return phases1, phases2, Js
    
    
    
    def GradientDescent_3channel(self, learning_rate = 1, num_iter = 100, std_level = 0.0, show_plots = False, verbose = False, trouble = False, save_fig = False):
        
        # Gradient Descent changing all 6 parameters
        #
        #
        # Parameters
        # --------------------
        # learning_rate
        #       the weight of each change in parameter
        # num_iter
        #       number of iterations to converge
        # show_plots, verbose, save_fig
        #       self-explanatory
        # trouble
        #       trouble-shooting

        # Initializations
        parameter_array = [0.5, 0.5, 0.5, np.pi, np.pi, np.pi]
        del_parameters = np.zeros(len(parameter_array))
        
        del_J = np.zeros(6)

        Js = []
        J = 0
        maxJ = 0
        
        for j in range(num_iter):
            
            # Test Propagation and Evaluating Image Quality
            self.setSMFParameters(parameter_array)
            psf, wf_foc_u = self.prop(show_plots = False)
            J_new = self.strehl_ratio_integral(psf.power, self.refPSF.power)
            
            
            if save_fig:
                fig, ax = plt.subplots()
                hc.imshow_field(psf.power)
                plt.title('Receiver (Iteration %i)' % j)
                plt.xlabel('x (um)')
                plt.ylabel('y (um)')
                plt.colorbar()
                path = save_dir + 'psf/'
                
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + 'iter' + str(j) + '.png')
                plt.close(fig)
                
                fig1, ax1 = plt.subplots()
                hc.imshow_field(wf_foc_u.power)
                circ = plt.Circle((0,0), self.fiberCoreRadius, 
                                  edgecolor = 'white', 
                                  fill = False,
                                  linewidth = 2,
                                  alpha = 0.5)
                plt.gca().add_artist(circ)
                plt.xlabel('x (um)')
                plt.ylabel('y (um)')
                plt.title('Lantern Output (Iteration %i)' % j)
                plt.colorbar()
                
                path = save_dir + 'lanternOut/'
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + 'iter' + str(j) + '.png')
                plt.close(fig1)

                # hc.imshow_field(wf_foc_u.phase, cmap = 'cmr.emergency')
                # plt.colorbar()
                # plt.show()
            

                
            # Finite Differences calculation
            if j == 0:
                
                # random assignment of parameter to change
                k = math.floor(np.random.rand() * 6)
                
                # if k > 2, the parameter to change is phase, if k < 3, the parameter to change is intensity
                if k  > 2:
                    del_parameters[k] = 2 * np.random.rand() - 1
                elif k < 3:
                    del_parameters[k] = 0.1 * np.random.rand() - 0.05
                
            else:
                
                # change in image quality
                del_J[k] = float(J_new - J)
                
                # random assignment of parameter to change
                k = math.floor(np.random.rand() * 6)
                
                # if k > 2, the parameter to change is phase, if k < 3, the parameter to change is intensity
                if k > 2:
                    
                    # change-in-phase calculation
                    if del_J[k] == 0:     
                        del_parameters[k] = 2*np.random.rand() - 1
                    else:  
                        del_parameters[k] = math.fmod(learning_rate*del_J[k]/del_parameters[k], 2*np.pi)
                        
                    if del_parameters[k] == 0:
                        del_parameters[k] += 1e-5
                        
                elif k < 3:
                    
                    # change-in-phase calculation
                    if del_J[k] == 0:     
                        del_parameters[k] = 0.1*np.random.rand() - 0.05
                    else:  
                        del_parameters[k] = learning_rate*del_J[k]/del_parameters[k]
                        
                    if del_parameters[k] == 0:
                        del_parameters[k] += 1e-3
                
                
                
            if trouble:
    
                print("--------------- iteration ", j)
                print("Phase: ", parameter_array[3:6])
                print("image quality: ", J_new)
                print("Changing Paramater ", k)
                print("Change in Parameters: ", del_parameters)
                
            if J_new > maxJ:
                maxJ = J_new
                parameter_max_array = np.copy(parameter_array)
            
            J = J_new
            Js.append(J_new)
            
            if j > 20:
                
                std = np.std(Js[-10:])
                # print("std: ", std)
                
                if std < std_level:
                    break
                
            # setting parameters
            # if k > 2, the parameter to change is phase, if k < 3, the parameter to change is intensity
            if k > 2:
                
                parameter_array[k] += del_parameters[k]
                
                if parameter_array[k] < 0:
                    parameter_array[k] = 2*np.pi + parameter_array[k]
                elif parameter_array[k] > 2*np.pi:
                    parameter_array[k] = parameter_array[k] - 2*np.pi  
                    
            if k < 3:
                
                parameter_array[k] += del_parameters[k]
                
                if parameter_array[k] < 0:
                    parameter_array[k] = 0
                elif parameter_array[k] > 1:
                    parameter_array[k] = 1
            
            

                
        
        if show_plots:
            self.setSMFParameters(parameter_max_array)
            psf, wf_foc_u = self.prop(show_plots = False)
            
            hc.imshow_field(psf.power)
            plt.title('Gradient Descent (3 Mode)' )
            plt.colorbar()
            plt.show()
            
            hc.imshow_field(wf_foc_u.power)
            plt.title('Gradient Descent Lantern Output (3 Mode)')
            plt.colorbar()
            plt.show()
        
        if verbose:
            print("-------Gradient Descent (3 Mode)---------")
            print('Max J: ', maxJ)
            print('Intensities: ', parameter_max_array[0:3])
            print('Phases: ', parameter_max_array[3:6])
        
        return maxJ


    def GradientDescent_experimental(self, learning_rate = 1, num_iter = 100, std_level = 0.0, show_plots = False, verbose = False, trouble = False, save_fig = False, save_folder = 'psf/'):
        nModes = self.nModes
        parameter_array = np.concatenate((0.5*np.ones(nModes), np.pi * np.ones(nModes)))
        
        del_parameters = np.zeros(len(parameter_array))
        del_J = np.zeros(len(parameter_array))
        
        Js = []
        J = 0
        maxJ = 0
        
        for j in range(num_iter):
            self.setSMFParameters(parameter_array)
            psf, wf_foc_u = self.prop(show_plots = False)
            J_new = self.strehl_ratio_integral(psf.power, self.refPSF.power)
            
            if save_fig:
                fig, ax = plt.subplots()
                hc.imshow_field(psf.power)
                plt.title('Receiver (Iteration %i)' % j)
                plt.xlabel('x (um)')
                plt.ylabel('y (um)')
                txt = "J = " + str(J)
                text_box = AnchoredText(txt[0:9], frameon = True, loc = 4, pad = 0.5)
                plt.setp(text_box.patch, facecolor = 'white', alpha = 0.9)
                plt.gca().add_artist(text_box)
                plt.colorbar()
                path = save_dir + save_folder
                
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                plt.savefig(path + 'iter' + str(j) + '.png')
                plt.close(fig)
            
            if j == 0:
                k = math.floor(np.random.rand() * nModes * 2)
                
                if k < nModes:
                    del_parameters[k] = 0.1 * np.random.rand() - 0.05
                    
                elif k > nModes - 1:
                    del_parameters[k] = 2 * np.random.rand() - 1
                    
            
            else:
                
                del_J[k] = float(J_new - J)
                
                k = math.floor(np.random.rand() * nModes * 2)
            
                if k < nModes:
                    
                    if del_J[k] == 0:
                        del_parameters[k] = 0.1 * np.random.rand() - 0.05
                    else:
                        del_parameters[k] = learning_rate * del_J[k]/del_parameters[k]
                    
                    if del_parameters[k] == 0:
                        del_parameters[k] += 1e-3
                        
                elif k > nModes - 1:
                    
                    if del_J[k] == 0:
                        del_parameters[k] = 2*np.random.rand() - 1
                    else:
                        del_parameters[k] = math.fmod(learning_rate * del_J[k] / del_parameters[k], 2*np.pi)
                        
                    if del_parameters[k] == 0:
                        del_parameters[k] += 1e-5
                        
            
            if trouble:
                print("--------------- iteration ", j)
                print("Parameters: ", parameter_array)
                print("image quality: ", J_new)
                print("Changing Paramater ", k)
                print("Change in Parameters: ", del_parameters)
                
                
            if J_new > maxJ:
                maxJ = J_new
                parameter_max_array = np.copy(parameter_array)
                
            J = J_new
            Js.append(J_new)
            
            
            
            if k < nModes:
                
                parameter_array[k] += del_parameters[k]
                
                if parameter_array[k] < 0:
                    parameter_array[k] = 0
                elif parameter_array[k] > 1:
                    parameter_array[k] = 1
                
            elif k > nModes - 1:
                
                parameter_array[k] += del_parameters[k]
                
                if parameter_array[k] < 0:
                    parameter_array[k] = 2*np.pi + parameter_array[k]
                elif parameter_array[k] > 2*np.pi:
                    parameter_array[k] = parameter_array[k] - 2*np.pi 
                    
            
        if show_plots:
            
            parameter_array = np.concatenate((0.5*np.ones(nModes), np.pi * np.ones(nModes)))
            
            self.setSMFParameters(parameter_array)
            psf, wf_foc_u = self.prop(show_plots = False)
            J = self.strehl_ratio_integral(psf.power, self.refPSF.power)
            
            
            self.setSMFParameters(parameter_max_array)
            psf2, wf_foc_u = self.prop(show_plots = False)
            J2 = self.strehl_ratio_integral(psf2.power, self.refPSF.power)
            
            print("Initial J: ", J)
            
            # vmax = max(psf2.power)
            
            hc.imshow_field(psf.power)
            plt.title('Without GD (%i modes)' %nModes)
            txt = "J = " + str(J)
            text_box = AnchoredText(txt[0:9], frameon = True, loc = 4, pad = 0.5)
            plt.setp(text_box.patch, facecolor = 'white', alpha = 0.9)
            plt.gca().add_artist(text_box)
            plt.colorbar()
            plt.show()
            
            hc.imshow_field(psf2.power)
            plt.title('Gradient Descent (%i modes)' %nModes)
            txt = "J = " + str(J2)
            text_box = AnchoredText(txt[0:9], frameon = True, loc = 4, pad = 0.5)
            plt.setp(text_box.patch, facecolor = 'white', alpha = 0.9)
            plt.gca().add_artist(text_box)
            plt.colorbar()
            plt.show()
            
            hc.imshow_field(wf_foc_u.power)
            plt.title('Gradient Descent Lantern Output (%i modes)' %nModes)
            plt.colorbar()
            plt.show()
            
        if verbose:
            print("-------Gradient Descent (", nModes, "Mode)---------")
            print('Max J: ', maxJ)
            print('Intensities: ', parameter_max_array[0:nModes])
            print('Phases: ', parameter_max_array[nModes:nModes*2])
            
            
        return maxJ
            
    
    
    
    def animate(self, path, save_path, num_iter = 100):
        
        # Animates a series of images into gifs
        #
        #
        # Parameters
        # ------------------
        # path
        #       path of images
        # save_path
        #       path of the saved gif
        
        with imageio.get_writer(save_path + path.replace(save_path, "") +'.gif', mode = 'I', duration = 30, loop = 0) as writer:
            for i in range(num_iter - 1):
                img_name = path + '/iter' + str(i) + '.png'
                try:
                    image = imageio.imread(img_name)
                    writer.append_data(image)

                except Exception as e:
                    print(e)
                       
        writer.close()


    def singleRun(self, parameter_array):
        self.setSMFParameters(parameter_array)
        psf, wf_foc_u = self.prop(show_plots = False)
        J = self.strehl_ratio_integral(psf.power, self.refPSF.power)
        
        Jrev = 1 - J
        
        return Jrev
    
    
    def scipyOpt(self, method, show_plots = False, verbose = False):
        
        bounds = [(0,1), (0,1), (0,1), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        initial_guess = [0.5, 0.5, 0.5, np.pi, np.pi, np.pi]
        
        if method == 'global':
            res = optimize.differential_evolution(self.singleRun, bounds = bounds)
        
        else:
            res = optimize.minimize(self.singleRun, initial_guess, method = method, bounds = bounds)
        
        if verbose:
            print('------Scipy Optimization-------')
            print('Max J: ', 1 - res.fun)
            print('Intensities: ', res.x[0:3])
            print('Phases: ', res.x[3:6])
            
        if show_plots:
            self.setSMFParameters(res.x)
            psf, wf_foc_u = self.prop(show_plots = True)
        
        return res
        
