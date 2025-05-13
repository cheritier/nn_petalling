# -*- coding: utf-8 -*-
"""
Created on Tue May 13 08:00:01 2025

@author: cheritier
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.tools.tools import warning

# %%
plt.ion()
# number of subaperture for the WFS
n_subaperture = 20 # Harmoni: 90


#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 10,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.25,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 0 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

# Apply spiders to the telescope pupil
thickness_spider    = 0.2                                              # thickness of the spiders in m

if thickness_spider<(tel.pixelSize*2):
    warning('The thickness of the spider is too small and will not be shannon sampled! Consider using more pixel in the telescope resolution')
angle               = [0, 90, 180, 270]                                        # angle in degrees for each spider
offset_Y            = [-0.1, -0.1, 0.1, 0.1]                                     # shift offsets for each spider
offset_X            = None

tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

# display current pupil
plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel


#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object (1 layer atm)
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                  r0            = 0.15,                              # Fried Parameter @500 nm[m]
                  L0            = 25,                                # Outer Scale [m]
                  fractionalR0  = [1   ], # Cn2 Profile
                  windSpeed     = [10    ], # Wind Speed in [m]
                  windDirection = [0   ], # Wind Direction in [degrees]
                  altitude      = [50000 ]) # Altitude Layers in [m]

# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration


# specifying a given number of actuators along the diameter: 
nAct = n_subaperture+1
    
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = nAct-1,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct)                 # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# plot the dm actuators coordinates with respect to the pupil
plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**5,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')


#%% -----------------------     PYRAMID WFS   ----------------------------------
from OOPAO.Pyramid import Pyramid

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()

wfs = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 0,                            # Tip tilt modulation radius
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
              postProcessing    = 'fullFrame_incidence_flux')  # slopesMaps,

# propagate the light to the Wave-Front Sensor
tel*wfs



#%% -----------------------     Modal Basis - KL Basis  ----------------------------------
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel, atm, dm,lim = 1e-2,n_batch = 4) # matrix to apply modes on the DM

#%% -----------------------     Calibration: Interaction/Reconstruction Matrix  ----------------------------------
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

# # amplitude of the modes in m
# stroke=ngs.wavelength/40
# # zonal Interaction Matrix
# M2C_zonal = np.eye(dm.nValidAct)

# # modal Interaction Matrix for 300 modes
# M2C_modal = M2C_KL[:,:300]

# tel-atm
# # zonal interaction matrix
# calib_modal = InteractionMatrix(ngs            = ngs,
#                                 atm            = atm,
#                                 tel            = tel,
#                                 dm             = dm,
#                                 wfs            = wfs,   
#                                 M2C            = M2C_modal, # M2C matrix used 
#                                 stroke         = stroke,    # stroke for the push/pull in M2C units
#                                 nMeasurements  = 6,        # number of simultaneous measurements
#                                 noise          = 'off',     # disable wfs.cam noise 
#                                 display        = True,      # display the time using tqdm
#                                 single_pass    = True)      # only push to compute the interaction matrix instead of push-pull
#
# reconstructor = M2C_modal@calib_CL.M


def estimate_dm_correction(dm,wfs_signal_delayed, reconstructor):
    integrator_gain = 0.5
    dm_correction = dm.coefs-integrator_gain*np.matmul(reconstructor,wfs_signal_delayed)


#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector

# initialize Telescope DM commands
tel.resetOPD()
dm.coefs = 0
ngs*tel*dm*wfs
wfs*wfs.focal_plane_camera


# Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.r0 = 0.1
atm.generateNewPhaseScreen(seed = 10)

# combine telescope with atmosphere
tel+atm

# length of the simulation in AO loop frame (see tel.samplingTime)
nLoop = 500
# allocate memory to save data
input_turbulence            = np.zeros(nLoop)
residual_turbulence         = np.zeros(nLoop)
buffer_wfs_signal           = np.zeros((nLoop,wfs.nSignal))

# variable where the delayed wfs signal is stored
wfs_signal_delayed          = np.arange(0,wfs.nSignal)*0

# loop parameters
wfs.cam.photonNoise     = False
frame_delay             = 2
ao_correction           = False
display                 = True

# display
plt.close('all')
f = plt.figure()
plt.subplot(121)
opd_turb = plt.imshow(atm.OPD*tel.pupil)
plt.subplot(122)
opd_res = plt.imshow(atm.OPD*tel.pupil)


for i in range(nLoop):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    input_turbulence[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
    # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
    atm*ngs*tel*dm*wfs
    # save residuals corresponding to the NGS
    residual_turbulence[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9

    if frame_delay ==1:        
        wfs_signal_delayed = wfs.signal
    
    if ao_correction:
        # estimate the DM commands
        dm_correction = estimate_dm_correction(dm,wfs_signal_delayed)
        # apply the commands on the DM
        dm.coefs = dm_correction 
    
    # store the slopes after computing the commands => 2 frames delay
    if frame_delay ==2:        
        wfs_signal_delayed = wfs.signal
        
    if i%50==0 and display:
        # display only every 50 iteration
        opd_turb.set_data(atm.OPD*tel.pupil)
        opd_res.set_data(tel.OPD)
        plt.draw()
        plt.pause(0.1)
        
    
    print('Elapsed time: ' + str(time.time()-a) +' s')
    
    print('Loop'+str(i)+'/'+str(nLoop)+' input WFE: '+str(input_turbulence[i])+' -- residual WFE:' +str(residual_turbulence[i])+ '\n')