"""
MAKEPLOT_U19_COVARIANCES
  Generates plots that compare the wind component covariances between
  model and observations. The covariances from MAGIC are stored as 
  time series, so we must calculate the equivalent values from the model
  data -- this will be implemented once we know how it is done!

"""
date = 17 # Options: 15, 17 (dates in September 2019)

makefig = (1, )
# FIGURE 1: time--height plots of wind fields
# FIGURE 2: time series line plots of various wind properties
# FIGURE 3: line plots of the covariance terms.
# FIGURE 4: profiles of wind and temperature.

print("<MAKEPLOT_U19_COVARIANCES> Running...")

import iris
import iris.plot as iplt
import iris.time as irt
import numpy as np 
import matplotlib.pyplot as plt
plt.interactive(True)

### ----- BUILD THE FILENAMES.

# Create suite names and fileroots.
suitestr_17 = 'u-bv624'
startstr_17 = '20190916'
obsstart_17 = 135
obsend_17 = 186
suitestr_15 = 'u-bx053'
startstr_15 = '20190914'
obsstart_15 = 37
obsend_15 = 88

# Set up the filenames and roots.
fileroot_17 = '/data/users/joshonk/data/'+suitestr_17+'/'
fileroot_15 = '/data/users/joshonk/data/'+suitestr_15+'/'

filename_500_17 = startstr_17+'T1800Z_500m.nc9'
filename_500_17_pp0 = startstr_17+'T1800Z_500m.pp0'
filename_500_17_pp7 = startstr_17+'T1800Z_500m.pp7'

filename_100_17 = startstr_17+'T1800Z_100m.nc9'
filename_100_17_pp0 = startstr_17+'T1800Z_100m.pp0'
filename_100_17_pp7 = startstr_17+'T1800Z_100m.pp7'

filename_500_15 = startstr_15+'T1800Z_500m.nc9'
filename_500_15_pp0 = startstr_15+'T1800Z_500m.pp0'
filename_500_15_pp7 = startstr_15+'T1800Z_500m.pp7'

filename_100_15 = startstr_15+'T1800Z_100m.nc9'
filename_100_15_pp0 = startstr_15+'T1800Z_100m.pp0'
filename_100_15_pp7 = startstr_15+'T1800Z_100m.pp7'

filename_z = startstr_17+'T1800Z_500m.pp0'
filename_z_100 = startstr_17+'T1800Z_100m.pp0'

fileroot_obs = '/data/users/joshonk/data/obs/MAGIC/'
filename_obs = 'MAGIC_Campaign_lidar2sonics_30min.nc'

### ----- FETCH THE OBSERVED DATA IN IRIS.CUBE FORM
print("<MAKEPLOT_U19_COVARIANCES> Loading observed data...")
      
# Fetch wind data (components) from the observation file.
windx_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                iris.Constraint(name="u"))
windy_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                iris.Constraint(name="v"))
windz_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                iris.Constraint(name="w_mean"))
windzvar_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                   iris.Constraint(name="w_var"))

# Fetch BL depths (in the form of mixing heights).
#   NOTE: the reversal of the "sgr" and "sthr" variables is intentional --
#         there was an error in the processing, still to be resolved.
MLdepthsgr_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                iris.Constraint(name="mh_sthr"))
MLdepthsthr_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                iris.Constraint(name="mh_sgr"))

# Fetch wind data (covariances) from the observation file.
windxy_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                 iris.Constraint(name="uv_LSBU"))
windyz_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                 iris.Constraint(name="vw_LSBU"))
windxz_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                 iris.Constraint(name="uw_LSBU"))

# Fetch flux terms from the observation file.
SHsurf_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                 iris.Constraint(name="H_LSBU"))
LHsurf_cube_obs = iris.load_cube(fileroot_obs+filename_obs, \
                                 iris.Constraint(name="LE_LSBU"))

### ----- FETCH THE MODEL DATA IN IRIS.CUBE FORM
print("<MAKEPLOT_U19_COVARIANCES> Loading model data...")

windx_cube_500_17 \
    = iris.load_cube(fileroot_17+filename_500_17, \
                     iris.Constraint(name="x_wind"))
windy_cube_500_17 \
    = iris.load_cube(fileroot_17+filename_500_17, \
                     iris.Constraint(name="y_wind"))
windz_cube_500_17 \
    = iris.load_cube(fileroot_17+filename_500_17, \
                     iris.Constraint(name="upward_air_velocity"))
windx_cube_100_17 \
    = iris.load_cube(fileroot_17+filename_100_17, \
                     iris.Constraint(name="x_wind"))
windy_cube_100_17 \
    = iris.load_cube(fileroot_17+filename_100_17, \
                     iris.Constraint(name="y_wind"))
windz_cube_100_17 \
    = iris.load_cube(fileroot_17+filename_100_17, \
                     iris.Constraint(name="upward_air_velocity"))

pottemp_cube_500_17 \
    = iris.load_cube(fileroot_17+filename_500_17, \
                     iris.Constraint(name="air_potential_temperature"))
BLthick_cube_500_17 \
    = iris.load_cubes(fileroot_17+filename_500_17_pp0, \
                     iris.AttributeConstraint(STASH="m01s00i025"))
LHsurf_cube_500_17 \
    = iris.load_cube(fileroot_17+filename_500_17_pp0, \
                     iris.Constraint(name="surface_upward_latent_heat_flux"))
SHsurf_cube_500_17 \
    = iris.load_cube(fileroot_17+filename_500_17_pp7, \
                     iris.Constraint(name="SURFACE SENSIBLE  HEAT FLUX     W/M2"))
pottemp_cube_100_17 \
    = iris.load_cube(fileroot_17+filename_100_17, \
                     iris.Constraint(name="air_potential_temperature"))
BLthick_cube_100_17 \
    = iris.load_cubes(fileroot_17+filename_100_17_pp0, \
                     iris.AttributeConstraint(STASH="m01s00i025"))

windx_cube_500_15 \
    = iris.load_cube(fileroot_15+filename_500_15, \
                     iris.Constraint(name="x_wind"))
windy_cube_500_15 \
    = iris.load_cube(fileroot_15+filename_500_15, \
                     iris.Constraint(name="y_wind"))
windz_cube_500_15 \
    = iris.load_cube(fileroot_15+filename_500_15, \
                     iris.Constraint(name="upward_air_velocity"))
windx_cube_100_15 \
    = iris.load_cube(fileroot_15+filename_100_15, \
                     iris.Constraint(name="x_wind"))
windy_cube_100_15 \
    = iris.load_cube(fileroot_15+filename_100_15, \
                     iris.Constraint(name="y_wind"))
windz_cube_100_15 \
    = iris.load_cube(fileroot_15+filename_100_15, \
                     iris.Constraint(name="upward_air_velocity"))

pottemp_cube_500_15 \
    = iris.load_cube(fileroot_15+filename_500_15, \
                     iris.Constraint(name="air_potential_temperature"))
BLthick_cube_500_15 \
    = iris.load_cubes(fileroot_15+filename_500_15_pp0, \
                     iris.AttributeConstraint(STASH="m01s00i025"))
LHsurf_cube_500_15 \
    = iris.load_cube(fileroot_15+filename_500_15_pp0, \
                     iris.Constraint(name="surface_upward_latent_heat_flux"))
#SHsurf_cube_500_15 \
#    = iris.load_cube(fileroot_15+filename_500_15_pp7, \
#                     iris.Constraint(name="SURFACE SENSIBLE  HEAT FLUX     W/M2"))
pottemp_cube_100_15 \
    = iris.load_cube(fileroot_15+filename_100_15, \
                     iris.Constraint(name="air_potential_temperature"))
BLthick_cube_100_15 \
    = iris.load_cubes(fileroot_15+filename_100_15_pp0, \
                     iris.AttributeConstraint(STASH="m01s00i025"))

# Fetch a 3D field to extract the vertical axis.
flux_cube = iris.load_cube(fileroot_17+filename_z, \
                           iris.Constraint(name="upward_heat_flux_in_air"))
flux_cube_100 = iris.load_cube(fileroot_17+filename_z_100, \
                           iris.Constraint(name="upward_heat_flux_in_air"))

### ----- EXTRACT THE OBSERVED VALUES OF WIND
print("<MAKEPLOT_U19_COVARIANCES> Extracting and processing observed fields...")

# Extract the observed height profile (from the "range")
z_obs = windx_cube_obs.coord('range').points
z_obs_vert = windz_cube_obs.coord('z').points

# Extract wind components at the nearest level to 200 m, and the 
#   periods that line up with the forecast times.
windx_obs_17_full = windx_cube_obs.data[obsstart_17:obsend_17,:]
windy_obs_17_full = windy_cube_obs.data[obsstart_17:obsend_17,:]
windz_obs_17_full = windz_cube_obs.data[obsstart_17:obsend_17,:]
windzvar_obs_17_full = windzvar_cube_obs.data[obsstart_17:obsend_17,:]
windx_obs_15_full = windx_cube_obs.data[obsstart_15:obsend_15,:]
windy_obs_15_full = windy_cube_obs.data[obsstart_15:obsend_15,:]
windz_obs_15_full = windz_cube_obs.data[obsstart_15:obsend_15,:]
windzvar_obs_15_full = windzvar_cube_obs.data[obsstart_15:obsend_15,:]

lev_obs = np.where(windx_cube_obs.coord('range').points<210.0)[0][-1]
lev_obs_vert = np.where(windz_cube_obs.coord('z').points<210.0)[0][-1]
windx_obs_17_lev = windx_obs_17_full[:,lev_obs]
windy_obs_17_lev = windy_obs_17_full[:,lev_obs]
windz_obs_17_lev = windz_obs_17_full[:,lev_obs_vert]
windx_obs_15_lev = windx_obs_15_full[:,lev_obs]
windy_obs_15_lev = windy_obs_15_full[:,lev_obs]
windz_obs_15_lev = windz_obs_15_full[:,lev_obs_vert]

# Convert to speed and direction.
windm_obs_17_full = np.sqrt(windx_obs_17_full**2 + windy_obs_17_full**2)
winddir_obs_17_full = 90.0 - \
                     (np.arctan2(-windy_obs_17_full,-windx_obs_17_full) \
                      / np.pi * 180.0)
ptlist = np.where(winddir_obs_17_full < 0.0)
winddir_obs_17_full[ptlist] = winddir_obs_17_full[ptlist] + 360.0

windm_obs_15_full = np.sqrt(windx_obs_15_full**2 + windy_obs_15_full**2)
winddir_obs_15_full = 90.0 - \
                     (np.arctan2(-windy_obs_15_full,-windx_obs_15_full) \
                      / np.pi * 180.0)
ptlist = np.where(winddir_obs_15_full < 0.0)
winddir_obs_15_full[ptlist] = winddir_obs_15_full[ptlist] + 360.0

windm_obs_17_lev = np.sqrt(windx_obs_17_lev**2 + windy_obs_17_lev**2)
winddir_obs_17_lev = 90.0 - \
                     (np.arctan2(-windy_obs_17_lev,-windx_obs_17_lev) \
                      / np.pi * 180.0)
ptlist = np.where(winddir_obs_17_lev < 0.0)
winddir_obs_17_lev[ptlist] = winddir_obs_17_lev[ptlist] + 360.0

windm_obs_15_lev = np.sqrt(windx_obs_15_lev**2 + windy_obs_15_lev**2)
winddir_obs_15_lev = 90.0 - \
                     (np.arctan2(-windy_obs_15_lev,-windx_obs_15_lev) \
                      / np.pi * 180.0)
ptlist = np.where(winddir_obs_15_lev < 0.0)
winddir_obs_15_lev[ptlist] = winddir_obs_15_lev[ptlist] + 360.0

### ----- EXTRACT THE OBSERVED SURFACE FIELDS (COVARIANCES/FLUXES/BL DEPTHS)

# Extract wind covariances and surface fluxes.
windxy_obs_17 = windxy_cube_obs.data[obsstart_17:obsend_17]
windyz_obs_17 = windyz_cube_obs.data[obsstart_17:obsend_17]
windxz_obs_17 = windxz_cube_obs.data[obsstart_17:obsend_17]
SHsurf_obs_17 = SHsurf_cube_obs.data[obsstart_17:obsend_17]
LHsurf_obs_17 = LHsurf_cube_obs.data[obsstart_17:obsend_17]
time_obs_17 = windxy_cube_obs.coord("time")[obsstart_17:obsend_17].points
windxy_obs_15 = windxy_cube_obs.data[obsstart_15:obsend_15]
windyz_obs_15 = windyz_cube_obs.data[obsstart_15:obsend_15]
windxz_obs_15 = windxz_cube_obs.data[obsstart_15:obsend_15]
SHsurf_obs_15 = SHsurf_cube_obs.data[obsstart_15:obsend_15]
LHsurf_obs_15 = LHsurf_cube_obs.data[obsstart_15:obsend_15]
time_obs_15 = windxy_cube_obs.coord("time")[obsstart_15:obsend_15].points
timehr_obs = np.arange(-5.5,20.0,0.5)
    
MLdepthsgr_obs_17 = MLdepthsgr_cube_obs.data[obsstart_17:obsend_17]
MLdepthsthr_obs_17 = MLdepthsthr_cube_obs.data[obsstart_17:obsend_17]
MLdepthsgr_obs_15 = MLdepthsgr_cube_obs.data[obsstart_15:obsend_15]
MLdepthsthr_obs_15 = MLdepthsthr_cube_obs.data[obsstart_15:obsend_15]

# Calculate the friction velocity as a metric of drag.
ustar_obs_17 = np.sqrt(windxz_obs_17**2 + windyz_obs_17**2)
ustar_obs_15 = np.sqrt(windxz_obs_15**2 + windyz_obs_15**2)

#cheese.cake
### ----- EXTRACT THE MODEL VALUES OF WIND
print("<MAKEPLOT_U19_COVARIANCES> Extracting and processing model fields...")

# Extract the model height profile and co-ordinates.
z_500 = flux_cube.coord("level_height").points[0:100]
lat_500 = flux_cube.coord("grid_latitude").points
long_500 = flux_cube.coord("grid_longitude").points
z_100 = flux_cube_100.coord("level_height").points[0:100]
lat_100 = flux_cube_100.coord("grid_latitude").points
long_100 = flux_cube_100.coord("grid_longitude").points

# Extract the full fields.
windx_500_17_all_full = windx_cube_500_17.data
windy_500_17_all_full = windy_cube_500_17.data
windz_500_17_all_full = windz_cube_500_17.data
pottemp_500_17_all_full = pottemp_cube_500_17.data
windx_100_17_all_full = windx_cube_100_17.data
windy_100_17_all_full = windy_cube_100_17.data
windz_100_17_all_full = windz_cube_100_17.data
pottemp_100_17_all_full = pottemp_cube_100_17.data

windx_500_15_all_full = windx_cube_500_15.data
windy_500_15_all_full = windy_cube_500_15.data
windz_500_15_all_full = windz_cube_500_15.data
pottemp_500_15_all_full = pottemp_cube_500_15.data
windx_100_15_all_full = windx_cube_100_15.data
windy_100_15_all_full = windy_cube_100_15.data
windz_100_15_all_full = windz_cube_100_15.data
pottemp_100_15_all_full = pottemp_cube_100_15.data

# Extract wind components at the nearest level to 200 m.
lev_500 = np.where(z_500<210.0)[0][-1]
lev_100 = np.where(z_100<210.0)[0][-1]

windx_500_17_all_lev = windx_500_17_all_full[:,lev_500]
windy_500_17_all_lev = windy_500_17_all_full[:,lev_500]
windz_500_17_all_lev = windz_500_17_all_full[:,lev_500]
windx_100_17_all_lev = windx_100_17_all_full[:,lev_100]
windy_100_17_all_lev = windy_100_17_all_full[:,lev_100]
windz_100_17_all_lev = windz_100_17_all_full[:,lev_100]

windx_500_15_all_lev = windx_500_15_all_full[:,lev_500]
windy_500_15_all_lev = windy_500_15_all_full[:,lev_500]
windz_500_15_all_lev = windz_500_15_all_full[:,lev_500]
windx_100_15_all_lev = windx_100_15_all_full[:,lev_100]
windy_100_15_all_lev = windy_100_15_all_full[:,lev_100]
windz_100_15_all_lev = windz_100_15_all_full[:,lev_100]

time_500 = windx_cube_500_17.coord("time").points * 24.0 - 24.0
time_100 = windx_cube_100_17.coord("time").points * 24.0


### ----- EXTRACT THE LSBU POINT FROM THE SURFACE FIELDS

latpt_500 = 205
longpt_500 = 365
latpt_100 = 372
longpt_100 = 403
latpt_55 = 669
longpt_55 = 725

BLthick_500_17 = BLthick_cube_500_17[0].data[:,latpt_500,longpt_500]
BLthick_500_15 = BLthick_cube_500_15[0].data[:,latpt_500,longpt_500]
BLthick_100_17 = BLthick_cube_100_17[0].data[:,latpt_100,longpt_100]
BLthick_100_15 = BLthick_cube_100_15[0].data[:,latpt_100,longpt_100]


time_BL_500 = np.arange(-5.5,20.5,1.0)
time_BL_100 = np.arange(4.5,20.5,1.0)

### ----- AVERAGE MODEL WIND INTO 30-MINUTE SECTIONS
print("<MAKEPLOT_U19_COVARIANCES> Averaging model data on to a 30-minute grid...")

# Create blank arrays or Python whinges.
windx_500_17_mean_full = np.zeros((50,100))
windy_500_17_mean_full = np.zeros((50,100))
windz_500_17_mean_full = np.zeros((50,100))
windz_500_17_var_full = np.zeros((50,100))
pottemp_500_17_mean_full = np.zeros((50,100))
windx_100_17_mean_full = np.zeros((30,100))
windy_100_17_mean_full = np.zeros((30,100))
windz_100_17_mean_full = np.zeros((30,100))
windz_100_17_var_full = np.zeros((30,100))
pottemp_100_17_mean_full = np.zeros((30,100))

windx_500_15_mean_full = np.zeros((50,100))
windy_500_15_mean_full = np.zeros((50,100))
windz_500_15_mean_full = np.zeros((50,100))
windz_500_15_var_full = np.zeros((50,100))
pottemp_500_15_mean_full = np.zeros((50,100))
windx_100_15_mean_full = np.zeros((30,100))
windy_100_15_mean_full = np.zeros((30,100))
windz_100_15_mean_full = np.zeros((30,100))
windz_100_15_var_full = np.zeros((30,100))
pottemp_100_15_mean_full = np.zeros((30,100))

windx_500_17_mean_lev = np.zeros((50,))
windy_500_17_mean_lev = np.zeros((50,))
windz_500_17_mean_lev = np.zeros((50,))
windx_100_17_mean_lev = np.zeros((30,))
windy_100_17_mean_lev = np.zeros((30,))
windz_100_17_mean_lev = np.zeros((30,))

windx_500_15_mean_lev = np.zeros((50,))
windy_500_15_mean_lev = np.zeros((50,))
windz_500_15_mean_lev = np.zeros((50,))
windx_100_15_mean_lev = np.zeros((30,))
windy_100_15_mean_lev = np.zeros((30,))
windz_100_15_mean_lev = np.zeros((30,))

time_500_mean = np.zeros((50,))
time_100_mean = np.zeros((30,))

# Start and end time steps for each 30-minute period: 500 m model...
startstep = np.arange(0,9000,180)
endstep = startstep + 180

# Calculate means (500 m model).
for istep in range(50): 

    windx_500_17_mean_full[istep] \
        = np.mean(windx_500_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windy_500_17_mean_full[istep] \
        = np.mean(windy_500_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_500_17_mean_full[istep] \
        = np.mean(windz_500_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_500_17_var_full[istep] \
        = np.var(windz_500_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    pottemp_500_17_mean_full[istep] \
        = np.mean(pottemp_500_17_all_full[startstep[istep] \
                                          :endstep[istep],:],axis=0) 
    windx_500_15_mean_full[istep] \
        = np.mean(windx_500_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windy_500_15_mean_full[istep] \
        = np.mean(windy_500_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_500_15_mean_full[istep] \
        = np.mean(windz_500_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_500_15_var_full[istep] \
        = np.var(windz_500_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    pottemp_500_15_mean_full[istep] \
        = np.mean(pottemp_500_15_all_full[startstep[istep] \
                                          :endstep[istep],:],axis=0) 

    windx_500_17_mean_lev[istep] \
        = np.mean(windx_500_17_all_lev[startstep[istep]:endstep[istep]]) 
    windy_500_17_mean_lev[istep] \
        = np.mean(windy_500_17_all_lev[startstep[istep]:endstep[istep]]) 
    windz_500_17_mean_lev[istep] \
        = np.mean(windz_500_17_all_lev[startstep[istep]:endstep[istep]]) 
    windx_500_15_mean_lev[istep] \
        = np.mean(windx_500_15_all_lev[startstep[istep]:endstep[istep]]) 
    windy_500_15_mean_lev[istep] \
        = np.mean(windy_500_15_all_lev[startstep[istep]:endstep[istep]]) 
    windz_500_15_mean_lev[istep] \
        = np.mean(windz_500_15_all_lev[startstep[istep]:endstep[istep]]) 

    time_500_mean[istep] = time_500[startstep[istep]]

# Start and end time steps for each 30-minute period: 500 m model...
startstep = np.arange(0,18000,600)
endstep = startstep + 600

# Calculate means.
for istep in range(30): 

    windx_100_17_mean_full[istep] \
        = np.mean(windx_100_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windy_100_17_mean_full[istep] \
        = np.mean(windy_100_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_100_17_mean_full[istep] \
        = np.mean(windz_100_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_100_17_var_full[istep] \
        = np.var(windz_100_17_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    pottemp_100_17_mean_full[istep] \
        = np.mean(pottemp_100_17_all_full[startstep[istep] \
                                          :endstep[istep],:],axis=0) 
    windx_100_15_mean_full[istep] \
        = np.mean(windx_100_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windy_100_15_mean_full[istep] \
        = np.mean(windy_100_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_100_15_mean_full[istep] \
        = np.mean(windz_100_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    windz_100_15_var_full[istep] \
        = np.var(windz_100_15_all_full[startstep[istep] \
                                        :endstep[istep],:],axis=0) 
    pottemp_100_15_mean_full[istep] \
        = np.mean(pottemp_100_15_all_full[startstep[istep] \
                                          :endstep[istep],:],axis=0) 

    windx_100_17_mean_lev[istep] \
        = np.mean(windx_100_17_all_lev[startstep[istep]:endstep[istep]]) 
    windy_100_17_mean_lev[istep] \
        = np.mean(windy_100_17_all_lev[startstep[istep]:endstep[istep]]) 
    windz_100_17_mean_lev[istep] \
        = np.mean(windz_100_17_all_lev[startstep[istep]:endstep[istep]]) 
    windx_100_15_mean_lev[istep] \
        = np.mean(windx_100_15_all_lev[startstep[istep]:endstep[istep]]) 
    windy_100_15_mean_lev[istep] \
        = np.mean(windy_100_15_all_lev[startstep[istep]:endstep[istep]]) 
    windz_100_15_mean_lev[istep] \
        = np.mean(windz_100_15_all_lev[startstep[istep]:endstep[istep]]) 

    time_100_mean[istep] = time_100[startstep[istep]]

# Convert to speed and direction.
windm_500_17_mean_full = np.sqrt(windx_500_17_mean_full**2 + windy_500_17_mean_full**2)
winddir_500_17_mean_full = 90.0 - \
                      (np.arctan2(-windy_500_17_mean_full,-windx_500_17_mean_full) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_500_17_mean_full < 0.0)
winddir_500_17_mean_full[ptlist] = winddir_500_17_mean_full[ptlist] + 360.0

windm_100_17_mean_full = np.sqrt(windx_100_17_mean_full**2 + windy_100_17_mean_full**2)
winddir_100_17_mean_full = 90.0 - \
                      (np.arctan2(-windy_100_17_mean_full,-windx_100_17_mean_full) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_100_17_mean_full < 0.0)
winddir_100_17_mean_full[ptlist] = winddir_100_17_mean_full[ptlist] + 360.0

windm_500_15_mean_full = np.sqrt(windx_500_15_mean_full**2 + windy_500_15_mean_full**2)
winddir_500_15_mean_full = 90.0 - \
                      (np.arctan2(-windy_500_15_mean_full,-windx_500_15_mean_full) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_500_15_mean_full < 0.0)
winddir_500_15_mean_full[ptlist] = winddir_500_15_mean_full[ptlist] + 360.0

windm_100_15_mean_full = np.sqrt(windx_100_15_mean_full**2 + windy_100_15_mean_full**2)
winddir_100_15_mean_full = 90.0 - \
                      (np.arctan2(-windy_100_15_mean_full,-windx_100_15_mean_full) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_100_15_mean_full < 0.0)
winddir_100_15_mean_full[ptlist] = winddir_100_15_mean_full[ptlist] + 360.0

windm_500_17_mean_lev = np.sqrt(windx_500_17_mean_lev**2 + windy_500_17_mean_lev**2)
winddir_500_17_mean_lev = 90.0 - \
                      (np.arctan2(-windy_500_17_mean_lev,-windx_500_17_mean_lev) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_500_17_mean_lev < 0.0)
winddir_500_17_mean_lev[ptlist] = winddir_500_17_mean_lev[ptlist] + 360.0

windm_100_17_mean_lev = np.sqrt(windx_100_17_mean_lev**2 + windy_100_17_mean_lev**2)
winddir_100_17_mean_lev = 90.0 - \
                      (np.arctan2(-windy_100_17_mean_lev,-windx_100_17_mean_lev) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_100_17_mean_lev < 0.0)
winddir_100_17_mean_lev[ptlist] = winddir_100_17_mean_lev[ptlist] + 360.0

windm_500_15_mean_lev = np.sqrt(windx_500_15_mean_lev**2 + windy_500_15_mean_lev**2)
winddir_500_15_mean_lev = 90.0 - \
                      (np.arctan2(-windy_500_15_mean_lev,-windx_500_15_mean_lev) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_500_15_mean_lev < 0.0)
winddir_500_15_mean_lev[ptlist] = winddir_500_15_mean_lev[ptlist] + 360.0

windm_100_15_mean_lev = np.sqrt(windx_100_15_mean_lev**2 + windy_100_15_mean_lev**2)
winddir_100_15_mean_lev = 90.0 - \
                      (np.arctan2(-windy_100_15_mean_lev,-windx_100_15_mean_lev) \
                       / np.pi * 180.0)
ptlist = np.where(winddir_100_15_mean_lev < 0.0)
winddir_100_15_mean_lev[ptlist] = winddir_100_15_mean_lev[ptlist] + 360.0

#cheese.cake

### ----- GENERATE 30-MINUTE WIND COVARIANCES
print("<MAKEPLOT_U19_COVARIANCES> Calculating 30-minute wind covariances...")

# Create blank arrays or Python whinges.
windxz_500_17_mean_full = np.zeros((50,100))
windyz_500_17_mean_full = np.zeros((50,100))
windxz_500_15_mean_full = np.zeros((50,100))
windyz_500_15_mean_full = np.zeros((50,100))
windxz_100_17_mean_full = np.zeros((30,100))
windyz_100_17_mean_full = np.zeros((30,100))
windxz_100_15_mean_full = np.zeros((30,100))
windyz_100_15_mean_full = np.zeros((30,100))

windxz_500_17_mean_lev = np.zeros((50,))
windyz_500_17_mean_lev = np.zeros((50,))
windxz_500_15_mean_lev = np.zeros((50,))
windyz_500_15_mean_lev = np.zeros((50,))
windxz_100_17_mean_lev = np.zeros((30,))
windyz_100_17_mean_lev = np.zeros((30,))
windxz_100_15_mean_lev = np.zeros((30,))
windyz_100_15_mean_lev = np.zeros((30,))

# Cycle through 30-minute windows and calculate mean covariance.
startstep = np.arange(0,9000,180)
endstep = startstep + 180

for istep in range(50): 
    windx_500_17_block  \
        = windx_500_17_all_lev[startstep[istep]:endstep[istep]] \
        - windx_500_17_mean_lev[istep]
    windy_500_17_block  \
        = windy_500_17_all_lev[startstep[istep]:endstep[istep]] \
        - windy_500_17_mean_lev[istep]
    windz_500_17_block  \
        = windz_500_17_all_lev[startstep[istep]:endstep[istep]] \
        - windz_500_17_mean_lev[istep]

    windxz_500_17_mean_lev[istep] \
        = np.mean(windx_500_17_block * windz_500_17_block)
    windyz_500_17_mean_lev[istep] \
        = np.mean(windy_500_17_block * windz_500_17_block)

    windx_500_15_block  \
        = windx_500_15_all_lev[startstep[istep]:endstep[istep]] \
        - windx_500_15_mean_lev[istep]
    windy_500_15_block  \
        = windy_500_15_all_lev[startstep[istep]:endstep[istep]] \
        - windy_500_15_mean_lev[istep]
    windz_500_15_block  \
        = windz_500_15_all_lev[startstep[istep]:endstep[istep]] \
        - windz_500_15_mean_lev[istep]

    windxz_500_15_mean_lev[istep] \
        = np.mean(windx_500_15_block * windz_500_15_block)
    windyz_500_15_mean_lev[istep] \
        = np.mean(windy_500_15_block * windz_500_15_block)

for istep in range(50): 
    windx_500_17_stack  \
        = windx_500_17_all_full[startstep[istep]:endstep[istep],:] \
        - windx_500_17_mean_full[istep]
    windy_500_17_stack  \
        = windy_500_17_all_full[startstep[istep]:endstep[istep],:] \
        - windy_500_17_mean_lev[istep]
    windz_500_17_stack  \
        = windz_500_17_all_full[startstep[istep]:endstep[istep],:] \
        - windz_500_17_mean_full[istep]

    windxz_500_17_mean_full[istep,:] \
        = np.mean(windx_500_17_stack * windz_500_17_stack,axis=0)
    windyz_500_17_mean_full[istep] \
        = np.mean(windy_500_17_stack * windz_500_17_stack,axis=0)

    windx_500_15_stack  \
        = windx_500_15_all_full[startstep[istep]:endstep[istep],:] \
        - windx_500_15_mean_full[istep]
    windy_500_15_stack  \
        = windy_500_15_all_full[startstep[istep]:endstep[istep],:] \
        - windy_500_15_mean_lev[istep]
    windz_500_15_stack  \
        = windz_500_15_all_full[startstep[istep]:endstep[istep],:] \
        - windz_500_15_mean_full[istep]

    windxz_500_15_mean_full[istep,:] \
        = np.mean(windx_500_15_stack * windz_500_15_stack,axis=0)
    windyz_500_15_mean_full[istep] \
        = np.mean(windy_500_15_stack * windz_500_15_stack,axis=0)

# Repeat for the 100 m model.
startstep = np.arange(0,18000,600)
endstep = startstep + 600

for istep in range(30): 
    windx_100_17_block  \
        = windx_100_17_all_lev[startstep[istep]:endstep[istep]] \
        - windx_100_17_mean_lev[istep]
    windy_100_17_block  \
        = windy_100_17_all_lev[startstep[istep]:endstep[istep]] \
        - windy_100_17_mean_lev[istep]
    windz_100_17_block  \
        = windz_100_17_all_lev[startstep[istep]:endstep[istep]] \
        - windz_100_17_mean_lev[istep]

    windxz_100_17_mean_lev[istep] \
        = np.mean(windx_100_17_block * windz_100_17_block)
    windyz_100_17_mean_lev[istep] \
        = np.mean(windy_100_17_block * windz_100_17_block)

    windx_100_15_block  \
        = windx_100_15_all_lev[startstep[istep]:endstep[istep]] \
        - windx_100_15_mean_lev[istep]
    windy_100_15_block  \
        = windy_100_15_all_lev[startstep[istep]:endstep[istep]] \
        - windy_100_15_mean_lev[istep]
    windz_100_15_block  \
        = windz_100_15_all_lev[startstep[istep]:endstep[istep]] \
        - windz_100_15_mean_lev[istep]

    windxz_100_15_mean_lev[istep] \
        = np.mean(windx_100_15_block * windz_100_15_block)
    windyz_100_15_mean_lev[istep] \
        = np.mean(windy_100_15_block * windz_100_15_block)

for istep in range(30): 
    windx_100_17_stack  \
        = windx_100_17_all_full[startstep[istep]:endstep[istep],:] \
        - windx_100_17_mean_full[istep]
    windy_100_17_stack  \
        = windy_100_17_all_full[startstep[istep]:endstep[istep],:] \
        - windy_100_17_mean_lev[istep]
    windz_100_17_stack  \
        = windz_100_17_all_full[startstep[istep]:endstep[istep],:] \
        - windz_100_17_mean_full[istep]

    windxz_100_17_mean_full[istep,:] \
        = np.mean(windx_100_17_stack * windz_100_17_stack,axis=0)
    windyz_100_17_mean_full[istep] \
        = np.mean(windy_100_17_stack * windz_100_17_stack,axis=0)

    windx_100_15_stack  \
        = windx_100_15_all_full[startstep[istep]:endstep[istep],:] \
        - windx_100_15_mean_full[istep]
    windy_100_15_stack  \
        = windy_100_15_all_full[startstep[istep]:endstep[istep],:] \
        - windy_100_15_mean_lev[istep]
    windz_100_15_stack  \
        = windz_100_15_all_full[startstep[istep]:endstep[istep],:] \
        - windz_100_15_mean_full[istep]

    windxz_100_15_mean_full[istep,:] \
        = np.mean(windx_100_15_stack * windz_100_15_stack,axis=0)
    windyz_100_15_mean_full[istep] \
        = np.mean(windy_100_15_stack * windz_100_15_stack,axis=0)


# Calculate the friction velocity from this.
ustar_500_17_lev = np.sqrt(windxz_500_17_mean_lev**2   \
                           + windyz_500_17_mean_lev**2)
ustar_500_15_lev = np.sqrt(windxz_500_15_mean_lev**2   \
                           + windyz_500_15_mean_lev**2)
ustar_500_17_full = np.sqrt(windxz_500_17_mean_full**2   \
                            + windyz_500_17_mean_full**2)
ustar_500_15_full = np.sqrt(windxz_500_15_mean_full**2   \
                            + windyz_500_15_mean_full**2)
ustar_100_17_lev = np.sqrt(windxz_100_17_mean_lev**2   \
                           + windyz_100_17_mean_lev**2)
ustar_100_15_lev = np.sqrt(windxz_100_15_mean_lev**2   \
                           + windyz_100_15_mean_lev**2)
ustar_100_17_full = np.sqrt(windxz_100_17_mean_full**2   \
                            + windyz_100_17_mean_full**2)
ustar_100_15_full = np.sqrt(windxz_100_15_mean_full**2   \
                            + windyz_100_15_mean_full**2)


#--- Create a figure.

# FIGURE 2: time series line plots of various wind properties
if 2 in makefig:
    print("<MAKEPLOT_U19_COVARIANCES> Making Figure 2...")
   
    # Time series of friction velocity, wind and wind direction at 200 m
    plt.figure(figsize=(10,9))
    
    plt.subplot(3,2,1)
    plt.plot(timehr_obs,ustar_obs_15, \
         color="red", linewidth=2)
    plt.plot(time_500_mean,ustar_500_15_lev, \
             color="green", linewidth=2)
    plt.plot(time_100_mean,ustar_100_15_lev, \
             color="blue", linewidth=2)
    #plt.xlabel("Time after 00:00 / h")
    plt.ylabel("friction velocity / $\mathrm{m}\, \mathrm{s}^{-1}$")
    plt.grid(True)
    plt.ylim((-0.5,2.0))
    #plt.legend(("u'w'","v'w'"))
    plt.title("15 Sep 2019 (westerly case)")
    
    plt.subplot(3,2,3)
    plt.plot(timehr_obs,windm_obs_15_lev, \
             color="red", linewidth=2)
    plt.plot(time_500_mean,windm_500_15_mean_lev, \
             color="green", linewidth=2)
    plt.plot(time_100_mean,windm_100_15_mean_lev, \
             color="blue", linewidth=2)
    #plt.xlabel("Time after 00:00 / h")
    plt.ylabel("wind speed / $\mathrm{m}\, \mathrm{s}^{-1}$")
    plt.grid(True)
    plt.ylim((0,10))
    
    plt.subplot(3,2,5)
    plt.plot(timehr_obs,winddir_obs_15_lev,'o', \
             color="red", linewidth=2)
    plt.plot(time_500_mean,winddir_500_15_mean_lev,'o', \
             color="green", linewidth=2)
    plt.plot(time_100_mean,winddir_100_15_mean_lev,'o', \
             color="blue", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    plt.ylabel("wind direction / $^\circ$")
    plt.grid(True)
    plt.ylim((0,360))
    plt.legend(("obs","500 m","100 m"))
    
    plt.subplot(3,2,2)
    plt.plot(timehr_obs,ustar_obs_17, \
             color="red", linewidth=2)
    plt.plot(time_500_mean,ustar_500_17_lev, \
             color="green", linewidth=2)
    plt.plot(time_100_mean,ustar_100_17_lev, \
             color="blue", linewidth=2)
    #plt.xlabel("Time after 00:00 / h")
    #plt.ylabel("friction velocity / $\mathrm{m}\, \mathrm{s}^{-1}$")
    plt.grid(True)
    plt.ylim((-0.5,2.0))
    plt.title("17 Sep 2019 (northerly case)")
    
    plt.subplot(3,2,4)
    plt.plot(timehr_obs,windm_obs_17_lev, \
             color="red", linewidth=2)
    plt.plot(time_500_mean,windm_500_17_mean_lev, \
             color="green", linewidth=2)
    plt.plot(time_100_mean,windm_100_17_mean_lev, \
             color="blue", linewidth=2)
    #plt.xlabel("Time after 00:00 / h")
    #plt.ylabel("wind speed / $\mathrm{m}\, \mathrm{s}^{-1}$")
    plt.grid(True)
    plt.ylim((0,10))
    plt.text(-5,8,'Obs wind speed/dir from BT',fontsize=10)
    
    plt.subplot(3,2,6)
    plt.plot(timehr_obs,winddir_obs_17_lev,'o', \
             color="red", linewidth=2)
    plt.plot(time_500_mean,winddir_500_17_mean_lev,'o', \
             color="green", linewidth=2)
    plt.plot(time_100_mean,winddir_100_17_mean_lev,'o', \
             color="blue", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    #plt.ylabel("wind direction / $^\circ$")
    plt.grid(True)
    plt.ylim((0,360))
    
    plt.suptitle("Time series of wind and turbulence properties "+ \
                 "from the two test cases.")

    # Time series of boundary layer dimensions
    plt.figure(figsize=(10,6))
    
    plt.subplot(1,2,1)
    plt.plot(time_BL_500,BLthick_500_15,color="green",linewidth=2)
    plt.plot(time_BL_100,BLthick_100_15,color="blue", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    plt.ylabel("boundary layer depth / m")
    plt.grid(True)
    plt.ylim((0,2000))
    plt.title("15 Sep 2019 (westerly case)")

    plt.subplot(1,2,2)
    plt.plot(time_BL_500,BLthick_500_17,color="green",linewidth=2)
    plt.plot(time_BL_100,BLthick_100_17,color="blue", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
#    plt.ylabel("boundary layer depth / m")
    plt.grid(True)
    plt.ylim((0,2000))
    plt.title("15 Sep 2019 (westerly case)")
    plt.legend(("500 m","100 m"))

    plt.suptitle("Time series of boundary layer dimensions "+ \
                 "from the two test cases.")
 

# FIGURE 3: line plots of the covariance terms; now replaced by Figure 2 
if 3 in makefig:
    print("<MAKEPLOT_U19_COVARIANCES> Making Figure 3...")

    plt.figure(figsize=(10,9))
    
    plt.subplot(1,2,1)
    #plt.plot(timehr_obs,windxz_obs_15, \
        #         color="red", linewidth=2)
    #plt.plot(timehr_obs,windyz_obs_15, \
        #         color="green", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    plt.ylabel("wind components / $\mathrm{m}\, \mathrm{s}^{-1}$")
    plt.grid(True)
    plt.ylim((-0.5,1.5))
    plt.legend(("u'w'","v'w'"))
    plt.title("Vertical component; 15 Sep 2019")
    
    plt.subplot(1,2,2)
    plt.plot(timehr_obs,windxz_obs_17, \
             color="red", linewidth=2)
    #plt.plot(timehr_obs,windyz_obs_17, \
        #         color="green", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    plt.ylabel("wind components / $\mathrm{m}\, \mathrm{s}^{-1}$")
    plt.grid(True)
    plt.ylim((-0.5,1.5))
    plt.title("Vertical component; 17 Sep 2019")
    
    
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,2,1)
    plt.plot(timehr_obs,windxy_obs_15, \
             color="blue", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    plt.ylabel("stress term (u'v') / $\mathrm{m}^2\, \mathrm{s}^{-2}$")
    plt.grid(True)
    plt.ylim((-1,1))
    plt.legend(("u'v'"))
    plt.title("Horizontal component; 15 Sep 2019")
    
    plt.subplot(1,2,2)
    plt.plot(timehr_obs,windxy_obs_17, \
             color="blue", linewidth=2)
    plt.xlabel("Time after 00:00 / h")
    plt.ylabel("stress term / $\mathrm{m}^2\, \mathrm{s}^{-2}$")
    plt.grid(True)
    plt.ylim((-1,1))
    plt.title("Horizontal component; 17 Sep 2019")

 
# FIGURE SET 1: time--height plots of wind fields
if 1 in makefig:  # 500 m vs obs; test plots at matching resolution
    print("<MAKEPLOT_U19_COVARIANCES> Making Figure 1...")

    # FIGURE A1. Wind magnitude time--height plot.
    plt.figure(figsize=(12,10))

    plt.subplot(3,2,1)
    plt.pcolor(timehr_obs,z_obs,np.transpose(windm_obs_17_full), \
               vmin=0,vmax=10,cmap="hot")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_17[0:-1:2],'bs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_17[0:-1:2],'bD', \
             MarkerSize=5,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 17 September")
    cb1 = plt.colorbar()
#    cb1.set_label("wind speed / m/s")
    
    plt.subplot(3,2,3)
    plt.pcolor(time_500_mean,z_500,np.transpose(windm_500_17_mean_full), \
               vmin=0,vmax=10,cmap="hot")
    plt.plot(time_BL_500,BLthick_500_17,'bo',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("wind speed / m/s")

    plt.subplot(3,2,5)
    plt.pcolor(time_100_mean,z_100,np.transpose(windm_100_17_mean_full), \
               vmin=0,vmax=10,cmap="hot")
    plt.plot(time_BL_100,BLthick_100_17,'bo', \
             MarkerSize=6,MarkerFaceColor="None",label="parcel method")
    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("wind speed / m/s")
    plt.legend(loc="upper left")
 
    plt.subplot(3,2,2)
    plt.pcolor(timehr_obs,z_obs,np.transpose(windm_obs_15_full), \
               vmin=0,vmax=10,cmap="hot")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_15[0:-1:2],'bs', \
             MarkerSize=5,MarkerFaceColor="None", \
             label="gradient method")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_15[0:-1:2],'bD', \
             MarkerSize=5,MarkerFaceColor="None", \
             label="threshold method")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 15 September")
    cb1 = plt.colorbar()
    cb1.set_label("wind speed / m/s")
    plt.legend(loc="upper left")
    
    plt.subplot(3,2,4)
    plt.pcolor(time_500_mean,z_500,np.transpose(windm_500_15_mean_full), \
               vmin=0,vmax=10,cmap="hot")
    plt.plot(time_BL_500,BLthick_500_15,'bo',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("wind speed / m/s")

    plt.subplot(3,2,6)
    plt.pcolor(time_100_mean,z_100,np.transpose(windm_100_15_mean_full), \
               vmin=0,vmax=10,cmap="hot")
    plt.plot(time_BL_100,BLthick_100_15,'bo',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 15 Sep / h")
 #   plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("wind speed / m/s")

    plt.suptitle("FIGURE A1. Time--height sections at LSBU, with model data averaged "\
                 +"into 30-minute intervals. From MAKEPLOT_U19_COVARIANCES")


    # FIGURE A2. Wind direction time--height plot.
    plt.figure(figsize=(12,10))

    plt.subplot(3,2,1)
    plt.pcolor(timehr_obs,z_obs,np.transpose(winddir_obs_17_full), \
               vmin=0,vmax=360,cmap="hsv")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_17[0:-1:2],'ks', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_17[0:-1:2],'kD', \
             MarkerSize=5,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 17 September")
    cb1 = plt.colorbar()
#    cb1.set_label("wind direction / \circ")
    
    plt.subplot(3,2,3)
    plt.pcolor(time_500_mean,z_500,np.transpose(winddir_500_17_mean_full), \
               vmin=0,vmax=360,cmap="hsv")
    plt.plot(time_BL_500,BLthick_500_17,'ko',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("wind direction / \circ")

    plt.subplot(3,2,5)
    plt.pcolor(time_100_mean,z_100,np.transpose(winddir_100_17_mean_full), \
               vmin=0,vmax=360,cmap="hsv")
    plt.plot(time_BL_100,BLthick_100_17,'ko', \
             MarkerSize=6,MarkerFaceColor="None",label="parcel method")
    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("wind direction / \circ")
    plt.legend(loc="upper left")
 
    plt.subplot(3,2,2)
    plt.pcolor(timehr_obs,z_obs,np.transpose(winddir_obs_15_full), \
               vmin=0,vmax=360,cmap="hsv")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_15[0:-1:2],'ks', \
             MarkerSize=5,MarkerFaceColor="None", \
             label="gradient method")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_15[0:-1:2],'kD', \
             MarkerSize=5,MarkerFaceColor="None", \
             label="threshold method")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 15 September")
    cb1 = plt.colorbar()
    cb1.set_label("wind direction / \circ")
    plt.legend(loc="upper left")
    
    plt.subplot(3,2,4)
    plt.pcolor(time_500_mean,z_500,np.transpose(winddir_500_15_mean_full), \
               vmin=0,vmax=360,cmap="hsv")
    plt.plot(time_BL_500,BLthick_500_15,'ko',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("wind direction / \circ")

    plt.subplot(3,2,6)
    plt.pcolor(time_100_mean,z_100,np.transpose(winddir_100_15_mean_full), \
               vmin=0,vmax=360,cmap="hsv")
    plt.plot(time_BL_100,BLthick_100_15,'ko',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 15 Sep / h")
 #   plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("wind direction / \circ")

    plt.suptitle("FIGURE A2. Time--height sections at LSBU, with model data averaged "\
                 +"into 30-minute intervals. From MAKEPLOT_U19_COVARIANCES")


#**** FIGURE A3. Zonal wind component time--height plot.
    plt.figure(figsize=(12,10))

    plt.subplot(3,2,1)
    plt.pcolor(timehr_obs,z_obs,np.transpose(windx_obs_17_full), \
               vmin=-8,vmax=8,cmap="PiYG")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_17[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_17[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 17 September")
    cb1 = plt.colorbar()
#    cb1.set_label("zonal wind / m/s")
    
    plt.subplot(3,2,3)
    plt.pcolor(time_500_mean,z_500,np.transpose(windx_500_17_mean_full), \
               vmin=-8,vmax=8,cmap="PiYG")
    plt.plot(time_BL_500,BLthick_500_17,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("zonal wind / m/s")

    plt.subplot(3,2,5)
    plt.pcolor(time_100_mean,z_100,np.transpose(windx_100_17_mean_full), \
               vmin=-8,vmax=8,cmap="PiYG")
    plt.plot(time_BL_100,BLthick_100_17,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("zonal wind / m/s")
 
    plt.subplot(3,2,2)
    plt.pcolor(timehr_obs,z_obs,np.transpose(windx_obs_15_full), \
               vmin=-8,vmax=8,cmap="PiYG")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_15[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_15[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 15 September")
    cb1 = plt.colorbar()
    cb1.set_label("zonal wind / m/s")
    
    plt.subplot(3,2,4)
    plt.pcolor(time_500_mean,z_500,np.transpose(windx_500_15_mean_full), \
               vmin=-8,vmax=8,cmap="PiYG")
    plt.plot(time_BL_500,BLthick_500_15,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("zonal wind / m/s")

    plt.subplot(3,2,6)
    plt.pcolor(time_100_mean,z_100,np.transpose(windx_100_15_mean_full), \
               vmin=-8,vmax=8,cmap="PiYG")
    plt.plot(time_BL_100,BLthick_100_15,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 15 Sep / h")
 #   plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("zonal wind / m/s")
 
    plt.suptitle("FIGURE A3. Time--height sections at LSBU, with model data averaged "\
                 +"into 30-minute intervals. From MAKEPLOT_U19_COVARIANCES")

#**** FIGURE A4. Meridional wind component time--height plot.
    plt.figure(figsize=(12,10))

    plt.subplot(3,2,1)
    plt.pcolor(timehr_obs,z_obs,np.transpose(windy_obs_17_full), \
               vmin=-8,vmax=8,cmap="PuOr")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_17[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_17[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 17 September")
    cb1 = plt.colorbar()
#    cb1.set_label("meridional wind / m/s")
    
    plt.subplot(3,2,3)
    plt.pcolor(time_500_mean,z_500,np.transpose(windy_500_17_mean_full), \
               vmin=-8,vmax=8,cmap="PuOr")
    plt.plot(time_BL_500,BLthick_500_17,'ro',MarkerSize=6,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("meridional wind / m/s")

    plt.subplot(3,2,5)
    plt.pcolor(time_100_mean,z_100,np.transpose(windy_100_17_mean_full), \
               vmin=-8,vmax=8,cmap="PuOr")
    plt.plot(time_BL_100,BLthick_100_17,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("meridional wind / m/s")
 
    plt.subplot(3,2,2)
    plt.pcolor(timehr_obs,z_obs,np.transpose(windy_obs_15_full), \
               vmin=-8,vmax=8,cmap="PuOr")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_15[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_15[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 15 September")
    cb1 = plt.colorbar()
    cb1.set_label("meridional wind / m/s")
    
    plt.subplot(3,2,4)
    plt.pcolor(time_500_mean,z_500,np.transpose(windy_500_15_mean_full), \
               vmin=-8,vmax=8,cmap="PuOr")
    plt.plot(time_BL_500,BLthick_500_15,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("meridional wind / m/s")

    plt.subplot(3,2,6)
    plt.pcolor(time_100_mean,z_100,np.transpose(windy_100_15_mean_full), \
               vmin=-8,vmax=8,cmap="PuOr")
    plt.plot(time_BL_100,BLthick_100_15,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("meridional wind / m/s")

    plt.suptitle("FIGURE A4. Time--height sections at LSBU, with model data averaged "\
                 +"into 30-minute intervals. From MAKEPLOT_U19_COVARIANCES")


#**** FIGURE A5. Vertical wind component time--height plot.
    plt.figure(figsize=(12,10))

    plt.subplot(3,2,1)
    plt.pcolor(timehr_obs,z_obs_vert,np.transpose(windz_obs_17_full), \
               vmin=-1,vmax=1,cmap="RdGy")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_17[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_17[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 17 September")
    cb1 = plt.colorbar()
#    cb1.set_label("vertical wind / m/s")
    
    plt.subplot(3,2,3)
    plt.pcolor(time_500_mean,z_500,np.transpose(windz_500_17_mean_full), \
               vmin=-1,vmax=1,cmap="RdGy")
    plt.plot(time_BL_500,BLthick_500_17,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("vertical wind / m/s")

    plt.subplot(3,2,5)
    plt.pcolor(time_100_mean,z_100,np.transpose(windz_100_17_mean_full), \
               vmin=-1,vmax=1,cmap="RdGy")
    plt.plot(time_BL_100,BLthick_100_17,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("vertical wind / m/s")
 
    plt.subplot(3,2,2)
    plt.pcolor(timehr_obs,z_obs_vert,np.transpose(windz_obs_15_full), \
               vmin=-1,vmax=1,cmap="RdGy")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_15[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_15[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 15 September")
    cb1 = plt.colorbar()
    cb1.set_label("vertical wind / m/s ")
    
    plt.subplot(3,2,4)
    plt.pcolor(time_500_mean,z_500,np.transpose(windz_500_15_mean_full), \
               vmin=-1,vmax=1,cmap="RdGy")
    plt.plot(time_BL_500,BLthick_500_15,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("vertical wind / m/s")

    plt.subplot(3,2,6)
    plt.pcolor(time_100_mean,z_100,np.transpose(windz_100_15_mean_full), \
               vmin=-1,vmax=1,cmap="RdGy")
    plt.plot(time_BL_100,BLthick_100_15,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("vertical wind / m/s")

    plt.suptitle("FIGURE A5. Time--height sections at LSBU, with model data averaged "\
                 +"into 30-minute intervals. From MAKEPLOT_U19_COVARIANCES")


#**** FIGURE A6. Variance in vertical wind component, time--height plot 
#          -- an indicator of turbulence.
    plt.figure(figsize=(12,10))

    plt.subplot(3,2,1)
    plt.pcolor(timehr_obs,z_obs_vert,np.transpose(windzvar_obs_17_full), \
               vmin=0,vmax=2,cmap="cool")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_17[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_17[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
 #   plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 17 September")
    cb1 = plt.colorbar()
#    cb2.set_label("vertical wind variance / m$^2$/s$^2$")
    
    plt.subplot(3,2,3)
    plt.pcolor(time_500_mean,z_500,np.transpose(windz_500_17_var_full), \
               vmin=0,vmax=2,cmap="cool")
    plt.plot(time_BL_500,BLthick_500_17,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("vertical wind variance / m$^2$/s$^2$")

    plt.subplot(3,2,5)
    plt.pcolor(time_100_mean,z_100,np.transpose(windz_100_17_var_full), \
               vmin=0,vmax=2,cmap="cool")
    plt.plot(time_BL_100,BLthick_100_17,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 17 Sep / h")
    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bv624; 17 September")
    cb2 = plt.colorbar()
#    cb2.set_label("vertical wind variance / $m^2/s^2$")
 
    plt.subplot(3,2,2)
    plt.pcolor(timehr_obs,z_obs_vert,np.transpose(windzvar_obs_15_full), \
               vmin=0,vmax=1,cmap="cool")
    plt.plot(timehr_obs[0:-1:2],MLdepthsgr_obs_15[0:-1:2],'rs', \
             MarkerSize=5,MarkerFaceColor="None")
    plt.plot(timehr_obs[0:-1:2],MLdepthsthr_obs_15[0:-1:2],'rD', \
             MarkerSize=5,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Observations: MAGIC project; 15 September")
    cb1 = plt.colorbar()
    cb1.set_label("vertical wind variance / m$^2$/s$^2$")
    
    plt.subplot(3,2,4)
    plt.pcolor(time_500_mean,z_500,np.transpose(windz_500_15_var_full), \
               vmin=0,vmax=1,cmap="cool")
    plt.plot(time_BL_500,BLthick_500_15,'ro',MarkerSize=6,MarkerFaceColor="None")
#    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.xticks(np.arange(-5.0,20.0,5.0),[])
    plt.title("Model output (500 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("vertical wind variance / m$^2$/s$^2$")

    plt.subplot(3,2,6)
    plt.pcolor(time_100_mean,z_100,np.transpose(windz_100_15_var_full), \
               vmin=0,vmax=1,cmap="cool")
    plt.plot(time_BL_100,BLthick_100_15,'ro',MarkerSize=6,MarkerFaceColor="None")
    plt.xlabel("Time after 00:00 on 15 Sep / h")
#    plt.ylabel("Height / m")
    plt.xlim(-6.0,18.5)
    plt.ylim(0,2000)
    plt.title("Model output (100 m): suite u-bx043; 15 September")
    cb2 = plt.colorbar()
    cb2.set_label("vertical wind variance / m$^2$/s$^2$")

    plt.suptitle("FIGURE A6. Time--height sections at LSBU, with model data averaged "\
                 +"into 30-minute intervals. From MAKEPLOT_U19_COVARIANCES")


# FIGURE SET 4: model vs observed wind magnitude profiles
if 4 in makefig:
    print("<MAKEPLOT_U19_COVARIANCES> Making Figure 4...")

    avprd = 2.0
    hpts = [4, 6, 8, 10, 12, 14, 16, 18]
    pts_obs = np.zeros(np.array(hpts).shape)
    pts_500 = np.zeros(np.array(hpts).shape)
    pts_100 = np.zeros(np.array(hpts).shape)
    pts_BL_500 = np.zeros(np.array(hpts).shape)
    pts_BL_100 = np.zeros(np.array(hpts).shape)
    for ipt in np.arange(len(hpts)):
        pts_obs[ipt] = int(np.where(timehr_obs<=hpts[ipt])[0][-1])
        pts_500[ipt] = int(np.where(time_500_mean<=hpts[ipt])[0][-1])
        pts_100[ipt] = int(np.where(time_100_mean<=hpts[ipt])[0][-1])
        pts_BL_500[ipt] = int(np.where(time_BL_500>=hpts[ipt])[0][0])
        pts_BL_100[ipt] = int(np.where(time_BL_100>=hpts[ipt])[0][0])
        print('500 m: '+str(ipt)+': time = '+str(pts_BL_500[ipt]))
        print('100 m: '+str(ipt)+': time = '+str(pts_BL_100[ipt]))

    # Wind speed profiles: 15 Sep
    plt.figure(figsize=(12,10))
    for ipt in np.arange(len(hpts)):
        plt.subplot(2,4,ipt+1)
        plt.plot(np.mean(windm_obs_15_full[int(pts_obs[ipt]-avprd):  \
                                           int(pts_obs[ipt]+avprd+1),:]   \
                                           ,axis=0), \
                 z_obs,'r-',label="obs")
        plt.plot(windm_500_15_mean_full[int(pts_500[ipt]),:], \
                 z_500,'g-',label="500 m")
        plt.plot(windm_100_15_mean_full[int(pts_100[ipt]),:], \
                 z_100,'b-',label="100 m")
        plt.plot((0,20),(BLthick_500_15[int(pts_BL_500[ipt])], \
                         BLthick_500_15[int(pts_BL_500[ipt])]),'g--')
        plt.plot((0,20),(BLthick_100_15[int(pts_BL_100[ipt])], \
                         BLthick_100_15[int(pts_BL_100[ipt])]),'b--')
        plt.ylim(0,1600)
        plt.xlim(0,20)
        plt.grid(True)
        plt.title("Hour: "+str(hpts[ipt]))
        if ipt in (0, 4):
            plt.ylabel("height / m")
        if ipt in (4, 5, 6, 7):
            plt.xlabel("wind speed / m/s")
        if ipt==7:
            plt.legend()
            plt.suptitle("Wind magnitude profiles on 15 September "+ \
                         "(northerly), averaged over "+str(avprd)+" hours. "+ \
                         "Boundary layer depth indicated as dashed lines")

    # Wind speed profiles: 17 Sep
    plt.figure(figsize=(12,10))
    for ipt in np.arange(len(hpts)):
        plt.subplot(2,4,ipt+1)
        plt.plot(np.mean(windm_obs_17_full[int(pts_obs[ipt]-avprd):  \
                                           int(pts_obs[ipt]+avprd+1),:]   \
                                           ,axis=0), \
                 z_obs,'r-',label="obs")
        plt.plot(windm_500_17_mean_full[int(pts_500[ipt]),:], \
                 z_500,'g-',label="500 m")
        plt.plot(windm_100_17_mean_full[int(pts_100[ipt]),:], \
                 z_500,'b-',label="100 m")
        plt.plot((0,20),(BLthick_500_17[int(pts_BL_500[ipt])], \
                         BLthick_500_17[int(pts_BL_500[ipt])]),'g--')
        plt.plot((0,20),(BLthick_100_17[int(pts_BL_100[ipt])], \
                         BLthick_100_17[int(pts_BL_100[ipt])]),'b--')
        plt.ylim(0,1600)
        plt.xlim(0,20)
        plt.grid(True)
        plt.title("Hour: "+str(hpts[ipt]))
        if ipt in (0, 4):
            plt.ylabel("height / m")
        if ipt in (4, 5, 6, 7):
            plt.xlabel("wind speed / m/s")
        if ipt==7:
            plt.legend()
            plt.suptitle("Wind magnitude profiles on 17 September "+ \
                         "(northerly), averaged over "+str(avprd)+" hours. "+ \
                         "Boundary layer depth indicated as dashed lines")

    # Potential temperature profiles: both days
    plt.figure(figsize=(12,10))
    colstrs = ('navy','blue','green','yellow',  \
               'orange','red','pink','purple')

    plt.subplot(1,2,1)
    for ipt in np.arange(len(hpts)):
        plt.plot(pottemp_500_15_mean_full[int(pts_500[ipt]),:], \
                 z_500,color=colstrs[ipt],linestyle="--", \
                 label="Hour: "+str(hpts[ipt])+", 500 m")
        plt.plot(281+5*ipt/len(hpts), \
                 BLthick_500_15[int(pts_BL_500[ipt])],"o", \
                  color=colstrs[ipt])
        if ipt > 0:
            plt.plot(pottemp_100_15_mean_full[int(pts_100[ipt]),:], \
                     z_100,color=colstrs[ipt],linestyle="-", \
                     label="Hour: "+str(hpts[ipt])+", 100 m")
            plt.plot(281+5*ipt/len(hpts)+0.5, \
                     BLthick_100_15[int(pts_BL_100[ipt])],"s", \
                     color=colstrs[ipt])
    plt.ylim(0,1500)
    plt.xlim(280,300)
    plt.grid(True)
    plt.title("15 September (westerly)")
    plt.ylabel("height / m")
    plt.xlabel("potential temperature / K")

    plt.subplot(1,2,2)
    for ipt in np.arange(len(hpts)):
        plt.plot(pottemp_500_17_mean_full[int(pts_500[ipt]),:], \
                 z_500,color=colstrs[ipt],linestyle="--", \
                 label="Hour: "+str(hpts[ipt])+", 500 m")
        plt.plot(281+5*ipt/len(hpts), \
                 BLthick_500_17[int(pts_BL_500[ipt])],"o", \
                  color=colstrs[ipt])
        if ipt > 0:
            plt.plot(pottemp_100_17_mean_full[int(pts_100[ipt]),:], \
                     z_100,color=colstrs[ipt],linestyle="-", \
                     label="Hour: "+str(hpts[ipt])+", 100 m")
            plt.plot(281+5*ipt/len(hpts)+0.5, \
                     BLthick_100_17[int(pts_BL_100[ipt])],"s", \
                     color=colstrs[ipt])
    plt.ylim(0,1500)
    plt.xlim(280,300)
    plt.grid(True)
    plt.title("17 September (northerly)")
    plt.xlabel("potential temperature / K")
    plt.legend(loc="lower right")

    plt.suptitle("Potential temperature profiles, averaged "+ \
                 "over 30 minutes. Boundary layer depths indicated -- "+ \
                 "500 m (circles), 100 m (squares)")
