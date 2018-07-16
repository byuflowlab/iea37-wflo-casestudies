import numpy as np
import sys
import yaml
# Written by PJ Stanley, Jared Thomas, and Nicholas F. Baker
# BYU FLOW lab
# Completed 10 June 2018
# Updated 11 Jul 2018 to include read-in of .yaml turb locs and wind freq dist, for release

def WindFrame(turbineX, turbineY, windDirectionDeg):
    """ Calculates the locations of each turbine in the wind direction reference frame """

    # convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=E)
    windDirectionDeg = 270. - windDirectionDeg
    if windDirectionDeg < 0.:
        windDirectionDeg += 360.
    windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

    # convert to downwind(x)-crosswind(y) coordinates
    turbineXw = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
    turbineYw = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

    return turbineXw, turbineYw


def GaussianWake(turbineXw, turbineYw):
    """ Returns each turbines total loss from wakes """
    # Equations and values explained in <iea37-wakemodel.pdf>
    nTurbines = len(turbineXw)

    CT = 4.0*1./3.*(1.0-1./3.)  # constant thrust coefficient
    k = 0.0324555   # constant turbulence

    D = 130.  # IEA37 3.35MW onshore reference turbine rotor diameter

    loss = np.zeros(nTurbines)

    for i in range(nTurbines):
        loss_array = np.zeros(nTurbines) #  loss contribution from each turbine
        for j in range(nTurbines):
            x = turbineXw[i]-turbineXw[j]
            y = turbineYw[i]-turbineYw[j]
            if x > 0.:  #  Simplified Bastankhah Gaussian wake model, applied to downstream turbines
                sigma = k*(x)+D/np.sqrt(8.)
                loss_array[j] = (1.-np.sqrt(1.-CT/(8.*sigma**2/D**2)))*np.exp(-0.5*(y/sigma)**2)
            else:
                loss_array[j] = 0.
        loss[i] = np.sqrt(np.sum(loss_array**2))  #  total wake loss, sqrt of sum of sqrs

    return loss


def DirPower(turbineX,turbineY,windDirectionDeg,wind_speed):
    """ Returns the power produced by each turbine for a given wind speed and direction """
    nTurbines = len(turbineX)

    turbineXw, turbineYw = WindFrame(turbineX,turbineY,windDirectionDeg)  #  turbines in wind frame coordinates
    loss = GaussianWake(turbineXw, turbineYw)  #  wake losses

    effective_wind_speed = wind_speed*(1.-loss)

    #  turbine parameters
    cut_in_wind_speed = 4.
    cut_out_wind_speed = 25.
    rated_wind_speed = 9.8
    rated_power = 3.35E6

    turbine_power = np.zeros(nTurbines)

    #  power from each wind turbines
    for i in range(nTurbines):
        if effective_wind_speed[i] <= cut_in_wind_speed:
            turbine_power[i] = 0.
        elif cut_in_wind_speed < effective_wind_speed[i] < cut_out_wind_speed:
            turbine_power[i] = rated_power*((effective_wind_speed[i]-cut_in_wind_speed)/(rated_wind_speed-cut_in_wind_speed))**3
        else:
            turbine_power[i] = rated_power

    dir_power = np.sum(turbine_power) #  total directional power

    return dir_power


def calcAEP(turbineX, turbineY, windFrequencies):
    """ Calculate the wind farm AEP """
    nDirections = len(windFrequencies)  # Windrose number of bins
    dTheta = 360./float(nDirections)
    windDirections = np.linspace(0.,360.-dTheta,nDirections)
    wind_speed = 9.8  # Constant throughout farm, based on rated wind speed of IEA37 3.35MW turbine

    #  wind farm power from each wind direction
    powers = np.zeros(nDirections)
    for i in range(nDirections):
        powers[i] = DirPower(turbineX,turbineY,windDirections[i],wind_speed)

    #  convert power to AEP
    hours_per_year = 365.*24.
    AEP = hours_per_year*np.sum(windFrequencies*powers)
    AEP /= 1.E6 #convert to MWh

    return AEP


def getTurbLocYAML(sFileName):
    turbineX = np.array([])
    turbineY = np.array([])

    # Read in the .yaml file
    with open(sFileName, 'r') as f:
        doc = yaml.load(f)

    # rip the x- and y-coordinates
    turbineX = np.asarray(doc['definitions']['position']['items']['xc']) # Convert from <list> to <ndarray>
    turbineY = np.asarray(doc['definitions']['position']['items']['yc'])
    # rip the expected AEP, used for comparison
    # AEP = doc['definitions']['plant_energy']['properties']['annual_energy_production']['default']

    return turbineX, turbineY#, AEP


def getWindFreqYAML(sFileName):
    windFreq = np.array([])

    # Read in the .yaml file
    with open(sFileName, 'r') as f:
        doc = yaml.load(f)

    # rip wind frequency distribution array
    windFreq = np.asarray(doc['definitions']['wind']['properties']['probability']['default']) # Convert from <list> to <ndarray>

    return windFreq

if __name__ == "__main__":
    """ Used for demonstration """
    turbineX = np.array([])
    turbineY = np.array([])

    # Reads turbine locations and wind freq distribution from .yaml files
    # For Python .yaml capability, in the terminal type "pip install pyyaml".
    # An Example command line syntax to run this file is "python iea37-aepcal.py iea37-ex16.yaml iea37-windrose.yaml"
    turbineX, turbineY = getTurbLocYAML(sys.argv[1])    # Get turbine locations from .yaml file
    windFreq = getWindFreqYAML(sys.argv[2])             # Get wind frequency distribution from .yaml file
    AEP = calcAEP(turbineX, turbineY, windFreq)         # Calculate the AEP from ripped values
    print AEP                                           # Print calculated AEP