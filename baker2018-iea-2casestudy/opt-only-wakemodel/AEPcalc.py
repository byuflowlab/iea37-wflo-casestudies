import numpy as np
import sys
import csv
# Written by PJ Stanley, Jared Thomas, and Nicholas Baker
# BYU FLOW lab
# Completed 10 June 2018
# Updated 18 Jun 2018 to include read in of .csv turbine locations

def WindFrame(turbineX, turbineY, windDirectionDeg):
    """ Calculates the locations of each turbine in the wind direction reference frame """
    # nTurbines = len(turbineX)

    # convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=E)
    windDirectionDeg = 270. - windDirectionDeg
    if windDirectionDeg < 0.:
        windDirectionDeg += 360.
    windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

    # convert to downwind(x)-crosswind(y) coordinates
    turbineXw = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
    turbineYw = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

    return turbineXw, turbineYw #, dturbineXw_dturbineX, dturbineXw_dturbineY, dturbineYw_dturbineX, dturbineYw_dturbineY


def SortTurbs(turbineXw, turbineYw):
    """ Sorts the turbines in the wind frame from left to right """
    ind = np.argsort(turbineXw)
    nTurbines = len(turbineXw)

    turbineXsort = np.zeros(nTurbines)
    turbineYsort = np.zeros(nTurbines)
    for i in range(nTurbines):
        turbineXsort[i] = turbineXw[ind[i]]
        turbineYsort[i] = turbineYw[ind[i]]

    return turbineXsort, turbineYsort


def GaussianWake(turbineXsort, turbineYsort):
    """ Returns each turbines total loss from wakes """
    nTurbines = len(turbineXsort)

    CT = 4.0*1./3.*(1.0-1./3.)
    k = 0.0324555

    D = 126.4
    x0 = 0.

    loss = np.zeros(nTurbines)
    for i in range(nTurbines):
        loss_array = np.zeros(i)
        for j in range(i):
            x = turbineXsort[i]-turbineXsort[j]
            y = turbineYsort[i]-turbineYsort[j]
            sigma = k*(x-x0)+D/np.sqrt(8.)

            loss_array[j] = (1.-np.sqrt(1.-CT/(8.*sigma**2/D**2)))*np.exp(-0.5*(y/sigma)**2)
        loss[i] = np.sqrt(np.sum(loss_array**2))

    return loss


def DirPower(turbineX,turbineY,windDirectionDeg,wind_speed):
    """ Returns the power produced by each turbine for a given wind speed and direction """
    nTurbines = len(turbineX)

    turbineXw, turbineYw = WindFrame(turbineX,turbineY,windDirectionDeg)
    turbineXsort,turbineYsort = SortTurbs(turbineXw,turbineYw)
    loss = GaussianWake(turbineXsort, turbineYsort)
    effective_wind_speed = wind_speed*(1.-loss)

    cut_in_wind_speed = 3.
    cut_out_wind_speed = 25.
    rated_wind_speed = 11.4
    rated_power = 5.E6

    turbine_power = np.zeros(nTurbines)

    for i in range(nTurbines):
        if effective_wind_speed[i] <= cut_in_wind_speed:
            turbine_power[i] = 0.
        elif cut_in_wind_speed < effective_wind_speed[i] < cut_out_wind_speed:
            turbine_power[i] = rated_power*((effective_wind_speed[i]-cut_in_wind_speed)/(rated_wind_speed-cut_in_wind_speed))**3
        else:
            turbine_power[i] = rated_power

    dir_power = np.sum(turbine_power)

    return dir_power


def WindRose(wind_direction):
    sigma1 = 20.
    mu1 = 180.
    p1 = np.sqrt(1./(2.*np.pi*sigma1**2))
    exponent1 = -(wind_direction-mu1)**2/(2.*sigma1**2)

    sigma2 = 40.
    mu2 = 350.
    mu3 = -10.
    p2 = np.sqrt(1./(2.*np.pi*sigma2**2))
    exponent2 = -(wind_direction-mu2)**2/(2.*sigma2**2)
    exponent3 = -(wind_direction-mu3)**2/(2.*sigma2**2)

    w1 = 0.5
    w2 = 0.5

    return w1*p1*np.exp(exponent1) + w2*p2*np.exp(exponent2) + w2*p2*np.exp(exponent3)


def SampleWindFreq(nDirections):
    dTheta = 360./float(nDirections)
    dirs = np.linspace(dTheta/2.,360.-dTheta/2.,nDirections)
    freqs = np.zeros(nDirections)
    nIntegrate = 100
    for i in range(nDirections):
        intArray = np.zeros(nIntegrate)
        directions = np.linspace(dirs[i]-dTheta/2.,dirs[i]+dTheta/2.,nIntegrate)
        freqs[i] = np.trapz(WindRose(directions),directions)

    print repr(freqs)
    print sum(freqs)


def calcAEP(turbineX, turbineY):
    """calculate the wind farm AEP"""
    nDirections = 16                    # Chosen to match Combined Case Study numbers
    dTheta = 360./float(nDirections)
    windDirections = np.linspace(dTheta/2.,360.-dTheta/2.,nDirections)
    # Bi-directional wind rose to simulate canyon
    #windFrequencies = np.array([0.03384792,  0.03133345,  0.02808351,  0.02437032,  0.02047566,
    #                            0.0166564,  0.01311872,  0.01000387,  0.00738604,  0.00527987,
    #                            0.00365435,  0.00244933,  0.00159192,  0.00101283,  0.00066779,
    #                            0.00057669,  0.00089915,  0.00205219,  0.00481945,  0.01030317,
    #                            0.01951741,  0.03257653,  0.04784983,  0.06183345,  0.07029184,
    #                            0.07029485,  0.06184392,  0.0478725,  0.03262081,  0.01959987,
    #                            0.0104511,  0.00507567,  0.00248079,  0.00159146,  0.00165606,
    #                            0.00229131,  0.00336739,  0.00488204,  0.00687487,  0.00937791,
    #                            0.01238636,  0.01583988,  0.01961223,  0.02351087,  0.02728836,
    #                            0.03066569,  0.0333653,  0.03514828,  0.03584923,  0.03540157])
    # LaGuardia, Tri-directional
    windFrequencies = np.array([.055,  .035, .121,  .081,
                                .029,  .022, .027,  .036,
                                .118,  .050, .054,  .033,
                                .072,  .082, .119,  .066])
    # Denver, Quad-directional
    #windFrequencies = np.array([.107,  .045,  .055,  .051,
    #                            .108,  .039,  .041,  .061,
    #                            .100,  .053,  .057,  .057,
    #                            .109,  .042,  .032,  .043])
    # Santa Barbara, Bi-Directional
    #windFrequencies = np.array([.025,  .024,  .029,  .036,
    #                            .063,  .065,  .100,  .122,
    #                            .063,  .038,  .039,  .083,
    #                            .213,  .046,  .032,  .022])
    wind_speed = 13.            # Constant throughout farm

    powers = np.zeros(nDirections)
    for i in range(nDirections):
        powers[i] = DirPower(turbineX,turbineY,windDirections[i],wind_speed)

    hours_per_year = 365.*24.
    AEP = hours_per_year*np.sum(windFrequencies*powers)
    AEP /= 1.E6 #convert to MWh

    return AEP


def getTurbLoc(sFileName):

    turbineX = np.array([])
    turbineY = np.array([])

    csvFile = csv.reader(open(sFileName, "rb"))
    for row in csvFile:
            turbineX = np.append(turbineX, float(row[0]))
            turbineY = np.append(turbineY, float(row[1]))

    return turbineX, turbineY

if __name__=="__main__":
    """for testing during development"""
    turbineX = np.array([])
    turbineY = np.array([])

    # Reads turbine locations from a .csv file
    # example command line syntax is "Python AEPcal.py BaseCoords16.csv"
    turbineX, turbineY = getTurbLoc(sys.argv[1])
    AEP = calcAEP(turbineX,turbineY)