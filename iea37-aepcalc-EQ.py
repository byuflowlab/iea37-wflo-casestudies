"""IEA Task 37 Combined Case Study AEP Calculation Code

Written by Nicholas F. Baker, PJ Stanley, and Jared Thomas
BYU FLOW lab
Created 10 June 2018
Updated 11 Jul 2018 to include read-in of .yaml turb locs and wind freq dist.
Completed 26 Jul 2018 for commenting and release

Changes by Erik Quaeghebeur [20-21 August 2018]:
- 2 blank lines around function definitions
- Too long lines cut to < 80 chars
- Whitespace cleanup
- Use print function for compatibility with Python 3
- Remove unused variable 'sWindRose'
- Add some missing minimal docstrings
- Renamed iea37-335mw.yaml to iea37-335MW.yaml for things to work
- Remove useless "If it's left of North ..."
- Correct comment about "so "North" aligns with -x direction" (DirPower)
- Remove strange, incorrect "not in this directory"
- DRY: avoid calculating the same quantity multiple times
  (cos_min_wind_dir, sin_min_wind_dir)
- Don't calculate already established 0 values (GaussianWake, DirPower)
- Avoid using for loops when the loop contents do not depend on previous
  iterations; use vectorized code instead (GaussianWake, DirPower, calcAEP).
- Remove some useless empty array initializations
  (getTurbLocYAML, getWindRoseYAML, __main__)
- Use yaml.safe_load.
- Retain just the needed parts of the loaded yaml docs.
- for loops can loop over arbitrary values, don't construct unnecessary indices
  (getTurbLocYAML)
- Use generator expression if only initial successful iteration is needed
  (getTurbLocYAML)
- Use structured datatype (coordinate) and recarray to keep x and y coordinates
  together
- TODO: use Python variable and function naming conventions
  https://www.python.org/dev/peps/pep-0008/#prescriptive-naming-conventions
  (so *not* CamelCase for function and variable names)
"""

from __future__ import print_function
import numpy as np
import sys
import yaml  # For reading .yaml files
from math import radians as DegToRad  # For converting from degrees to radians


# structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


def WindFrame(coords, windDirDeg):
    """Convert map coordinates to downwind/crosswind coordinates"""

    # Convert from meteorological polar system (CW, 0 deg.=N)
    # to standard polar system (CCW, 0 deg.=W)
    # Shift so North comes "along" x-axis, from left to right.
    windDirDeg = 270. - windDirDeg
    # Convert inflow wind direction from degrees to radians
    windDirRad = DegToRad(windDirDeg)

    # Convert to downwind(x) & crosswind(y) coordinates
    cos_min_wind_dir = np.cos(-windDirRad)
    sin_min_wind_dir = np.sin(-windDirRad)
    frame_coords = np.recarray(coords.shape, coordinate)
    frame_coords.x = coords.x * cos_min_wind_dir - coords.y * sin_min_wind_dir
    frame_coords.y = coords.x * sin_min_wind_dir + coords.y * cos_min_wind_dir

    return frame_coords


def GaussianWake(frame_coords):
    """Returns each turbine's total loss from wakes"""
    # Equations and values explained in <iea37-wakemodel.pdf>

    CT = 4. * 1. / 3. * (1. - 1. / 3.)  # Constant thrust coefficient
    k = 0.0324555  # Constant, relating to a turbulence intensity of 0.075

    D = 130.  # IEA37 3.35MW onshore reference turbine rotor diameter

    # Calculate matrices of pairwise downwind and crosswind distances
    frame_coords_matrix = np.tile(frame_coords, (len(frame_coords), 1))
    dist = np.recarray(frame_coords_matrix.shape, coordinate)
    dist.x = frame_coords_matrix.x - frame_coords_matrix.x.T
    dist.y = frame_coords_matrix.y - frame_coords_matrix.y.T

    # If the turbine of interest is downwind of the turbine generating the
    # wake, there is a wake loss; calculate it using the Simplified Bastankhah
    # Gaussian wake model
    downwind = dist.x > 0
    sigma = k * dist.x[downwind] + D / np.sqrt(8.)
    losses = np.zeros(dist.shape)
    losses[downwind] = ((1. - np.sqrt(1. - CT / (8. * sigma ** 2 / D ** 2)))
                        * np.exp(-0.5 * (dist.y[downwind] / sigma) ** 2))

    # Array holding the wake deficit seen at each turbine
    loss = np.sqrt(np.sum(losses ** 2, axis=0))

    return loss


def DirPower(coords, windDirDeg, windSpeed,
             turbCI, turbCO, turbRtdWS, turbRtdPwr):
    """Return the power produced by each turbine"""

    # Shift coordinate frame of reference to downwind/crosswind
    frame_coords = WindFrame(coords, windDirDeg)
    # Use the Simplified Bastankhah Gaussian wake model
    # to calculate wake deficits
    loss = GaussianWake(frame_coords)

    # Effective windspeed is freestream multiplied by
    # the calculated wake deficits
    windSpeedEff = windSpeed * (1. - loss)

    # Calculate the power from each turbine
    # based on experienced wind speed & power curve
    # 1. By default, power output is zero
    pwrTurb = np.zeros(frame_coords.shape)
    # 2. Determine which effectve wind speeds are between cut-in and cut-out
    #    or on the curve
    between_cut_speeds = np.logical_and(turbCI <= windSpeedEff,
                                        windSpeedEff < turbCO)
    below_rated = windSpeedEff < turbRtdWS
    on_curve = np.logical_and(between_cut_speeds, below_rated)
    # 3. Between cut-in and cut-out, power is a fraction of rated
    pwrTurb[between_cut_speeds] = turbRtdPwr
    # 4. Only below rated (on curve) not at 100%, but based on curve
    pwrTurb[on_curve] *= ((windSpeedEff[on_curve] - turbCI)
                          / (turbRtdWS - turbCI)) ** 3

    # Sum the power from all turbines for this direction
    pwrDir = np.sum(pwrTurb)

    return pwrDir


def calcAEP(coords, windFreq, windSpeed, windDir,
            turbCutInWS, turbCutOutWS, turbRtdWS, turbRtdPwr):
    """Calculate the wind farm AEP"""
    def specDirPower(d):
        return DirPower(coords, d, windSpeed,
                        turbCutInWS, turbCutOutWS, turbRtdWS, turbRtdPwr)
    # Power produced by the wind farm for each wind direction
    PwrProduced = np.fromiter((specDirPower(d) for d in windDir), np.double)

    #  Convert power to AEP
    hrsPerYr = 365. * 24.  # Const Value for hours in a year
    AEP = hrsPerYr * (windFreq * PwrProduced)  # AEP for each binned direction
    AEP /= 1.E6  # Convert to MWh

    return AEP


def getTurbLocYAML(sFileName):
    """Retrieve turbine locations and auxiliary file names from YAML file

    The auxiliary files are for the wind rose and the turbine attributes.

    """
    with open(sFileName, 'r') as f:
        defs = yaml.safe_load(f)['definitions']

    # rip the x- and y-coordinates (Convert from <list> to <ndarray>)
    turbineX = np.asarray(defs['position']['items']['xc'])
    turbineY = np.asarray(defs['position']['items']['yc'])
    coords = np.recarray(turbineX.shape, coordinate)
    coords.x, coords.y = turbineX, turbineY

    # Read the filenames for the windrose and the turbine attributes
    lTurbRefs = defs['wind_plant']['properties']['layout']['items']
    lWindRefs = (defs['plant_energy']['properties']['wind_resource_selection']
                                                   ['properties']['items'])
    # Loop through all references until we find a file reference
    # The files we want correspond to the first references that do not
    # reference a section in this document
    sTurbFile = next(ref['$ref'] for ref in lTurbRefs if ref['$ref'][0] != '#')
    sRoseFile = next(ref['$ref'] for ref in lWindRefs if ref['$ref'][0] != '#')

    # rip the expected AEP, used for comparison
    # AEP = (defs['plant_energy']['properties']['annual_energy_production']
    #                                                              ['default'])

    # Return turbine (x, y) locations, and the filenames for the others .yamls
    return coords, sRoseFile, sTurbFile


def getWindRoseYAML(sFileName):
    """Retrieve wind rose (directions, frequencies, speeds) from YAML file"""
    with open(sFileName, 'r') as f:
        props = (yaml.safe_load(f)['definitions']['wind_inflow']['properties'])

    # rip wind directional bins, their frequency, and the farm windspeed
    # (Convert from <list> to <ndarray>)
    windDir = np.asarray(props['direction']['bins'])
    windFreq = np.asarray(props['probability']['default'])
    # (Convert from <list> to <float>)
    windSpeed = float(props['speed']['default'])

    return windDir, windFreq, windSpeed


def getTurbAtrbtYAML(sFileName):
    """Retrieve turbine attributes from YAML file"""
    with open(sFileName, 'r') as f:
        definitions = yaml.safe_load(f)['definitions']
        opmodeprops = definitions['operating_mode']['properties']

    # rip the turbine attributes
    # (Convert from <list> to <float>)
    CutInWS = float(opmodeprops['cut_in_wind_speed']['default'])
    CutOutWS = float(opmodeprops['cut_out_wind_speed']['default'])
    RtdWS = float(opmodeprops['rated_wind_speed']['default'])
    RtdPwr = float(definitions['wind_turbine_lookup']['properties']['power']
                                                                   ['maximum'])

    return CutInWS, CutOutWS, RtdWS, RtdPwr


if __name__ == "__main__":
    """Used for demonstration

    An example command line syntax to run this file is

        python iea37-aepcalc.py iea37-ex16.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".

    """

    # Read necessary values from .yaml files
    # - Get turbine locations and auxiliary file names from .yaml file
    coords, sRoseFile, sTurbFile = getTurbLocYAML(sys.argv[1])
    # - Get the array wind sampling bins, frequency at each bin, and wind speed
    windDir, windFreq, windSpeed = getWindRoseYAML(sRoseFile)
    # - Pull needed values from the turbine file
    turbCutInWS, turbCutOutWS, \
        turbRtdWS, turbRtdPwr = getTurbAtrbtYAML(sTurbFile)

    # Calculate the AEP from ripped values
    AEP = calcAEP(coords, windFreq, windSpeed, windDir,
                  turbCutInWS, turbCutOutWS, turbRtdWS, turbRtdPwr)
    # Print AEP for each binned direction, with 5 digits behind the decimal.
    print(np.around(AEP, decimals=5))
    # Print AEP summed for all directions
    print(np.around(np.sum(AEP), decimals=5))
