"""IEA Task 37 Case Study 3 AEP Calculation Code

Written by Nicholas F. Baker, PJ Stanley, Jared Thomas (BYU FLOW lab) and Erik Quaeghebeur (TU Delft)
Released 22 Aug 2018 with case studies 1 & 2
Modified 15 Apr 2019 for case studies 3 and 4
"""

from __future__ import print_function   # For Python 3 compatibility
import numpy as np
import sys
import yaml                             # For reading .yaml files
from math import radians as DegToRad    # For converting degrees to radians
from math import log as ln              # For natural logrithm

# Structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


def WindFrame(turb_coords, wind_dir_deg):
    """Convert map coordinates to downwind/crosswind coordinates."""

    # Convert from meteorological polar system (CW, 0 deg.=N)
    # to standard polar system (CCW, 0 deg.=W)
    # Shift so North comes "along" x-axis, from left to right.
    wind_dir_deg = 270. - wind_dir_deg
    # Convert inflow wind direction from degrees to radians
    wind_dir_rad = DegToRad(wind_dir_deg)

    # Constants to use below
    cos_dir = np.cos(-wind_dir_rad)
    sin_dir = np.sin(-wind_dir_rad)
    # Convert to downwind(x) & crosswind(y) coordinates
    frame_coords = np.recarray(len(turb_coords), coordinate)
    frame_coords.x = (turb_coords[:, 0] * cos_dir) - \
        (turb_coords[:, 1] * sin_dir)
    frame_coords.y = (turb_coords[:, 0] * sin_dir) + \
        (turb_coords[:, 1] * cos_dir)

    return frame_coords


def GaussianWake(frame_coords, turb_diam):
    """Return each turbine's total loss due to wake from upstream turbines"""
    # Equations and values explained in <iea37-wakemodel.pdf>
    num_turb = len(frame_coords)

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555
    # Array holding the wake deficit seen at each turbine
    loss = np.zeros(num_turb)

    for i in range(num_turb):            # Looking at each turb (Primary)
        loss_array = np.zeros(num_turb)  # Calculate the loss from all others
        for j in range(num_turb):        # Looking at all other turbs (Target)
            x = frame_coords.x[i] - frame_coords.x[j]   # Calculate the x-dist
            y = frame_coords.y[i] - frame_coords.y[j]   # And the y-offset
            if x > 0.:                   # If Primary is downwind of the Target
                sigma = k*x + turb_diam/np.sqrt(8.)  # Calculate the wake loss
                # Simplified Bastankhah Gaussian wake model
                exponent = -0.5 * (y/sigma)**2
                radical = 1. - CT/(8.*sigma**2 / turb_diam**2)
                loss_array[j] = (1.-np.sqrt(radical)) * np.exp(exponent)
            # Note that if the Target is upstream, loss is defaulted to zero
        # Total wake losses from all upstream turbs, using sqrt of sum of sqrs
        loss[i] = np.sqrt(np.sum(loss_array**2))

    return loss


def DirPower(frame_coords, dir_loss, wind_speed,
             turb_ci, turb_co, rated_ws, rated_pwr):
    """Return the power produced by each turbine."""
    num_turb = frame_coords.shape[0]

    # Effective windspeed is freestream multiplied by wake deficits
    wind_speed_eff = wind_speed*(1.-dir_loss)
    # By default, the turbine's power output is zero
    turb_pwr = np.zeros(num_turb)

    # Check to see if turbine produces power for experienced wind speed
    for n in range(num_turb):
        # If we're between the cut-in and rated wind speeds
        if ((turb_ci <= wind_speed_eff[n])
                and (wind_speed_eff[n] < rated_ws)):
            # Calculate the curve's power
            turb_pwr[n] = rated_pwr * ((wind_speed_eff[n]-turb_ci)
                                       / (rated_ws-turb_ci))**3
        # If we're between the rated and cut-out wind speeds
        elif ((rated_ws <= wind_speed_eff[n])
                and (wind_speed_eff[n] < turb_co)):
            # Produce the rated power
            turb_pwr[n] = rated_pwr

    # Sum the power from all turbines for this direction
    pwrDir = np.sum(turb_pwr)

    return pwrDir


def calcAEPcs3(turb_coords, wind_freq, wind_speeds, wind_speed_probs, wind_dir,
            turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    """Calculate the wind farm AEP."""
    num_dir_bins = wind_freq.shape[0]       # Number of bins used for our windrose
    num_speed_bins = wind_speeds.shape[0]   # Number of wind speed bins

    # Power produced by the wind farm from each wind direction
    pwr_prod_dir = np.zeros(num_dir_bins)
    #  Power produced by the wind farm at a given windspeed
    pwr_prod_ws = np.zeros((num_dir_bins, num_speed_bins))

    # For each directional bin
    for i in range(num_dir_bins):
        # For each wind speed bin
        # Shift coordinate frame of reference to downwind/crosswind
        frame_coords = WindFrame(turb_coords, wind_dir[i])
        # Use the Simplified Bastankhah Gaussian wake model for wake deficits
        dir_loss = GaussianWake(frame_coords, turb_diam)

        for j in range(num_speed_bins):
            # Find the farm's power for the current direction and speed,
            # multiplied by the probability that the speed will occur
            pwr_prod_ws[i][j] = DirPower(frame_coords, dir_loss, wind_speeds[j],
                                        turb_ci, turb_co, rated_ws,
                                        rated_pwr) * wind_speed_probs[i][j]
        pwr_prod_dir[i] = sum(pwr_prod_ws[i]) * wind_freq[i]

    #  Convert power to AEP
    hrs_per_year = 365.*24.
    AEP = hrs_per_year * pwr_prod_dir
    AEP /= 1.E6  # Convert to MWh

    return AEP


def getTurbLocYAML(file_name):
    """ Retrieve turbine locations and auxiliary file names from <.yaml> file.

    Auxiliary (reference) files supply wind rose and turbine attributes.
    """
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        defs = yaml.safe_load(f)['definitions']

    # Rip the (x,y) coordinates (Convert from <list> to <ndarray>)
    turb_coords = np.asarray(defs['position']['items'])

    # Rip the expected AEP, used for comparison
    # AEP = defs['plant_energy']['properties']
    #           ['annual_energy_production']['default']

    # Read the auxiliary filenames for the windrose and the turbine attributes
    ref_list_turbs = defs['wind_plant']['properties']['turbine']['items']
    ref_list_wr = (defs['plant_energy']['properties']
                       ['wind_resource']['properties']['items'])

    # Iterate through all listed references until we find the one we want
    # The one we want is the first reference not internal to the document
    # Note: internal references use '#' as the first character
    fname_turb = next(ref['$ref']
                      for ref in ref_list_turbs if ref['$ref'][0] != '#')
    fname_wr = next(ref['$ref']
                    for ref in ref_list_wr if ref['$ref'][0] != '#')

    # Return turbine (x,y) locations, and the filenames for the others .yamls
    return turb_coords, fname_turb, fname_wr


def getWindRoseYAML(file_name):
    """Retrieve wind rose data (bins, freqs, speeds) from <.yaml> file."""
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)['definitions']['wind_inflow']['properties']

    # Rip wind directional bins, their frequency, and the windspeed parameters for each bin
    # (Convert from <list> to <ndarray>)
    wind_dir = np.asarray(props['direction']['bins'])
    wind_dir_freq = np.asarray(props['direction']['frequency'])
    # (Convert from <list> to <float>)
    wind_speeds = np.asarray(props['speed']['bins'])
    wind_speed_probs = np.asarray(props['speed']['frequency'])
    # Get default number of windspeed bins per direction
    num_speed_bins = wind_speeds.shape[0]
    min_speed = props['speed']['minimum']
    max_speed = props['speed']['maximum']

    return wind_dir, wind_dir_freq, wind_speeds, wind_speed_probs, num_speed_bins, min_speed, max_speed


def getTurbAtrbtYAML(file_name):
    '''Retreive turbine attributes from the <.yaml> file'''
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        defs = yaml.safe_load(f)['definitions']
        ops = defs['operating_mode']
        turb = defs['wind_turbine']
        rotor = defs['rotor']

    # Rip the turbine attributes
    # (Convert from <list> to <float>)
    turb_ci = float(ops['cut_in_wind_speed']['default'])
    turb_co = float(ops['cut_out_wind_speed']['default'])
    rated_ws = float(ops['rated_wind_speed']['default'])
    rated_pwr = float(turb['rated_power']['maximum'])
    turb_diam = float(rotor['diameter']['default'])

    return turb_ci, turb_co, rated_ws, rated_pwr, turb_diam


if __name__ == "__main__":
    """Used for demonstration.

    An example command line syntax to run this file is:

        python iea37-aepcalc.py iea37-ex-opt3.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".
    """
    #-- Read necessary values from .yaml files --#
    # Get turbine locations and auxiliary <.yaml> filenames
    turb_coords, fname_turb, fname_wr = getTurbLocYAML(sys.argv[1])
    # Get the array wind sampling bins, frequency at each bin, and wind speed
    wind_dir, wind_dir_freq, wind_speeds, wind_speed_probs, num_speed_bins, min_speed, max_speed = getWindRoseYAML(
        fname_wr)
    # Pull the needed turbine attributes from file
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(
        fname_turb)

    #-- Calculate the AEP from ripped values --#
    AEP = calcAEPcs3(turb_coords, wind_dir_freq, wind_speeds, wind_speed_probs, wind_dir,
                turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    # Print AEP summed for all directions
    print(np.around(AEP, decimals=5))
    print(np.around(np.sum(AEP), decimals=5))
