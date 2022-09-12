#!/usr/bin/env python

#Python script to import data from AtomECS to use with Analysis_header.py

import numpy as np
import matplotlib.pyplot as plt

def get_coordinates(dat):
    """
    """
    x, y, z = "","",""
    coordinates = np.zeros(3)
    location = []
    i = 0

    while(i < len(dat)):
        if   dat[i] == "(":
            location.append(i)
        elif dat[i] == "," and i != 1:
            location.append(i)
        elif dat[i] == ")":
            location.append(i)
        i += 1

    for i in range(len(location)):
        location[i] = int(location[i])

    x = x + dat[(location[0]+1):location[1]]
    y = y + dat[(location[1]+1):location[2]]
    z = z + dat[(location[2]+1):location[3]]

    coordinates[0] = float(x)
    coordinates[1] = float(y)
    coordinates[2] = float(z)

    return np.array(coordinates)

def get_particle_data( file, particle_number, step_number, step_size):
    
    """
    """

    #particle array
    Ensamble = np.zeros((particle_number, int(step_number/step_size), 3))

    #importing data from the simulation
    file = open(file, 'r')

    for i in range(0, int((step_number+1)/step_size)):
        file.readline() #Header line for step and particle_number
        for j in range(particle_number):
            Ensamble[j,i] = get_coordinates(file.readline())

    file.close()

    return Ensamble



__author__ = "Brian Bostwick"
__copyright__ = "-"
__credits__ = ["Brian Bostwick"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Brian Bostwick"
__email__ = "blb42@cam.ac.uk"
__status__ = "Production"


