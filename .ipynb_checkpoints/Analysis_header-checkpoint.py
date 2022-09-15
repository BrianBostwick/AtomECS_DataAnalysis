#!/usr/bin/env python

#Python script to import data from AtomECS to use with DataImport_header.py

import numpy as np
import scipy as sp
from scipy import constants

import matplotlib.pyplot as plt
from timeit import timeit

#contants
e       = sp.constants.value(u'elementary charge')            #1.602176634e-19 C
epsilon = sp.constants.value(u'vacuum electric permittivity') #8.8541878128e-12 F m^-1
pi      = sp.pi                                               #3.141592653589793
m_e     = sp.constants.value(u'electron mass')                #9.1093837015e-31 kg
c       = sp.constants.value(u'speed of light in vacuum')     #299792458.0 m s^-1

kg2amu = sp.constants.value(u'kilogram-atomic mass unit relationship') #6.0221407621e+26 u
amu2kg = 1/kg2amu

line = "____________________________________________________"
bline= "===================================================="

def get_coordinates(dat):
    a = dat.index(',')
    location = [dat.index('('), a, dat.index(',', a+1), dat.index(')')]
    coordinates = [ float(dat[(location[i]+1):location[i+1]]) for i in range(3) ]
    return coordinates

def get_particle_data( file ):

    Step, ParticleNumber, ParticleData = [], [], []

    with open( file , 'r') as myfile:

        while( True ):

            #Getting file data line-by-line.
            MetaData = myfile.readline()
            #Check end of file.
            if len(MetaData) == 0:
                break

            Step_tmp           = int(MetaData[ MetaData.index('-')+1 : MetaData.index(',') ])
            ParticleNumber_tmp = int(MetaData[ MetaData.index(',')+1 :  ])

            ParticleID       = []
            ParticleLocation_tmp = []

            for i in range(ParticleNumber_tmp):
                PositionData = myfile.readline()
                ParticleID.append(PositionData[ PositionData.index(',')+1: PositionData.index(':') ])
                ParticleLocation_tmp.append(get_coordinates(PositionData[ PositionData.index(' ')+1:    PositionData.index('\n') ]))

            Step.append(Step_tmp)
            ParticleNumber.append(ParticleNumber_tmp)
            ParticleData.append(ParticleLocation_tmp)

    return Step, ParticleNumber, ParticleData

def get_power_data( file ):
    power = []
    with open( file , 'r') as myfile:
        while( True ):
            #Getting file data line-by-line.
            data = myfile.readline()
            #Check end of file.
            if len(data) == 0:
                break
            power.append(float(data[:len(data)-1]))
    return power

def get_Vrms( ParticleVelovity ):
    '''Gets root mean squeard velcity per run
    '''
    Vrms = [ np.sqrt(sum( [j[0]**2 + j[1]**2 + j[2]**2 for j in i] )/len(i)) if len(i) > 0 else 0.0 for i in ParticleVelovity]
    return Vrms

def get_kinetic_energy( ParticleVelovity, Mass):
    '''Gets kinetic_energy using root mean squeard velcity per run
    '''
    Vrms = get_Vrms(ParticleVelovity)
    KineticEnergy = [0.5*Mass*(v)**2 for v in Vrms]
    return KineticEnergy

def get_tempeture(ParticleVelovity, Mass):
    '''Gets temp[K] using kinetic_energy using root mean squeard velcity per run
    '''
    kB = constants.value(u'Boltzmann constant') #1.380649e-23 J K^-1
    KineticEnergy = get_kinetic_energy( ParticleVelovity, Mass)
    T = [ (2/3)*(Ke/kB) for Ke in KineticEnergy ]
    return T

def get_particle_velocity(x):
    return np.sqrt(sum([i**2 for i in x]))

# def TheroreticalFreq(power, e_sqrd_radius = 50.0e-6, transition_wavelength = 461.0e-9 , laser_wavelength = 1064.0e-9, mass = 87, transition_linewidth = 30.5e6):
#     P_0 = power
#     w_0 = e_sqrd_radius
#     omega_0 = 2*np.pi*c/transition_wavelength
#     omega   = 2*np.pi*c/laser_wavelength
#     m   = mass*amu2kg
    
#     z_r = np.pi * (omega_0**2) * 1 / laser_wavelength
        
#     I_0 = 2*P_0/(pi*w_0**2)
#     Gamma = 2*pi*transition_linewidth

#     co_rotating      = Gamma / (omega_0 - omega)
#     counter_rotating = Gamma / (omega_0 + omega)
#     contants = 3*pi*c**2/(2*omega_0**3)

#     U_0 = I_0 * contants * (co_rotating + counter_rotating)
    
#     omega_r_trap = np.sqrt( 4 * np.abs(U_0) / (m * omega_0**2) )
    
#     print(omega_0)
    
#     omega_z_trap = np.sqrt( 2 * np.abs(U_0) / (m * z_r**2 ) )
#     return  omega_r_trap,  omega_z_trap


# def plot_data_1d( Ensamble, fig, ax, params, particle_number, step_number, step_size):

#     t = np.linspace(step_size, step_number, int(step_number/step_size))

#     for i in range(particle_number): #partible
#         x = np.zeros(int(step_number/step_size))
#         y = np.zeros(int(step_number/step_size))
#         z = np.zeros(int(step_number/step_size))

#         for j in range(0, int(step_number/step_size)): #time step
#             x[j] = (Ensamble[i][j][0])
#             y[j] = (Ensamble[i][j][1])
#             z[j] = (Ensamble[i][j][2])

#         ax[0].plot(t, x)
#         ax[0].grid()

#         ax[1].plot(t, y)
#         ax[1].grid()

#         ax[2].plot(t, z)
#         ax[2].grid()

#         ax[0].set(title =params[2], ylabel="x")
#         ax[1].set(ylabel=params[1]+"\n y")
#         ax[2].set(xlabel=params[0], ylabel="z")

# def plot_data_3d( Ensamble, fig, ax, params, particle_number, step_number, step_size):

#     t = np.linspace(step_size, step_number, int(step_number/step_size))

#     for i in range(particle_number): #particle
#         x = np.zeros(int(step_number/step_size))
#         y = np.zeros(int(step_number/step_size))
#         z = np.zeros(int(step_number/step_size))

#         for j in range(0, int(step_number/step_size)): #time step
#             x[j] = (Ensamble[i][j][0])
#             y[j] = (Ensamble[i][j][1])
#             z[j] = (Ensamble[i][j][2])

#             ax.plot(x, y, z, label=params[2])

            
#Code from old header file. 

# def get_kinetic_energy( Ensamble, particle_number, step_number, step_size ):

#     kinetic_energy = np.zeros((particle_number, int(step_number/step_size)))
#     mass = 87*1.66e-27

#     for i in range(particle_number): #partible
#         for j in range(0, int(step_number/step_size)): #time step
#             kinetic_energy[i, j] = 0.5*mass*((Ensamble[i][j][0])**2 + (Ensamble[i][j][1])**2 + (Ensamble[i][j][2])**2)

#     return kinetic_energy

# def avg_kinetic_energy( kinetic_energy ):
#     avg = np.sum(kinetic_energy, axis = 0)/100
#     return avg


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    
    popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

# def get_PotentialEnergy( data ):
#     mass  = 87*1.66e-27
#     wr = 2356#*2*3.1415
#     wz = 3332#*2*3.1415
#     return 0.5*mass*( wr**2 * (data[0]**2 + data[1]**2) + wz**2 * data[2]**2 ) 

# def get_ParticleVelocity(x):
#     return np.sqrt(sum([i**2 for i in x]))

# def get_KineticEnergy( data ):
#     mass  = 87*1.66e-27
#     return 1/2 * mass *  get_ParticleVelocity( data )**2

def reformat(data, position):
    x = []
    for i in range(len(data)):
        x.append(data[i][0][position])
    return x

# def get_TotalEnergy_PerRun(Position_data, Velocity_data):
    
#     E = []
    
#     for i in range(len(Position_data)):
#         E.append(  get_PotentialEnergy(Position_data[i]) + get_KineticEnergy(Velocity_data[i]) )
        
#     return E





if __name__ == "__main__":
    # s = """get_particle_data("../vel.txt")"""
    # print(timeit(stmt=s, number=10000, setup="from __main__ import get_particle_data"))
    fileV = "../vel_dipole.txt"
    fileP = "../pos.txt"

    VelData = get_particle_data( fileV )[2]
    print(get_tempeture( VelData, 87*1.66e-27 ))

    plt.plot(get_particle_data( fileV )[0], get_tempeture( VelData, 87*1.66e-27 ))
    plt.show()

    
    
__author__ = "Brian Bostwick"
__copyright__ = "-"
__credits__ = ["Brian Bostwick"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Brian Bostwick"
__email__ = "blb42@cam.ac.uk"
__status__ = "Production"