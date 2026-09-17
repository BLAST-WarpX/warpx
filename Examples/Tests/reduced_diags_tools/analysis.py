#!/usr/bin/env python3

#####################################################
# examples of usage of the reduced diagnostic tools #
#####################################################

import sys

sys.path.append('../../../Python/pywarpx/diagnostics/data/')
mydir = '.'

##################
# ParticleEnergy #
##################

import ParticleEnergy as PE

diag = PE.ParticleEnergyData(fname='PartEne.txt')

# extract some general info
times = diag.get_times()
print('available times\n {}'.format(times))

steps = diag.get_steps()
print('available steps\n {}'.format(steps))

nspecies = diag.get_nspecies()
print('number of species = {}'.format(nspecies))

species_names = diag.get_species_names()
print('species names = {}'.format(species_names))

print('\n')

# extract data
data = diag.get_data('photons', 'electrons', times=None)
print('asking for total photon and electron energy at all available times')
print(data)
print('\n')

data = diag.get_data('protons_mean', 'electrons')
print('asking for mean proton energy and total electron energy at all available times')
print(data)
print('\n')

data = diag.get_data('protons', 'electrons', steps=200)
print('asking for proton and electron total energy at step 200')
print(data)
print('\n')

data = diag.get_data('photons_mean', 'photons', steps=[0,200])
print('asking for mean and total photon energy at steps 0 and 200')
print(data)
print('\n')

data = diag.get_data('photons', 'protons', 'electrons', times=10e-9)
print('asking for total photon, proton and electron energy at the time closest to 10 ns')
print(data)
print('\n')

data = diag.get_data('electrons', times=[2e-9, 4e-9, 9e-9])
print('asking for total electron energy at the times closest to 2, 4 and 9 ns')
print(data)
print('\n')

data = diag.get_data('electrons', steps=[0,20,40,100])
print('asking for total electron energy at steps 0, 20, 40 and 100')
print(data)
print('\n')

####################
# ParticleMomentum #
####################

import ParticleMomentum as PM

diag = PM.ParticleMomentumData(fname='PartMom.txt')

print(diag.get_valid_args())


species_names = diag.get_species_names()
print('species names = {}'.format(species_names))

data = diag.get_data('electrons_x', 'electrons_mean_x', steps=[0,20,60])
print('asking for total and mean electron momentum x at all steps')
print(data)
print('\n')

###############
# FieldEnergy #
###############

import FieldEnergy as FE

diag = FE.FieldEnergyData(red_diags_dir="./diags/reducedfiles", fname='FielEne.txt')

print(diag.get_valid_args())

data = diag.get_data('E_lev0', 'B_lev0')
print('asking for energy stored in E and B at level 0 at all steps')
print(data)

print('\n')

################
# FieldMaximum #
################

import FieldMaximum as FM

diag = FM.FieldMaximumData(fname='FielMax.txt')

print(diag.get_valid_args())

data = diag.get_data('max_|E|_lev0', 'max_|B|_lev0')
print('asking for max fields at all steps')
print(data)
print('\n')

##################
# ParticleNumber #
##################

import ParticleNumber as PN

diag = PN.ParticleNumberData(fname='PartNum.txt')

print(diag.get_valid_args())

species_names = diag.get_species_names()
print('species names = {}'.format(species_names))

data = diag.get_data('protons_macroparticles', 'photons_weight', 'total_weight')
print('asking for proton macroparticles, photon weight and total weight at all steps')
print(data)
print('\n')

##############
# RhoMaximum #
##############

import RhoMaximum as RM

diag = RM.RhoMaximumData(fname='DensMax.txt')

print(diag.get_valid_args())

species_names = diag.get_nonpho_species_names()
print('non-photon species names = {}'.format(species_names))

data = diag.get_data('max_electrons_|rho|_lev0', 'min_rho_lev0', 'max_rho_lev0')
print('asking for max electron rho, min rho, max rho at level 0')
print(data)
print('\n')

###################
# ParticleExtrema #
###################

import ParticleExtrema as PX

diag = PX.ParticleExtremaData(fname='PartEx_ele.txt')

print(diag.get_valid_args())

data = diag.get_data('xmin', 'pxmax', 'wmax')
print('asking for xmin, pxmax, wmax')
print(data)
print('\n')
