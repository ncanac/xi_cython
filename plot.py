"""

August 17, 2011
xi_cython.py v 1.0
Tree code to compute redshift space two-point correlation function
Nicolas Canac
University of California--Irvine
University of Texas--Austin
Please contact me with questions, comments, and suggestions at:
ncanac@uci.edu

"""

import csv
from numpy import *
from time import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors, ticker
import pylab as plt
import optparse

def commandLine():
# Parses command line options and returns:
	parser = optparse.OptionParser()
	parser.add_option("-D", "--DD", action="store", type="string", dest="DDfile", #default="DD.txt",
			help="name of DD file")
	parser.add_option("-R", "--RR", action="store", type="string", dest="RRfile", #default="RR.txt",
			help="name of RR file")
	parser.add_option("-C", "--DR", action="store", type="string", dest="DRfile", #default="DR.txt",
			help="name of DR file")
	(options, args) = parser.parse_args()
	return options.DDfile, options.RRfile, options.DRfile

def calcXi(DD_bins, RR_bins, DR_bins, Nd, Nr):
	#Xi = (DD_bins*(Nr*(Nr-1.)/2.)) / (RR_bins*(Nd*(Nd-1.)/2.)) - 1.
	DD_bins = 2.*DD_bins/(Nd*(Nd-1.))
	RR_bins = 2.*RR_bins/(Nr*(Nr-1.))
	DR_bins = DR_bins/(Nd*Nr)
	Xi = (DD_bins - 2*DR_bins + RR_bins)/RR_bins
	#Xi = DD_bins/RR_bins - 1.
	#Xi = (Nr*(Nr-1.)*DD_bins)/(Nd*(Nd-1.)*RR_bins) - ((Nr-1.)*DR_bins)/(Nd*RR_bins)
	return Xi

def readfile(filename):
	openfile = csv.reader(open(filename, 'r'), delimiter = ' ', skipinitialspace = True)
	pair_counts = []
	i = 0
	for row in openfile:
		if i == 0:
			BIN_MIN = float(row[0])
		elif i == 1:
			BIN_MAX = float(row[0])
		elif i == 2:
			BIN_WIDTH = float(row[0])
		elif i == 3:
			N = int(row[0])
		else:
			counts = []
			for count in row:
				counts.append(float(count))
			pair_counts.append(counts)
		i += 1
	return BIN_MIN, BIN_MAX, BIN_WIDTH, N, pair_counts

def plotXi(Xi, BIN_MIN, BIN_MAX, BIN_WIDTH):
	X_LoS = arange(-(BIN_MAX)+BIN_WIDTH, BIN_MAX, BIN_WIDTH)
	Y_trans = X_LoS.copy()
	X_LoS, Y_trans = meshgrid(X_LoS, Y_trans)
	Xi_rotated = zeros_like(X_LoS)
	for row in range(len(Xi_rotated)):
		for col in range(len(Xi_rotated[0])):
			x = X_LoS[row][col]/BIN_WIDTH
			y = Y_trans[row][col]/BIN_WIDTH
			Xi_rotated[row][col] = Xi[abs(x)][abs(y)]
				
	levels = array([-1, -.03, -.01, -.007, -.003, -.001, 0, .001, .003, .007, .01, .03, .07, .1, .3, .7, 1, 3, 100])
	colorcode = ('black', '#151B54', '#151B8D', '#0300ff', '#002cff', '#006cff', '#00bcff', '#00fff2', '#00ff92', '#00ff12', '#5eff00', '#beff00', '#feff00', '#ffc000', '#ff8000', '#ff3000', '#C11B17', '#800517')
	CS = plt.contourf(X_LoS, Y_trans, Xi_rotated, levels, colors=colorcode)
	cbar = plt.colorbar(CS)
	plt.xlabel('Transverse [Mpc]')
	plt.ylabel('LoS [Mpc]')
	plt.xlim((-BIN_MAX+BIN_WIDTH,BIN_MAX-BIN_WIDTH))
	plt.ylim((-BIN_MAX+BIN_WIDTH,BIN_MAX-BIN_WIDTH))
	plt.show()

def main():
	DDfile, RRfile, DRfile = commandLine()
	BIN_MIN, BIN_MAX, BIN_WIDTH, Nd, DD_bins = readfile(DDfile)
	BIN_MIN, BIN_MAX, BIN_WIDTH, Nr, RR_bins = readfile(RRfile)
	BIN_MIN, BIN_MAX, BIN_WIDTH, N, DR_bins = readfile(DRfile)
	Xi = calcXi(array(DD_bins), array(RR_bins), array(DR_bins), Nd, Nr)
	plotXi(Xi, BIN_MIN, BIN_MAX, BIN_WIDTH)

main()

