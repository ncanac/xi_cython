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

import tree
import csv
import time
import optparse
import multiprocessing

def ReadData(filename):
	########################### Read simulation data file ###########################
	points = []
	#openfile = csv.reader(open('galaxies.txt', 'r'), delimiter = ' ', skipinitialspace = True)
	#openfile = csv.reader(open('positions_z3.0_massive_4e12.txt', 'r'), delimiter = ' ', skipinitialspace = True)
	#openfile = csv.reader(open('wmap5baosn_8000points_pos2.txt', 'r'), delimiter = ' ', skipinitialspace = True)
	openfile = csv.reader(open(filename, 'r'), delimiter = ' ', skipinitialspace = True)
	for row in openfile:
		if row[0] != '#':
			points.append([float(row[0]), float(row[1]), float(row[2])])
	return points

def commandLine():
# Parses command line options and returns:
	parser = optparse.OptionParser()
	parser.add_option("-c", "--config", action="store", type="string", dest="configfile",
			help="name of configuration file")
	(options, args) = parser.parse_args()
	return options.configfile

def parseConfig(configfile):
	f = open(configfile, 'r').readlines()
	config_params = {'path': 'pathname', 'nproc': -1, 'datafile': 'data.txt', 'randomfile': 'random.txt', \
					 'rmin': 0, 'rmax': 150, 'nbins': 150, 'DD': True, 'RR': False, 'DR': False}
	for line in f:
		words_list = line.split()
		if len(words_list) > 0:
			if words_list[0] in config_params.keys():
				config_params[words_list[0]] = words_list[2]
	return config_params

def gen_rands(N, xmin, xmax, ymin, ymax, zmin, zmax):
	randoms = []
	for i in range(N):
		randoms.append([rand.uniform(xmin, xmax), rand.uniform(ymin, ymax), rand.uniform(zmin, zmax)])
	return randoms

def saveFile(savename, BIN_MIN, BIN_MAX, BIN_WIDTH, N, bins):
	f = open(savename, 'w')
	f.write(str(BIN_MIN)+'\n')
	f.write(str(BIN_MAX)+'\n')
	f.write(str(BIN_WIDTH)+'\n')
	f.write(str(N)+'\n')
	for row in bins:
		for num in row:
			f.write(' ' + str(num))
		f.write('\n')
	f.close()

def main():
	configfile = commandLine()
	if configfile is None:
		print "Please supply necessary command line arguments. Type '-h' or '--help' for help options."
		return
	config_options = parseConfig(configfile)
	BIN_MIN = float(config_options['rmin'])
	BIN_MAX = float(config_options['rmax'])
	BIN_WIDTH = (BIN_MAX - BIN_MIN) / float(config_options['nbins'])
	nproc = int(config_options['nproc']) if int(config_options['nproc']) > 0 else multiprocessing.cpu_count()
	DD = 'True' == config_options['DD']
	RR = 'True' == config_options['RR']
	DR = 'True' == config_options['DR']
	path = config_options['path']
	data = ReadData(path + config_options['datafile'])
	random = ReadData(path + config_options['randomfile'])
	t0 = time.time()
	if DD:
		DDbins = tree.compute_bins(data, data, BIN_MIN, BIN_MAX, BIN_WIDTH, nproc)
		saveFile('DD.txt', BIN_MIN, BIN_MAX, BIN_WIDTH, len(data), DDbins)
	t1 = time.time()
	if RR:
		RRbins = tree.compute_bins(random, random, BIN_MIN, BIN_MAX, BIN_WIDTH, nproc)
		saveFile('RR.txt', BIN_MIN, BIN_MAX, BIN_WIDTH, len(random), RRbins)
	t2 = time.time()
	if DR:
		DRbins = tree.compute_bins(random, data, BIN_MIN, BIN_MAX, BIN_WIDTH, nproc)
		saveFile('DR.txt', BIN_MIN, BIN_MAX, BIN_WIDTH, -99, DRbins)
	tend=time.time()

	print t1-t0
	print t2-t1
	print tend-t2
	print tend-t0

	#print DDbins
	#print RRbins
	#print DRbins

main()
