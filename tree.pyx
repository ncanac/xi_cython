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

import numpy as np
import csv
import time
import Queue
import multiprocessing
import math
import sys

cimport numpy as np

def callDualTreeCount(args):
	tree1 = np.array(args[0], dtype=np.int)
	tree2 = np.array(args[1], dtype=np.int)
	hrectree1 = np.array(args[2], dtype=np.float)
	hrectree2 = np.array(args[3], dtype=np.float)
	leaf_points1 = np.array(args[4], dtype=np.float)
	leaf_points2 = np.array(args[5], dtype=np.float)
	node = args[6]
	cdef float BIN_MIN = args[7]
	cdef float BIN_MAX = args[8]
	cdef float BIN_WIDTH = args[9]
	cross = args[10]
	if cross:
		return dualTreeCountCross(BIN_MIN, BIN_MAX, BIN_WIDTH, tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, node)
	return dualTreeCountAuto(BIN_MIN, BIN_MAX, BIN_WIDTH, tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, node)

def callDualTreeCountCross(args):
	tree1 = np.array(args[0], dtype=np.int)
	tree2 = np.array(args[1], dtype=np.int)
	hrectree1 = np.array(args[2], dtype=np.float)
	hrectree2 = np.array(args[3], dtype=np.float)
	leaf_points1 = np.array(args[4], dtype=np.float)
	leaf_points2 = np.array(args[5], dtype=np.float)
	node = args[6]
	return dualTreeCountCross(tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, node)

cdef inline float float_max(float a, float b): return a if a >= b else b

#cdef inline float float_min(float a, float b): return a if a <= b else b

cimport cython
@cython.boundscheck(False)
def dualTreeCountAuto(int BIN_MIN, int BIN_MAX, int BIN_WIDTH, 
					np.ndarray[np.int_t, ndim=2] tree1, np.ndarray[np.int_t, ndim=2] tree2,
					np.ndarray[np.float_t, ndim=3] hrectree1, np.ndarray[np.float_t, ndim=3] hrectree2,
					np.ndarray[np.float_t, ndim=3] leaf_points1, np.ndarray[np.float_t, ndim=3] leaf_points2,
					int node_idx1, int node_idx2=0):
	# initialize stack and bin counts
	cdef np.ndarray[np.int_t, ndim=1] stack = np.zeros((len(tree1)+len(tree2))*2, dtype=np.int)
	stack[0] = node_idx2
	stack[1] = node_idx1
	cdef int stack_ptr = 1
	cdef unsigned int numbins = int((BIN_MAX-BIN_MIN)/BIN_WIDTH)
	cdef np.ndarray[np.int_t, ndim=2] bins = np.zeros((numbins,numbins), dtype=np.int)
	cdef unsigned int idx1, idx2, numpoints1, numpoints2, left1, right1, \
						left2, right2, i, j, r_index, z_index
	cdef int leaf_ptr1, leaf_ptr2
	cdef float dmin, dmax, zmin, zmax, dist
	cdef np.ndarray[np.float_t, ndim=2] hrect1
	cdef np.ndarray[np.float_t, ndim=2] hrect2 
	while stack_ptr >= 0:
		idx1 = stack[stack_ptr]
		stack_ptr -= 1
		idx2 = stack[stack_ptr]
		stack_ptr -= 1
		if idx2 <= idx1:
			numpoints1 = tree1[idx1][0]
			leaf_ptr1 = tree1[idx1][1]
			left1 = tree1[idx1][2]
			right1 = tree1[idx1][3]
			numpoints2 = tree2[idx2][0]
			leaf_ptr2 = tree2[idx2][1]
			left2 = tree2[idx2][2]
			right2 = tree2[idx2][3]
			hrect1 = hrectree1[idx1]	# bounding coordinates for rectangle 1 on transverse plane
			hrect2 = hrectree2[idx2]	# bounding coordinates for rectangle 2 on transverse plane
			zmin = float_max(0,float_max(hrect1[0,2]-hrect2[1,2],hrect2[0,2]-hrect1[1,2]))
			if zmin <= BIN_MAX:
				zmax = float_max(hrect1[1,2]-hrect2[0,2],hrect2[1,2]-hrect1[0,2])
				if zmax >= BIN_MIN:
					#dmin = (((np.maximum(0,np.maximum(hrect1[0,:2]-hrect2[1,:2],hrect2[0,:2]-hrect1[1,:2])))**2).sum())**(1./2)
					dmin = ((float_max(0.,float_max(hrect1[0][0]-hrect2[1][0],hrect2[0][0]-hrect1[1][0])))**2 + \
							(float_max(0.,float_max(hrect1[0][1]-hrect2[1][1],hrect2[0][1]-hrect1[1][1])))**2)**(1./2)
					if dmin <= BIN_MAX:
						#dmax = (((np.maximum(hrect1[1,:2]-hrect2[0,:2],hrect2[1,:2]-hrect1[0,:2]))**2).sum())**(1./2)
						dmax = ((float_max(hrect1[1,0]-hrect2[0,0],hrect2[1,0]-hrect1[0,0]))**2 + \
								(float_max(hrect1[1,1]-hrect2[0,1],hrect2[1,1]-hrect1[0,1]))**2)**(1./2)
						if dmax >= BIN_MIN:
							if idx1 == idx2:	# comparing same node, so avoid repeats
								if int((dmin-BIN_MIN)/BIN_WIDTH) == int((dmax-BIN_MIN)/BIN_WIDTH) and \
								int((zmin-BIN_MIN)/BIN_WIDTH) == int((zmax-BIN_MIN)/BIN_WIDTH):
									bins[int((dmin-BIN_MIN)/BIN_WIDTH)][int((zmin-BIN_MIN)/BIN_WIDTH)] += \
									numpoints1*(numpoints1-1)/2	# all points lie within distance of interest
								elif leaf_ptr1 != -1:	# leaf node and some points lie within, some outside, must iterate through
									for i in range(numpoints1-1):
										for j in range(i+1, numpoints1):
											#dist = (((leaf_points1[leaf_ptr1][i][:2]-leaf_points1[leaf_ptr1][j][:2])**2.).sum())**(1./2)
											dist = (((leaf_points1[leaf_ptr1][i][0]-leaf_points1[leaf_ptr1][j][0])**2.) + \
													((leaf_points1[leaf_ptr1][i][1]-leaf_points1[leaf_ptr1][j][1])**2.))**(1./2)
											r_index = int((dist-BIN_MIN)/BIN_WIDTH)
											dist = leaf_points1[leaf_ptr1][i][2]-leaf_points1[leaf_ptr1][j][2]
											if dist < 0:
												dist = dist*-1
											z_index = int((dist-BIN_MIN)/BIN_WIDTH)
											if r_index < numbins and z_index < numbins:
												bins[r_index][z_index] += 1
								else:
									if numpoints2 > numpoints1:
										stack_ptr += 1
										stack[stack_ptr] = right2
										stack_ptr += 1
										stack[stack_ptr] = idx1
										stack_ptr += 1
										stack[stack_ptr] = left2
										stack_ptr += 1
										stack[stack_ptr] = idx1
									else:
										stack_ptr += 1
										stack[stack_ptr] = idx2
										stack_ptr += 1
										stack[stack_ptr] = right1
										stack_ptr += 1
										stack[stack_ptr] = idx2
										stack_ptr += 1
										stack[stack_ptr] = left1
							else:	# comparing different nodes
								if int((dmin-BIN_MIN)/BIN_WIDTH) == int((dmax-BIN_MIN)/BIN_WIDTH) and \
								int((zmin-BIN_MIN)/BIN_WIDTH) == int((zmax-BIN_MIN)/BIN_WIDTH):
									bins[int((dmin-BIN_MIN)/BIN_WIDTH)][int((zmin-BIN_MIN)/BIN_WIDTH)] += \
									numpoints1*numpoints2		# all points lie within distance of interest
								elif leaf_ptr1 != -1 and leaf_ptr2 != -1:	# leaf nodes and some points lie within, some outside, must iterate through
									for i in range(numpoints1):
										for j in range(numpoints2):
											#dist = (((leaf_points1[leaf_ptr1][i][:2]-leaf_points2[leaf_ptr2][j][:2])**2.).sum())**(1./2)
											dist = (((leaf_points1[leaf_ptr1][i][0]-leaf_points2[leaf_ptr2][j][0])**2.) + \
													((leaf_points1[leaf_ptr1][i][1]-leaf_points2[leaf_ptr2][j][1])**2.))**(1./2)
											r_index = int((dist-BIN_MIN)/BIN_WIDTH)
											dist = leaf_points1[leaf_ptr1][i][2]-leaf_points2[leaf_ptr2][j][2]
											if dist < 0:
												dist = dist*-1
											z_index = int((dist-BIN_MIN)/BIN_WIDTH)
											if r_index < numbins and z_index < numbins:
												bins[r_index][z_index] += 1
								else:
									if numpoints2 > numpoints1:
										stack_ptr += 1
										stack[stack_ptr] = right2
										stack_ptr += 1
										stack[stack_ptr] = idx1
										stack_ptr += 1
										stack[stack_ptr] = left2
										stack_ptr += 1
										stack[stack_ptr] = idx1
									else:
										stack_ptr += 1
										stack[stack_ptr] = idx2
										stack_ptr += 1
										stack[stack_ptr] = right1
										stack_ptr += 1
										stack[stack_ptr] = idx2
										stack_ptr += 1
										stack[stack_ptr] = left1
	return bins

cimport cython
@cython.boundscheck(False)
def dualTreeCountCross(int BIN_MIN, int BIN_MAX, int BIN_WIDTH,
					np.ndarray[np.int_t, ndim=2] tree1, np.ndarray[np.int_t, ndim=2] tree2,
					np.ndarray[np.float_t, ndim=3] hrectree1, np.ndarray[np.float_t, ndim=3] hrectree2,
					np.ndarray[np.float_t, ndim=3] leaf_points1, np.ndarray[np.float_t, ndim=3] leaf_points2,
					int node_idx1, int node_idx2=0):
	# initialize stack and bin counts
	cdef np.ndarray[np.int_t, ndim=1] stack = np.zeros((len(tree1)+len(tree2))*2, dtype=np.int)
	stack[0] = node_idx2
	stack[1] = node_idx1
	cdef int stack_ptr = 1
	cdef unsigned int numbins = int((BIN_MAX-BIN_MIN)/BIN_WIDTH)
	cdef np.ndarray[np.int_t, ndim=2] bins = np.zeros((numbins,numbins), dtype=np.int)
	cdef unsigned int idx1, idx2, numpoints1, numpoints2, left1, right1, \
						left2, right2, i, j, r_index, z_index
	cdef int leaf_ptr1, leaf_ptr2
	cdef float dmin, dmax, zmin, zmax, dist
	cdef np.ndarray[np.float_t, ndim=2] hrect1
	cdef np.ndarray[np.float_t, ndim=2] hrect2
	while stack_ptr >= 0:
		idx1 = stack[stack_ptr]
		stack_ptr -= 1
		idx2 = stack[stack_ptr]
		stack_ptr -= 1
#		if idx2 <= idx1:
		numpoints1 = tree1[idx1][0]
		leaf_ptr1 = tree1[idx1][1]
		left1 = tree1[idx1][2]
		right1 = tree1[idx1][3]
		numpoints2 = tree2[idx2][0]
		leaf_ptr2 = tree2[idx2][1]
		left2 = tree2[idx2][2]
		right2 = tree2[idx2][3]
		hrect1 = hrectree1[idx1]	# bounding coordinates for rectangle 1 on transverse plane
		hrect2 = hrectree2[idx2]	# bounding coordinates for rectangle 2 on transverse plane
		zmin = float_max(0,float_max(hrect1[0,2]-hrect2[1,2],hrect2[0,2]-hrect1[1,2]))
		if zmin <= BIN_MAX:
			zmax = float_max(hrect1[1,2]-hrect2[0,2],hrect2[1,2]-hrect1[0,2])
			if zmax >= BIN_MIN:
				dmin = ((float_max(0.,float_max(hrect1[0][0]-hrect2[1][0],hrect2[0][0]-hrect1[1][0])))**2 + \
						(float_max(0.,float_max(hrect1[0][1]-hrect2[1][1],hrect2[0][1]-hrect1[1][1])))**2)**(1./2)
				if dmin <= BIN_MAX:
					dmax = ((float_max(hrect1[1,0]-hrect2[0,0],hrect2[1,0]-hrect1[0,0]))**2 + \
							(float_max(hrect1[1,1]-hrect2[0,1],hrect2[1,1]-hrect1[0,1]))**2)**(1./2)
					if dmax >= BIN_MIN:
						if int((dmin-BIN_MIN)/BIN_WIDTH) == int((dmax-BIN_MIN)/BIN_WIDTH) and \
						int((zmin-BIN_MIN)/BIN_WIDTH) == int((zmax-BIN_MIN)/BIN_WIDTH): # all points lie within distance of interest
							bins[int((dmin-BIN_MIN)/BIN_WIDTH)][int((zmin-BIN_MIN)/BIN_WIDTH)] += numpoints1*numpoints2
						elif leaf_ptr1 != -1 and leaf_ptr2 != -1:	# leaf nodes and some points lie within, some outside, must iterate through
							for i in range(numpoints1):
								for j in range(numpoints2):
									dist = (((leaf_points1[leaf_ptr1][i][0]-leaf_points2[leaf_ptr2][j][0])**2.) + \
											((leaf_points1[leaf_ptr1][i][1]-leaf_points2[leaf_ptr2][j][1])**2.))**(1./2)
									r_index = int((dist-BIN_MIN)/BIN_WIDTH)
									dist = leaf_points1[leaf_ptr1][i,2]-leaf_points2[leaf_ptr2][j,2]
									if dist < 0:
										dist = dist*-1
									z_index = int((dist-BIN_MIN)/BIN_WIDTH)
									if r_index < numbins and z_index < numbins:
										bins[r_index][z_index] += 1
						else:
							if numpoints2 > numpoints1:
								stack_ptr += 1
								stack[stack_ptr] = right2
								stack_ptr += 1
								stack[stack_ptr] = idx1
								stack_ptr += 1
								stack[stack_ptr] = left2
								stack_ptr += 1
								stack[stack_ptr] = idx1
							else:
								stack_ptr += 1
								stack[stack_ptr] = idx2
								stack_ptr += 1
								stack[stack_ptr] = right1
								stack_ptr += 1
								stack[stack_ptr] = idx2
								stack_ptr += 1
								stack[stack_ptr] = left1
	return bins

def dualTreeParallel(tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, nodelist, cross, BIN_MIN, \
					BIN_MAX, BIN_WIDTH, nproc):
	arglist = []
	for node in nodelist:
		arglist.append((tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, node, BIN_MIN, BIN_MAX, BIN_WIDTH, cross))
	pool = multiprocessing.Pool(processes=nproc)
	countslist = pool.map(callDualTreeCount, arglist)
        pool.close()
	numbins = (BIN_MAX - BIN_MIN) / BIN_WIDTH
	bins = np.zeros((numbins,numbins))
	for counts in countslist:
		bins += counts
	return bins

def kdtree(data, leafsize=10):
# Constructs a kd-tree from data with leaves containing no more than N=leafsize data points.
# Returns tree, hrectree, and leaf_points, which are balanced lists containing (number of points, pointer to
# leaf_points, pointer to left child, pointer to right child), coordinates of bounding hyperrectangles, and
# data points in a leaf, respectively. Same indexes in tree and hrectree correspond to the same node in the
# kd-tree.
	xvalues = []
	yvalues = []
	zvalues = []
	for point in data:
		xvalues.append(point[0])
		yvalues.append(point[1])
		zvalues.append(point[2])
	data = np.array([xvalues,yvalues,zvalues])

	ndim = data.shape[0]
	ndata = data.shape[1]

	# find bounding hyper-rectangle
	hrect = np.zeros((2,data.shape[0]))	# shape=(rows, columns)
	hrect[0,:] = data.min(axis=1)	# axis=0 looks at columns
	hrect[1,:] = data.max(axis=1)	# axis=1 looks at rows

	# compute hyperrectangles of left and right child
	idx = np.argsort(data[0,:], kind='mergesort')
	data[:,:] = data[:,idx]
	splitval = data[0,ndata/2]

	left_hrect = hrect.copy()
	right_hrect = hrect.copy()
	left_hrect[1, 0] = splitval
	right_hrect[0, 0] = splitval

	# create root of kd-tree
	# tree = [(numpoints, array index of leaf points, left child, right child)] where -1 is placeholder
	tree = [[ndata, -1, -1, -1]]
	hrectree = [hrect.copy()]
	leaf_points = []

	# push values onto stack
	# (tree data, indexes of right tree data, hyperrectangle, depth, parent, left branch?)
	stack = [(data[:,ndata/2:], right_hrect, 1, 0, False),
			 (data[:,:ndata/2], left_hrect, 1, 0, True)]

	# recursively split data in halves using hyper-rectangles:
	while stack:
		
		# pop data off stack
		data, hrect, depth, parent, leftbranch = stack.pop()
		ndata = data.shape[1]
		node_ptr = len(tree)

		# update parent node with left and right children
		_numpoints, _leaf_idx, left, right = tree[parent]
		tree[parent] = [_numpoints, _leaf_idx, node_ptr, right] if leftbranch else [_numpoints, _leaf_idx, left, node_ptr]

		# insert node in kd-tree
		hrectree.append(hrect)
		if ndata <= leafsize:	# leaf node?
			_data = []
			for i in range(leafsize):
				if i < ndata:
					_data.append([data[0][i], data[1][i], data[2][i]])
				else:
					_data.append([-1, -1, -1])	# dummy values to balance array
			leaf_points.append(np.array(_data))
			leaf = [ndata, len(leaf_points)-1, 0, 0]
			tree.append(leaf) 
		else:	# not a leaf, split the data in two	
			splitdim = depth % ndim		# alternate splitting dimension
			idx = np.argsort(data[splitdim,:], kind='mergesort')
			data[:,:] = data[:,idx]		# sort by splitdim axis
			node_ptr = len(tree)
			splitval = data[splitdim,ndata/2]
			left_hrect = hrect.copy()
			right_hrect = hrect.copy()
			left_hrect[1, splitdim] = splitval
			right_hrect[0, splitdim] = splitval
			stack.append((data[:,ndata/2:], right_hrect, depth+1, node_ptr, False))
			stack.append((data[:,:ndata/2], left_hrect, depth+1, node_ptr, True))
			tree.append([ndata, -1, -1, -1])

	return tree, hrectree, leaf_points

def makeNodeList(tree, N, nproc):
	cores = nproc
	nodelist = []
	q = Queue.Queue()
	q.put(tree[0])
	min_level = math.log10(cores)/math.log10(2)	# level that has at least as many nodes as there are cores
	max_level = math.log10(N/10.)/math.log10(2)	# approximate number of levels in tree
	if min_level < max_level:
		level = int(min_level + .75*(max_level - min_level))
	else:
		level = max_level - 2
	while q.qsize() < 2**(level-1):
		node = q.get()
		q.put(tree[node[2]])
		q.put(tree[node[3]])
	while not q.empty():
		node = q.get()
		nodelist.append(node[2])
		nodelist.append(node[3])
	return nodelist

def ReadData(filename):
	########################### Read simulation data file ###########################
	points = []
	openfile = csv.reader(open(filename, 'r'), delimiter = ' ', skipinitialspace = True)
	count = 0
	for row in openfile:
		try:
			if row[0] != '#':
				points.append([float(row[3]), float(row[4])])
				count += 1
				if count >= 500000:
					break
		except:
			print "ERROR"
	return points

def savefile(bins, N):
	#savez('DD_bins_z.npz', bins=bins, N=N)
	#savez('RR_bins_z.npz', bins=bins, N=N)
	print "Save file? (N/Y)"
	s = raw_input('>> ')
	if s == 'N':
		return
	print "Save as..."
	savename = raw_input('>> ')
	np.savez(savename, bins=bins, N=N)

def compute_bins(points1, points2, BIN_MIN, BIN_MAX, BIN_WIDTH, nproc):
	# Reads in (x,y) coordinates in units of pixels
	# 1 pixel = .06 arcsec
	# Bounding rectangle coordinates (48.66, 104.04), (1368, 661.2) [arcsec]
	# Maximum distance = 1432.16
	#filename1 = 'uds2e_tf_h_110412s_multi.cat'
	#filename2 = 'uds2e_tf_h_110412s_multi.cat'
	#points1 = ReadData(filename1)
	#points2 = ReadData(filename2)
	cross = True
	if points1 == points2:
		cross = False
	points1 = np.array(points1)
	points2 = np.array(points2)
	N = len(points1)
	tree1, hrectree1, leaf_points1 = kdtree(points1.copy())
	tree2, hrectree2, leaf_points2 = kdtree(points2.copy())
	tree1 = np.array(tree1, dtype=np.int)
	tree2 = np.array(tree2, dtype=np.int)
	hrectree1 = np.array(hrectree1, dtype=np.float)
	hrectree2 = np.array(hrectree2, dtype=np.float)
	leaf_points1 = np.array(leaf_points1, dtype=np.float)
	leaf_points2 = np.array(leaf_points2, dtype=np.float)
	nodelist = makeNodeList(tree1, N, nproc)
	t0 = time.time()
	bins = dualTreeParallel(tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, nodelist, cross, BIN_MIN, \
							BIN_MAX, BIN_WIDTH, nproc)
	#bins = dualTreeCountAuto(tree1, tree2, hrectree1, hrectree2, leaf_points1, leaf_points2, 0, 0)
	t1 = time.time()
	#print bins
	return bins

#main()
