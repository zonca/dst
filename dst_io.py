import h5py
from mpi4py import MPI
import healpy as hp
import numpy as np
import logging as l

def get_qu_weights(pa):
    cos2pa = np.cos(pa)**2 - np.sin(pa)**2
    sin2pa = 2*np.sin(pa)*np.cos(pa)
    q_channel_w = {'Q': cos2pa, 'U': sin2pa}
    u_channel_w = {'Q': -sin2pa, 'U': cos2pa}
    return q_channel_w, u_channel_w

def read_data(filename, i_from, i_to, nside, BaselineLength, comm, pol=False, maskdestripe=True):
    """Read data serially with h5py

    a cython version using HDF5 parallel I/O is available, but complicated to build
    and less flexible"""

    l.info("Proc %d: Read %s %d-%d" % (comm.MyPID, filename, i_from, i_to))
    with h5py.File(filename, mode="r") as h5file:
        data = h5file["data"][i_from:i_to]

    pix = hp.ang2pix(nside,data['THETA'],data['PHI'])

    if maskdestripe:
        mask = hp.ud_grade(hp.read_map("mask.fits"), nside)
        data["FLAG"][mask[pix] == 0] = 1

    good_data = data["FLAG"] == 0
    BaselineLengths = np.sum((good_data).reshape(-1, BaselineLength), axis = 1)
    
    l.info("Proc %d: Total data %d, good data %d" % (comm.MyPID, len(data), (data['FLAG'] == 0).sum()))
    data = data [good_data]
    pix = pix [good_data]

    if pol:
        d = {'Q':data['Q'].copy(), 'U':data['U'].copy()}
        d['q_channel_w'], d['u_channel_w'] = get_qu_weights(np.radians(data['PSI']))
        d['PSI'] = data['PSI'].copy()
    else:
        d = {'T':data['TEMP'].copy()}

    return pix, d, BaselineLengths

def MPIwrite(filename, a, comm):
    """MPI I/O array write, using mpi4py"""
    if comm.MyPID == 0:
        l.info("Writing " + filename)
    f = MPI.File.Open(MPI.COMM_WORLD, filename, 
                      MPI.MODE_WRONLY | MPI.MODE_CREATE)
    f.Write_ordered(a)
    f.Close()
