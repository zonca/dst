import numpy as np
import os
from accumulate import accumulate
import h5py
import healpy as hp
from PyTrilinos import Epetra, AztecOO
from scipy import signal
import logging as l
l.root.level = l.DEBUG

from dst_operators import bin_map, TDestripeOperator, QUDestripeOperator, PrecOperator, CommMetadata
from utils import gal2eq, create_hitmap
from dst_io import read_data, MPIwrite

# read configuration
import sys
from ConfigParser import SafeConfigParser
config = SafeConfigParser()

if len(sys.argv) != 2:
    l.error("Usage: mpirun -np 3 python dst.py ch6_256.cfg")
config.read(sys.argv[1])

input_filename = config.get("dst", "input_filename")
sampling_frequency = config.getfloat("dst", "sampling_frequency")
bin_filtered = config.getboolean("dst", "bin_filtered")
nside = config.getint("dst", "nside")
BaselineLength = config.getint("dst", "BaselineLength")
gmres_iterations = config.getint("dst", "gmres_iterations")
gmres_residual = config.getfloat("dst", "gmres_residual")
scan_gal_input_map = config.get("dst", "scan_gal_input_map")

folder = "dst_out_%s_%d/" % (os.path.basename(input_filename).split('.')[0], nside)
npix = hp.nside2npix(nside)

# create the communicator object
comm = CommMetadata()

if comm.MyPID == 0:
    l.info("MPI procs %d" % comm.MPIcomm.NumProc())
    try:
        os.mkdir(folder)
    except:
        pass

# define data range
i_from = 0 
length = len(h5py.File(input_filename, mode='r')['data'])
# must be a multiple of baseline length
length /= BaselineLength
length *= BaselineLength
comm.create_global_map("bas", length/BaselineLength)

# temperature and polarization done in 2 different runs
for pol, comps in zip([False, True], ["T", "QU"]):

    # read data, pixels and baseline lengths in numpy arrays from hdf5 files
    pix, data, BaselineLengths = read_data(input_filename, 
                                           i_from + comm.maps["bas"].MinMyGID()*BaselineLength, 
                                           i_from + (comm.maps["bas"].MaxMyGID()+ 1)*BaselineLength, 
                                           nside, BaselineLength, comm, pol=pol, maskdestripe=True)

    # replace the measured signal with a simulated signal
    if scan_gal_input_map:
        gal_input_map = gal2eq(hp.read_map("test_rot_map.fits", [0,1,2]))
        if pol:
            data['Q'] = gal_input_map[1][pix] * data["q_channel_w"]['Q'] + gal_input_map[2][pix] *  data["q_channel_w"]['U']
            data['U'] = gal_input_map[1][pix] * data["u_channel_w"]['Q'] + gal_input_map[2][pix] *  data["u_channel_w"]['U']
        else:
            data['T'] = gal_input_map[0][pix] 

    l.info("Proc %d: Num Baselines %d per channel" % (comm.MyPID, comm.maps["bas"].NumMyElements()))

    NumBaselines = comm.maps["bas"].NumMyElements()

    if pol:
        # baselines for q and u channels
        comm.create_custom_global_map("bas", 2*NumBaselines)

    comm.create_global_map("pix", npix)
    comm.create_local_map("pix", np.unique(pix).tolist())

    l.info("Setting pixels local")
    for i in range(len(pix)):
        pix[i] = comm.maps["loc_pix"].LID(pix[i])

    hits_glob = create_hitmap(pix, comm)
    MPIwrite(folder + "hits.bin", hits_glob.array, comm)

    # all Epetra objects are distributed over the available procs
    tmap_local = Epetra.Vector(comm.maps["loc_pix"])
    tmap_glob = Epetra.Vector(comm.maps["pix"])

    # right hand side of the destriping equation
    RHS = Epetra.MultiVector(comm.maps["bas"], 1)
    baselines = Epetra.MultiVector(comm.maps["bas"], 1)

    # create binned maps -> signal removed streams -> RHS
    if pol:
        umap_local = Epetra.Vector(comm.maps["loc_pix"])
        umap_glob = Epetra.Vector(comm.maps["pix"])

        # bin filtered data
        if bin_filtered:
            dataf = {}
            filter_minutes = 60
            filter_hz = 1 / (filter_minutes * 60.)
            b, a = signal.butter(2, filter_hz / (sampling_frequency/2.), btype='high') # 3 hours
            for comp in "QU":
                dataf[comp] = signal.lfilter(b, a, data[comp])

            bin_map(pix, (dataf['Q'])*data['q_channel_w']['Q'] + (dataf['U'])*data['u_channel_w']['Q'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=True)
            bin_map(pix, (dataf['Q'])*data['q_channel_w']['U'] + (dataf['U'])*data['u_channel_w']['U'], umap_local, umap_glob, hits_glob, comm, broadcast_locally=True)
            tmap_glob[hits_glob == 0] = hp.UNSEEN
            umap_glob[hits_glob == 0] = hp.UNSEEN
            MPIwrite(folder + "binnedfQ.bin", tmap_glob.array, comm)
            MPIwrite(folder + "binnedfU.bin", umap_glob.array, comm)
        #/ bin filtered data

        bin_map(pix, data['Q']*data['q_channel_w']['Q'] + data['U']*data['u_channel_w']['Q'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=True)
        bin_map(pix, data['Q']*data['q_channel_w']['U'] + data['U']*data['u_channel_w']['U'], umap_local, umap_glob, hits_glob, comm, broadcast_locally=True)
        tmap_glob[hits_glob == 0] = hp.UNSEEN
        umap_glob[hits_glob == 0] = hp.UNSEEN
        MPIwrite(folder + "binnedQ.bin", tmap_glob.array, comm)
        MPIwrite(folder + "binnedU.bin", umap_glob.array, comm)

        signal_removed = {}
        signal_removed['Q'] = data['Q'] - tmap_local.array[pix] * data['q_channel_w']['Q'] - umap_local.array[pix] * data['q_channel_w']['U']
        signal_removed['U'] = data['U'] - tmap_local.array[pix] * data['u_channel_w']['Q'] - umap_local.array[pix] * data['u_channel_w']['U']

        assert len(BaselineLengths) == len(RHS.array[0][:NumBaselines])
        accumulate(signal_removed['Q'], BaselineLengths, RHS.array[0][:NumBaselines])
        assert len(BaselineLengths) == len(RHS.array[0][NumBaselines:])
        accumulate(signal_removed['U'], BaselineLengths, RHS.array[0][NumBaselines:])
    else:
        bin_map(pix, data['T'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=True)
        tmap_glob[hits_glob == 0] = hp.UNSEEN
        MPIwrite(folder + "binned.bin", tmap_glob.array, comm)

        signal_removed = data['T'] - tmap_local.array[pix]
        accumulate(signal_removed, BaselineLengths, RHS.array[0])

    # Create the operator objects that provide the left hand side of the destriping equation
    if pol:
        DOp = QUDestripeOperator(pix, tmap_local, tmap_glob, umap_local, umap_glob, hits_glob, BaselineLengths, data, comm, NumBaselines)
    else:
        DOp = TDestripeOperator(pix, tmap_local, tmap_glob, hits_glob, BaselineLengths, comm)

    # Create the preconditioning operator
    PrecOp = PrecOperator(BaselineLengths, comm)

    # Aggregate the Destriping Operator, first guess and RHS in Linear Problem object
    LinProb = Epetra.LinearProblem(DOp, baselines, RHS)

    # Solve the problem iteratively using GMRES
    IterSolver = AztecOO.AztecOO(LinProb)
    IterSolver.SetPrecOperator(PrecOp)
    IterSolver.Iterate(gmres_iterations, gmres_residual)

    # Write the baselines to disk, remove baselines from data, bin and write destriped maps
    if pol:
        MPIwrite(folder + "baselinesQ.bin", baselines.array[0][:NumBaselines], comm)
        MPIwrite(folder + "baselinesU.bin", baselines.array[0][NumBaselines:], comm)

        data['Q'] -= np.repeat(baselines.array[0][:NumBaselines], BaselineLengths)
        data['U'] -= np.repeat(baselines.array[0][NumBaselines:], BaselineLengths)
        bin_map(pix, data['Q']*data['q_channel_w']['Q'] + data['U']*data['u_channel_w']['Q'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=False)
        bin_map(pix, data['Q']*data['q_channel_w']['U'] + data['U']*data['u_channel_w']['U'], umap_local, umap_glob, hits_glob, comm, broadcast_locally=False)
        tmap_glob[hits_glob == 0] = hp.UNSEEN
        umap_glob[hits_glob == 0] = hp.UNSEEN
        MPIwrite(folder + "mapQ.bin", tmap_glob.array, comm)
        MPIwrite(folder + "mapU.bin", umap_glob.array, comm)
    else:
        MPIwrite(folder + "baselines.bin", baselines.array[0], comm)
        bin_map(pix, data['T'] - np.repeat(baselines.array[0], BaselineLengths), tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=False)
        tmap_glob[hits_glob == 0] = hp.UNSEEN
        MPIwrite(folder + "map.bin", tmap_glob.array, comm)
