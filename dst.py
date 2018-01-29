#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
from accumulate import accumulate
import h5py
import healpy as hp
from PyTrilinos import Epetra, AztecOO
import logging as l
import exceptions
l.basicConfig(format = "%(asctime)-15s - %(levelname)s:%(name)s:%(message)s",
              level = l.DEBUG)

from dst_operators import bin_map, TDestripeOperator, QUDestripeOperator, PrecOperator, CommMetadata
from utils import gal2eq, create_hitmap
from dst_io import read_data, MPIwriter
import timemonitor as tm
from signalremove import signalremove, signalremovet

# read configuration
import sys
from ConfigParser import SafeConfigParser
config = SafeConfigParser(dict(output_tag="", max_data_samples=None, scan_gal_input_map=None, bin_filetered=False))

if len(sys.argv) != 2:
    l.error("Usage: mpirun -np 3 python dst.py ch6_256.cfg")
config.read(sys.argv[1])

input_filename = config.get("dst", "input_filename")
mask_filename = config.get("dst", "mask_filename")
sampling_frequency = config.getfloat("dst", "sampling_frequency")
bin_filtered = config.getboolean("dst", "bin_filtered")
nside = config.getint("dst", "nside")
max_data_samples = config.get("dst", "max_data_samples")
baseline_length = config.getint("dst", "baseline_length")
gmres_residual = config.getfloat("dst", "gmres_residual")
gmres_iterations = config.getint("dst", "gmres_iterations")
scan_gal_input_map = config.get("dst", "scan_gal_input_map")
output_tag = config.get("dst", "output_tag")

if max_data_samples:
    max_data_samples = int(max_data_samples)
else:
    max_data_samples = None

folder_components = ["dst", "out", os.path.basename(input_filename).split('.')[0]]
if output_tag:
    folder_components.append(output_tag)
folder_components.append(str(nside))
folder = "_".join(folder_components) + "/"

npix = hp.nside2npix(nside)

# create the communicator object
comm = CommMetadata()

mpi_writer = MPIwriter(folder, comm)

tm.set_mypid(comm.MyPID)

if comm.MyPID == 0:
    l.info("MPI procs %d" % comm.MPIcomm.NumProc())
    try:
        os.mkdir(folder)
    except:
        pass
    # write copy of config file to output folder
    with open(os.path.join(folder, "config.cfg"), "w") as config_output_file:
        config.write(config_output_file)

# define data range
i_from = 0 
length = max_data_samples or len(h5py.File(input_filename, mode='r')['data'])
# must be a multiple of baseline length
length /= baseline_length
length *= baseline_length
comm.create_global_map("bas", length/baseline_length)

# temperature and polarization done in 2 different runs
for pol, comps in zip([False, True], ["T", "QU"]):
    with tm.TimeMonitor("Total"):
        with tm.TimeMonitor("Read input data"):
            # read data, pixels and baseline lengths in numpy arrays from hdf5 files
            pix, data, baseline_lengths = read_data(input_filename, 
                                                   i_from + comm.maps["bas"].MinMyGID()*baseline_length, 
                                                   i_from + (comm.maps["bas"].MaxMyGID()+ 1)*baseline_length, 
                                                   nside, baseline_length, comm, pol=pol, maskdestripe=mask_filename)

        # replace the measured signal with a simulated signal
        if scan_gal_input_map:
            gal_input_map = gal2eq(hp.read_map(scan_gal_input_map, [0,1,2]))
            if pol:
                data['Q'] = gal_input_map[1][pix] * data["q_channel_w"]['Q'] + gal_input_map[2][pix] *  data["q_channel_w"]['U']
                data['U'] = gal_input_map[1][pix] * data["u_channel_w"]['Q'] + gal_input_map[2][pix] *  data["u_channel_w"]['U']
            else:
                data['T'] = gal_input_map[0][pix] 

        l.info("Proc %d: Num Baselines %d per channel" % (comm.MyPID, comm.maps["bas"].NumMyElements()))

        num_baselines = comm.maps["bas"].NumMyElements()

        if pol:
            # baselines for q and u channels
            comm.create_custom_global_map("bas", 2*num_baselines)

        comm.create_global_map("pix", npix)
        glob_pix = np.unique(pix).tolist()
        comm.create_local_map("pix", glob_pix)

        l.info("Creating pixel dict")
        pix_dict = {}
        for i in range(len(glob_pix)):
             pix_dict[glob_pix[i]] = comm.maps["loc_pix"].LID(glob_pix[i])

        l.info("Setting pixels local")
        with tm.TimeMonitor("Set pixels local"):
            l.info("Total pixels %d" % len(pix))
            for i in range(len(pix)):
                pix[i] = pix_dict[pix[i]] 

        l.info("Hitmap")
        hits_glob = create_hitmap(pix, comm)
        mpi_writer.write("hits.bin", hits_glob.array)

        # all Epetra objects are distributed over the available procs
        tmap_local = Epetra.Vector(comm.maps["loc_pix"])
        tmap_glob = Epetra.Vector(comm.maps["pix"])

        # right hand side of the destriping equation
        RHS = Epetra.MultiVector(comm.maps["bas"], 1)
        baselines = Epetra.MultiVector(comm.maps["bas"], 1)

        # create binned maps -> signal removed streams -> RHS
        with tm.TimeMonitor("Create RHS"):
            l.info("Create RHS")
            if pol:
                umap_local = Epetra.Vector(comm.maps["loc_pix"])
                umap_glob = Epetra.Vector(comm.maps["pix"])

                # bin filtered data
                if bin_filtered:
                    from scipy import signal
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
                    mpi_writer.write("binnedfQ.bin", tmap_glob.array)
                    mpi_writer.write("binnedfU.bin", umap_glob.array)
                #/ bin filtered data

                l.info("Bin maps")
                bin_map(pix, data['Q']*data['q_channel_w']['Q'] + data['U']*data['u_channel_w']['Q'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=True)
                bin_map(pix, data['Q']*data['q_channel_w']['U'] + data['U']*data['u_channel_w']['U'], umap_local, umap_glob, hits_glob, comm, broadcast_locally=True)
                tmap_glob[hits_glob == 0] = hp.UNSEEN
                umap_glob[hits_glob == 0] = hp.UNSEEN
                l.info("Write maps")
                mpi_writer.write("binnedQ.bin", tmap_glob.array)
                mpi_writer.write("binnedU.bin", umap_glob.array)

                l.info("Signal remove")
                signal_removed = {
                'Q' : np.zeros(len(data['Q']), dtype=np.double),
                'U' : np.zeros(len(data['U']), dtype=np.double),
                }
                try:
                    signalremove(signal_removed['Q'], signal_removed['U'], data['Q'].astype(np.float32), data['U'].astype(np.float32), tmap_local.array, umap_local.array, data['q_channel_w']['Q'], data['q_channel_w']['U'], data['u_channel_w']['Q'], data['u_channel_w']['U'], pix)
                except exceptions.ValueError: # byteswap in case input h5 has wrong endianness
                    signalremove(signal_removed['Q'], signal_removed['U'], data['Q'].byteswap().newbyteorder().astype(np.float32), data['U'].byteswap().newbyteorder().astype(np.float32), tmap_local.array, umap_local.array, data['q_channel_w']['Q'], data['q_channel_w']['U'], data['u_channel_w']['Q'], data['u_channel_w']['U'], pix)

                assert len(baseline_lengths) == len(RHS.array[0][:num_baselines])
                accumulate(signal_removed['Q'], baseline_lengths, RHS.array[0][:num_baselines])
                assert len(baseline_lengths) == len(RHS.array[0][num_baselines:])
                accumulate(signal_removed['U'], baseline_lengths, RHS.array[0][num_baselines:])
            else:
                l.info("Bin maps")
                bin_map(pix, data['T'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=True)
                tmap_glob[hits_glob == 0] = hp.UNSEEN
                l.info("Write maps")
                mpi_writer.write("binned.bin", tmap_glob.array)

                l.info("Remove signal")
                signal_removed = np.zeros(len(data['T']), dtype=np.double)
                try:
                    signalremovet(signal_removed, data['T'].astype(np.float32), tmap_local.array, pix)
                except exceptions.ValueError: # byteswap in case input h5 has wrong endianness
                    signalremovet(signal_removed, data['T'].byteswap().newbyteorder(), tmap_local.array, pix)

                l.info("Accumulate")
                accumulate(signal_removed, baseline_lengths, RHS.array[0])

        l.info("Create destripe operator")
        # Create the operator objects that provide the left hand side of the destriping equation
        if pol:
            DOp = QUDestripeOperator(pix, tmap_local, tmap_glob, umap_local, umap_glob, hits_glob, baseline_lengths, data, comm, num_baselines)
        else:
            DOp = TDestripeOperator(pix, tmap_local, tmap_glob, hits_glob, baseline_lengths, comm)

        # Create the preconditioning operator
        PrecOp = PrecOperator(baseline_lengths, comm)

        # Aggregate the Destriping Operator, first guess and RHS in Linear Problem object
        LinProb = Epetra.LinearProblem(DOp, baselines, RHS)

        # Solve the problem iteratively using GMRES
        IterSolver = AztecOO.AztecOO(LinProb)
        IterSolver.SetPrecOperator(PrecOp)
        with tm.TimeMonitor("Iterations"):
            IterSolver.Iterate(gmres_iterations, gmres_residual)

        # Write the baselines to disk, remove baselines from data, bin and write destriped maps

        with tm.TimeMonitor("After destriping"):
            if pol:
                mpi_writer.write("baselinesQ.bin", baselines.array[0][:num_baselines])
                mpi_writer.write("baselinesU.bin", baselines.array[0][num_baselines:])

                data['Q'] -= np.repeat(baselines.array[0][:num_baselines], baseline_lengths)
                data['U'] -= np.repeat(baselines.array[0][num_baselines:], baseline_lengths)
                bin_map(pix, data['Q']*data['q_channel_w']['Q'] + data['U']*data['u_channel_w']['Q'], tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=False)
                bin_map(pix, data['Q']*data['q_channel_w']['U'] + data['U']*data['u_channel_w']['U'], umap_local, umap_glob, hits_glob, comm, broadcast_locally=False)
                tmap_glob[hits_glob == 0] = hp.UNSEEN
                umap_glob[hits_glob == 0] = hp.UNSEEN
                mpi_writer.write("mapQ.bin", tmap_glob.array)
                mpi_writer.write("mapU.bin", umap_glob.array)
            else:
                mpi_writer.write("baselines.bin", baselines.array[0])
                bin_map(pix, data['T'] - np.repeat(baselines.array[0], baseline_lengths), tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=False)
                tmap_glob[hits_glob == 0] = hp.UNSEEN
                mpi_writer.write("map.bin", tmap_glob.array)

    print tm.summarize()
    tm.reset()
