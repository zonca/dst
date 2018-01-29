import logging as l
import numpy as np
from accumulate import accumulate
from signalremove import signalremovei, signalremoveti
from PyTrilinos import Epetra
import timemonitor as tm

def bin_map(pix, tod, tmap_local, tmap_glob, hits_glob, comm, broadcast_locally=False):

    tmap_local[:] = 0
    tmap_glob[:] = 0
    l.info("bin_map: bincount")
    tmap_local[:] = np.bincount(pix, weights=tod)

    l.info("bin_map: loc 2 glob")
    comm.pix_local_to_global(tmap_local, tmap_glob)
    tmap_glob[hits_glob > 0] /= hits_glob[hits_glob > 0]

    if broadcast_locally:
        l.info("bin_map: glob 2 loc")
        comm.pix_global_to_local(tmap_glob, tmap_local)

class TDestripeOperator(Epetra.Operator):
    """Temperature only destriping
    
    Apply does the left side of the destriping equation"""

    def __init__(self, pix, tmap_local, tmap_glob, hits_glob, BaselineLengths, comm):
        Epetra.Operator.__init__(self)
        self.__label = "TDestripeOperator"
        self.pix = pix
        self.tmap_local = tmap_local
        self.tmap_glob = tmap_glob
        self.hits_glob = hits_glob
        self.comm = comm
        self.Map = comm.maps["bas"]
        self.BaselineLengths = BaselineLengths

    def Label(self):
        return self.__label

    def RealApply(self, x, y):
        with tm.TimeMonitor("Destriping Local Operations"):
            sig = np.repeat(x.array[0], self.BaselineLengths)
            self.tmap_local[:]=0
        with tm.TimeMonitor("Destriping Comm Operations"):
            bin_map(self.pix, sig, self.tmap_local, self.tmap_glob, self.hits_glob, self.comm, broadcast_locally=True)
        with tm.TimeMonitor("Destriping Local Operations"):
            signalremoveti(sig, self.tmap_local.array, self.pix)
            accumulate(sig, self.BaselineLengths, y.array[0])
        return 0

    def Apply(self, x, y):
        try:
           return self.RealApply(x, y)
        except Exception, e:
            l.error("A python exception was raised in %s:" % self.__label)
            print e
            return -1

    def Comm(self):
        return self.comm.MPIcomm
    def OperatorRangeMap(self):
        return self.Map
    def OperatorDomainMap(self):
        return self.Map
    def HasNormInf(self):
        return False

class QUDestripeOperator(TDestripeOperator):
    def __init__(self, pix, tmap_local, tmap_glob, umap_local, umap_glob, hits_glob, BaselineLengths, data, comm, NumBaselines):
        Epetra.Operator.__init__(self)
        self.__label = "QUDestripeOperator"
        self.pix = pix
        self.tmap_local = tmap_local
        self.tmap_glob = tmap_glob
        self.umap_local = umap_local
        self.umap_glob =  umap_glob
        self.hits_glob = hits_glob
        self.comm = comm
        self.Map = comm.maps["bas"]
        self.BaselineLengths = BaselineLengths
        self.length = len(pix)
        self.q_channel_w = data['q_channel_w']
        self.u_channel_w = data['u_channel_w']
        self.NumBaselines = NumBaselines

    def RealApply(self, x, y):
        # baseline to tod
        sig = {}
        with tm.TimeMonitor("Destriping Local Operations"):
            sig['Q'] = np.repeat(x.array[0][:self.NumBaselines], self.BaselineLengths)
            sig['U'] = np.repeat(x.array[0][self.NumBaselines:], self.BaselineLengths)
        self.SignalRemove(sig)
        # tod to baseline
        with tm.TimeMonitor("Destriping Local Operations"):
            accumulate(sig['Q'], self.BaselineLengths, y.array[0][:self.NumBaselines])
            accumulate(sig['U'], self.BaselineLengths, y.array[0][self.NumBaselines:])
        return 0

    def SignalRemove(self, sig):
        with tm.TimeMonitor("Destriping Comm Operations"):
            # bin maps
            bin_map(self.pix, sig['Q']*self.q_channel_w['Q'] + sig['U']*self.u_channel_w['Q'], self.tmap_local, self.tmap_glob, self.hits_glob, self.comm, broadcast_locally=True)
            bin_map(self.pix, sig['Q']*self.q_channel_w['U'] + sig['U']*self.u_channel_w['U'], self.umap_local, self.umap_glob, self.hits_glob, self.comm, broadcast_locally=True)
        # signal remove
        #tmap_tod = self.tmap_local[pix]
        #umap_tod = self.umap_local[pix]
        #sig['Q'] -= tmap_tod * self.q_channel_w['Q'] + umap_tod * self.q_channel_w['U']
        #sig['U'] -= tmap_tod * self.u_channel_w['Q'] + umap_tod * self.u_channel_w['U']
        with tm.TimeMonitor("Destriping Local Operations"):
            signalremovei(sig['Q'], sig['U'], self.tmap_local.array, self.umap_local.array, self.q_channel_w['Q'], self.q_channel_w['U'], self.u_channel_w['Q'], self.u_channel_w['U'], self.pix)

    def Label(self):
        return self.__label


class PrecOperator(TDestripeOperator):
    """Preconditioning operator

    just divides the accumulated quantity by the number of samples per baseline"""

    def ApplyInverse(self, x, y):
        return self.Apply(x,y)

    def __init__(self, SamplesPerBaselines, comm):
        Epetra.Operator.__init__(self)
        self.SamplesPerBaselines = SamplesPerBaselines.copy()
        self.DoubleSamplesPerBaselines = np.concatenate([SamplesPerBaselines, SamplesPerBaselines])
        #remove zeros
        for b in [self.SamplesPerBaselines, self.DoubleSamplesPerBaselines]:
            b[b == 0] = 1
        self.__label = "PrecOperator"
        self.Map = comm.maps["bas"]
        self.comm = comm

    def Label(self):
        return self.__label

    def RealApply(self, x, y):
        if len(x.array[0]) == len(self.SamplesPerBaselines):
            y[0, :] = x[0, :] / self.SamplesPerBaselines
        else:
            y[0, :] = x[0, :] / self.DoubleSamplesPerBaselines
        return 0

class CommMetadata:
    """MPI Communication metadata

    Stores data distributions and Export objects
    for communication
    """
    def __init__(self):
        self.MPIcomm = Epetra.PyComm()
        self.MyPID = self.MPIcomm.MyPID()
        self.maps = {}

    def create_global_map(self, name, numelements):
        self.maps[name] = Epetra.Map(numelements, 0, self.MPIcomm)

    def create_custom_global_map(self, name, elements):
        self.maps[name] = Epetra.Map(-1, elements, 0, self.MPIcomm)

    def create_local_map(self, name, elements):
        self.maps["loc_" + name] = Epetra.Map(-1, elements, 0, self.MPIcomm)
        if name == "pix":
            self.exporter = Epetra.Export(self.maps["loc_pix"], self.maps["pix"])

    def pix_local_to_global(self, localvector, globalvector):
        globalvector[:] = 0
        with tm.TimeMonitor("Comm loc->glob"):
            globalvector.Export(localvector, self.exporter, Epetra.Add)

    def pix_global_to_local(self, globalvector, localvector):
        with tm.TimeMonitor("Comm glob->loc"):
            localvector.Import(globalvector, self.exporter, Epetra.Insert)
