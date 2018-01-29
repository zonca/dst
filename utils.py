import healpy as hp
import numpy as np
from PyTrilinos import Epetra

def gal2eq(m):
    Rgal2eq = hp.Rotator(coord='CG')
    npix = len(m[0])
    nside = hp.npix2nside(npix)
    newpix = hp.vec2pix(nside, *Rgal2eq(hp.pix2vec(nside, np.arange(npix))))
    return [m[0][newpix], m[1][newpix], m[2][newpix]]

def eq2gal(m):
    Rgal2eq = hp.Rotator(coord='GC')
    npix = len(m[0])
    nside = hp.npix2nside(npix)
    newpix = hp.vec2pix(nside, *Rgal2eq(hp.pix2vec(nside, np.arange(npix))))
    return [mm[newpix] for mm in m]

def pix2map(pix, tod=None, dividebyhits=True):
    """Pixel array to hitmap, if TOD with same lenght of PIX is provided, 
    it is binned to a map"""
    pix = pix.astype(np.int)
    ids = np.bincount(pix, weights=None)
    hitmap = ids
    if tod is None:
        return hitmap
    else:
        ids_binned = np.bincount(pix, weights=tod)
        binned = ids_binned
        if dividebyhits:
            binned /= hitmap
        return hitmap, binned

def create_hitmap(pix, comm):
    hits=pix2map(pix)
    hits_local = Epetra.Vector(comm.maps["loc_pix"], hits.astype(np.double))
    hits_glob = Epetra.Vector(comm.maps["pix"])
    comm.pix_local_to_global(hits_local, hits_glob)
    return hits_glob

