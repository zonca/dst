import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

lim = { '': {'bin':20, 'destriped':20, 'baselines':10, 'filtbin':1},
        'Q':{'bin':500, 'destriped':4, 'baselines':500, 'filtbin':10} }
lim['U'] = lim['Q']
smooth = False
maps = {}
folder = sys.argv[1] + "/"
hits = np.array(np.memmap(folder + "hits.bin",dtype=np.double))
hp.write_map(folder + "hits.fits", hits)
comps = ['','Q','U']
binned = [hp.ma(np.array(np.memmap(folder + "binned%s.bin" % comp, dtype=np.double))) for comp in comps]
m = [hp.ma(np.array(np.memmap(folder + "map%s.bin" % comp, dtype=np.double))) for comp in comps]
baselines = [bb-mm for bb,mm in zip(binned, m)]
mapcombs = [(binned, 'bin'),  (m, 'destriped'), (baselines, 'baselines')]

try:
    binnedf = [binned[0].copy()]
    binnedf += [hp.ma(np.array(np.memmap(folder + "binnedf%s.bin" % comp, dtype=np.double))) for comp in comps[1:]]
    mapcombs.append((binnedf, 'filtbin'))
except:
    pass

hp.mollview(hits, unit='hitcount',xsize=2000)
plt.savefig(folder + "hits.png")

for iqumap, name in mapcombs:
    for comp, what in zip(comps, iqumap):
        l = lim[comp][name]
        what -= what.mean()
        if comp == '':
            what *=-1
        if smooth:
            whatsm = hp.ma(hp.smoothing(what, fwhm=np.radians(40/60.)))
            whatsm.mask = what.mask
            what = whatsm
            l /= 2
        maps[comp] = what
        hp.mollview((what*1e3), min=-l,max=l, unit='mK',title="",xsize=2000)
        plt.savefig(folder + "%s%s.png" % (name, comp))

    hp.write_map(folder + "%s.fits" % (name), iqumap)
