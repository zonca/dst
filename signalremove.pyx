cimport numpy as np

ctypedef np.long_t DTYPE_int_t
ctypedef np.double_t DTYPE_double_t
ctypedef np.float_t DTYPE_float_t

cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def signalremove(
       np.ndarray[DTYPE_double_t, ndim=1] out_qsignal not None,
       np.ndarray[DTYPE_double_t, ndim=1] out_usignal not None,
       np.ndarray[float, ndim=1] qsignal not None,
       np.ndarray[float, ndim=1] usignal not None,
       np.ndarray[DTYPE_double_t, ndim=1] qmap not None,
       np.ndarray[DTYPE_double_t, ndim=1] umap not None,
       np.ndarray[DTYPE_double_t, ndim=1] q_channel_w_q not None,
       np.ndarray[DTYPE_double_t, ndim=1] q_channel_w_u not None,
       np.ndarray[DTYPE_double_t, ndim=1] u_channel_w_q not None,
       np.ndarray[DTYPE_double_t, ndim=1] u_channel_w_u not None,
       np.ndarray[DTYPE_int_t, ndim=1] pix not None,
       ): 
    cdef unsigned int nsamp, i
    nsamp = pix.shape[0]
    for i in range(nsamp):
        out_qsignal[i] = qsignal[i] - qmap[pix[i]] * q_channel_w_q[i] - umap[pix[i]] * q_channel_w_u[i]
        out_usignal[i] = usignal[i] - qmap[pix[i]] * u_channel_w_q[i] - umap[pix[i]] * u_channel_w_u[i]
    return 0

cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def signalremovei(
       np.ndarray[DTYPE_double_t, ndim=1] qsignal not None,
       np.ndarray[DTYPE_double_t, ndim=1] usignal not None,
       np.ndarray[DTYPE_double_t, ndim=1] qmap not None,
       np.ndarray[DTYPE_double_t, ndim=1] umap not None,
       np.ndarray[DTYPE_double_t, ndim=1] q_channel_w_q not None,
       np.ndarray[DTYPE_double_t, ndim=1] q_channel_w_u not None,
       np.ndarray[DTYPE_double_t, ndim=1] u_channel_w_q not None,
       np.ndarray[DTYPE_double_t, ndim=1] u_channel_w_u not None,
       np.ndarray[DTYPE_int_t, ndim=1] pix not None,
       ): 
    cdef unsigned int nsamp, i
    nsamp = pix.shape[0]
    for i in range(nsamp):
        qsignal[i] -= qmap[pix[i]] * q_channel_w_q[i] + umap[pix[i]] * q_channel_w_u[i]
        usignal[i] -= qmap[pix[i]] * u_channel_w_q[i] + umap[pix[i]] * u_channel_w_u[i]
    return 0

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def signalremovet(
       np.ndarray[DTYPE_double_t, ndim=1] out_tsignal not None,
       np.ndarray[float, ndim=1] tsignal not None,
       np.ndarray[DTYPE_double_t, ndim=1] tmap not None,
       np.ndarray[DTYPE_int_t, ndim=1] pix not None,
       ): 
    cdef unsigned int nsamp, i
    nsamp = pix.shape[0]
    for i in range(nsamp):
        out_tsignal[i] = tsignal[i] - tmap[pix[i]]
    return 0

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def signalremoveti(
       np.ndarray[DTYPE_double_t, ndim=1] tsignal not None,
       np.ndarray[DTYPE_double_t, ndim=1] tmap not None,
       np.ndarray[DTYPE_int_t, ndim=1] pix not None,
       ): 
    cdef unsigned int nsamp, i
    nsamp = pix.shape[0]
    for i in range(nsamp):
        tsignal[i] -= tmap[pix[i]]
    return 0
