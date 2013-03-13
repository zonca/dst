cimport numpy as np

ctypedef np.long_t DTYPE_int_t
ctypedef np.double_t DTYPE_double_t

cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def accumulate(
       np.ndarray[DTYPE_double_t, ndim=1] a not None,
       np.ndarray[DTYPE_int_t, ndim=1] section_lengths not None,
       np.ndarray[DTYPE_double_t, ndim=1] out not None,
       ): 
    cdef unsigned int i_el, i_bas, sec_length, lenout
    cdef double tmp
    lenout = out.shape[0] 
    i_el = 0
    for i_bas in range(lenout):
        tmp = 0
        for sec_length in range(section_lengths[i_bas]):
            tmp += a[i_el]
            i_el+=1
        out[i_bas] = tmp
    return 0
