# cython: language_level=3
from cpython.ref cimport PyObject

cdef extern from *:
    """
    // We define a proxy struct that maps identically to PyAV's Frame struct.
    // Because PyAV uses cdef methods, Cython injects a vtable pointer 
    // immediately before the first declared attribute.
    typedef struct {
        PyObject_HEAD
        void *vtab;  // <-- THE FIX: Cython's hidden virtual table pointer
        void *ptr;   // The actual AVFrame *
    } PyAVFrameProxy;
    """
    ctypedef struct PyAVFrameProxy:
        void *vtab
        void *ptr

def get_avframe_address(obj):
    """
    Casts the PyAV Frame object to our proxy struct to extract the raw C pointer.
    """
    cdef PyAVFrameProxy *proxy = <PyAVFrameProxy *><PyObject *>obj
    return <size_t>proxy.ptr