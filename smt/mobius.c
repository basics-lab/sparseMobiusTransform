#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *mobius(PyObject *self, PyObject *args) {
    PyArrayObject *array;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return NULL;
    }

    // Check if the input is a NumPy array and is contiguous
    if (!PyArray_Check(array) || !PyArray_ISCONTIGUOUS(array)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a contiguous NumPy array.");
        return NULL;
    }

    // Get pointers to the data
    double *data = (double *)PyArray_DATA(array);
    npy_intp size = PyArray_SIZE(array);
    int i = 1;
    int idx=0;
    while (i < size){
        for (npy_intp j = 0; j < size/2; ++j) {
            data[idx + i] -= data[idx];
            idx += (((idx % (2*i)) == (i-1)) ? i+1 : 1);
        }
        i*= 2;
        idx=0;
    }
    // Return None
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *inversemobius(PyObject *self, PyObject *args) {
    PyArrayObject *array;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return NULL;
    }

    // Check if the input is a NumPy array and is contiguous
    if (!PyArray_Check(array) || !PyArray_ISCONTIGUOUS(array)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a contiguous NumPy array.");
        return NULL;
    }

    // Get pointers to the data
    double *data = (double *)PyArray_DATA(array);
    npy_intp size = PyArray_SIZE(array);
    int i = 1;
    int idx=0;
    while (i < size){
        for (npy_intp j = 0; j < size/2; ++j) {
            data[idx + i] += data[idx];
            idx += (((idx % (2*i)) == (i-1)) ? i+1 : 1);
        }
        i*= 2;
        idx=0;
    }
    // Return None
    Py_INCREF(Py_None);
    return Py_None;
}

// Method definitions
static PyMethodDef methods[] = {
    {"mobius", mobius, METH_VARARGS, "Compute the Mobius Transform of an array."},
    {"inversemobius", inversemobius, METH_VARARGS, "Compute the Inverse Mobius Transform of an array."},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module initialization function
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mobiusmodule",  // Module name
    NULL,           // Module docstring
    -1,             // Size of per-interpreter state or -1 if stateless
    methods         // Method definitions
};

PyMODINIT_FUNC PyInit_mobiusmodule(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&moduledef);
}
