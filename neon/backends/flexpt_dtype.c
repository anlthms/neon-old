/* Flexpoint decimal number type for use within Numpy */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

/* default parameters assigned when not specified */
#define DEF_SIGNED 1
#define DEF_INT_BITS 5
#define DEF_FRAC_BITS 10
#define DEF_OVERFLOW OFL_SATURATE
#define DEF_ROUNDING RND_TRUNCATE

/* types of overflow handling supported */
typedef enum {
  OFL_SATURATE,
  OFL_WRAP
} ofl_t;

/* types of rounding supported */
typedef enum {
  RND_TRUNCATE,
  RND_NEAREST,
  RND_CEILING,
  RND_FLOOR,
  RND_ZERO
} rnd_t;

/* some useful constants - used to speed up various operations below */
static const int64_t bin_pow[33] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
    2147483548, 4294967296
};
static const int64_t ones[33] = {
    1, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383,
    32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607,
    16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823,
    2147483547, 4294967295
};

typedef struct {
    uint8_t sign_bit;
    uint8_t int_bits;
    uint8_t frac_bits;
    int64_t val;
    ofl_t overflow;
    rnd_t rounding;
}  flexpt;


static NPY_INLINE flexpt make_flexpt_double(double float_val, uint8_t sign_bit,
                                          uint8_t int_bits, uint8_t frac_bits,
                                          ofl_t overflow, rnd_t rounding) {
    flexpt fi;
    int word_len = sign_bit + int_bits + frac_bits;
    double int_val, uint_val;
    int64_t max_int;

    fi.sign_bit = sign_bit;
    fi.int_bits = int_bits;
    fi.frac_bits = frac_bits;
    fi.overflow = overflow;
    fi.rounding = rounding;

    if (word_len <= 32) {
        // 1. scale input up to raw binary integer representation
        int_val = float_val * bin_pow[frac_bits];
        uint_val = fabs(int_val);
        // 2. perform rounding/truncation
        if (uint_val < 4.503599627370496E+15) {
            // scaled number fits entirely within fraction part of double
            // (52 bits)
            if (rounding == RND_TRUNCATE) {
                int_val = floor(int_val);
            } else if (rounding == RND_NEAREST) {
                if (uint_val >= 0.5) {
                    int_val = floor(int_val + 0.5);
                } else {
                    int_val *= 0.0;
                }
            }
        }
        // 3. handle overflows and assign value
        max_int = 1 << (word_len - sign_bit);
        if (int_val < max_int) {
            if (int_val >= -max_int) {
                // number within range
                fi.val = (int64_t) int_val;
            } else {
                // negative overflow
                if (overflow == OFL_SATURATE) {
                    fi.val = (int64_t) -max_int;
                } else {
                    // OFL_WRAP (this is undefined for signed ints)
                    printf("undefined negative overflow wrapping");
                    fi.val = ((int64_t) int_val) & ones[word_len];
                }
            }
        } else if (int_val >= max_int) {
            // positive overflow
            if (overflow == OFL_SATURATE) {
                fi.val = (int64_t) (max_int - 1);
            } else {
                // OFL_WRAP
                fi.val = ((int64_t) int_val) & ones[word_len];
            }
        } else {
            // non-representable double val (like nan, inf, etc.)
            fi.val = 0;
        }
    } else {
        printf("TODO: add larger word lengths");
    }
    return fi;
}

static NPY_INLINE flexpt flexpt_add(flexpt x, flexpt y) {
    flexpt res;
    int64_t max_int;

    if (x.sign_bit == y.sign_bit && x.int_bits == y.int_bits &&
        x.frac_bits == y.frac_bits) {
        res.sign_bit = x.sign_bit;
        res.int_bits = x.int_bits;
        res.frac_bits = x.frac_bits;
        res.overflow = x.overflow;
        res.rounding = x.rounding;
        res.val = x.val + y.val;
        max_int = 1 << (res.int_bits + res.frac_bits + (1 - res.sign_bit));
        if (res.val >= max_int) {
            /* overflow */
            res.val = max_int - 1;
        } 
    } else {
        printf("TODO: add mixed fp type conversion");
    }
    return res;
}

static NPY_INLINE flexpt flexpt_subtract(flexpt x, flexpt y) {
    flexpt res;
    int64_t max_int;

    if (x.sign_bit == y.sign_bit && x.int_bits == y.int_bits &&
        x.frac_bits == y.frac_bits) {
        res.sign_bit = x.sign_bit;
        res.int_bits = x.int_bits;
        res.frac_bits = x.frac_bits;
        res.overflow = x.overflow;
        res.rounding = x.rounding;
        res.val = x.val - y.val;
        max_int = 1 << (res.int_bits + res.frac_bits + (1 - res.sign_bit));
        if (res.val >= max_int) {
            /* overflow */
            res.val = max_int - 1;
        } 
    } else {
        printf("TODO: add mixed fp type conversion");
    }
    return res;
}

static NPY_INLINE flexpt flexpt_multiply(flexpt x, flexpt y) {
    flexpt res;
    int64_t max_int;

    if (x.sign_bit == y.sign_bit && x.int_bits == y.int_bits &&
        x.frac_bits == y.frac_bits) {
        res.sign_bit = x.sign_bit;
        res.int_bits = x.int_bits;
        res.frac_bits = x.frac_bits;
        res.overflow = x.overflow;
        res.rounding = x.rounding;
        res.val = x.val * y.val / bin_pow[res.frac_bits];
        max_int = 1 << (res.int_bits + res.frac_bits + (1 - res.sign_bit));
        if (res.val >= max_int) {
            /* overflow */
            res.val = max_int - 1;
        } 
    } else {
        printf("TODO: add mixed fp type conversion");
    }
    return res;
}

static NPY_INLINE flexpt flexpt_divide(flexpt x, flexpt y) {
    flexpt res;
    int64_t max_int;

    if (x.sign_bit == y.sign_bit && x.int_bits == y.int_bits &&
        x.frac_bits == y.frac_bits) {
        res.sign_bit = x.sign_bit;
        res.int_bits = x.int_bits;
        res.frac_bits = x.frac_bits;
        res.overflow = x.overflow;
        res.rounding = x.rounding;
        res.val = x.val / y.val * bin_pow[res.frac_bits];
        max_int = 1 << (res.int_bits + res.frac_bits + (1 - res.sign_bit));
        if (res.val >= max_int) {
            /* overflow */
            res.val = max_int - 1;
        } 
    } else {
        printf("TODO: add mixed fp type conversion");
    }
    return res;
}
typedef struct {
    PyObject_HEAD;
    flexpt fi;
} PyFlexPt;

static PyTypeObject PyFlexPt_Type;

static NPY_INLINE int PyFlexPt_Check(PyObject* object) {
    return PyObject_IsInstance(object,(PyObject*)&PyFlexPt_Type);
}

static PyObject* PyFlexPt_FromFlexPt(flexpt x) {
    PyFlexPt* p = (PyFlexPt*) PyFlexPt_Type.tp_alloc(&PyFlexPt_Type, 0);
    if (p) {
        p->fi = x;
    }
    return (PyObject*) p;
}

static PyObject* pyflexpt_new(PyTypeObject* type, PyObject* args,
                             PyObject* kwds) {
    
    double val = 0.0;
    uint8_t sign_bit = DEF_SIGNED;
    uint8_t int_bits = DEF_INT_BITS;
    uint8_t frac_bits = DEF_FRAC_BITS;
    ofl_t overflow = DEF_OVERFLOW;
    rnd_t rounding = DEF_ROUNDING;
    char* kwnames[] = {"val", "sign_bit", "int_bits", "frac_bits", "overflow",
                       "rounding", NULL};
    flexpt fi;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|bbbbb", kwnames,
                                    &val, &sign_bit, &int_bits, &frac_bits,
                                    &overflow, &rounding)) {
        fi = make_flexpt_double(val, sign_bit, int_bits, frac_bits, overflow,
                               rounding);
        return PyFlexPt_FromFlexPt(fi);
    } else {
        PyErr_SetString(PyExc_TypeError, "Problems parsing constructor args");
        return 0;
    }
}

#define AS_FLEXPT(dst, object) \
    flexpt dst = {0}; \
    if (PyFlexPt_Check(object)) { \
        dst = ((PyFlexPt*)object)->fi; \
    } else { \
        double n_ = PyFloat_AsDouble(object); \
        if (n_ == -1.0  && PyErr_Occurred()) { \
            if (PyErr_ExceptionMatches(PyExc_TypeError)) { \
                PyErr_Clear(); \
                Py_INCREF(Py_NotImplemented); \
                return Py_NotImplemented; \
            } \
            return 0; \
        } \
        PyObject* y_ = PyFloat_FromDouble(n_); \
        if (! y_) { \
            return 0; \
        } \
        int eq_ = PyObject_RichCompareBool(object, y_, Py_EQ); \
        Py_DECREF(y_); \
        if (eq_ < 0) { \
            return 0; \
        } \
        if (! eq_) { \
            Py_INCREF(Py_NotImplemented); \
            return Py_NotImplemented; \
        } \
        dst = make_flexpt_double(n_, DEF_SIGNED, DEF_INT_BITS, DEF_FRAC_BITS, \
                                DEF_OVERFLOW, DEF_ROUNDING); \
    }

#define FIXPT_BINOP_2(name, exp) \
    static PyObject* pyflexpt_##name(PyObject* a, PyObject* b) { \
        AS_FLEXPT(x, a); \
        AS_FLEXPT(y, b); \
        flexpt z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return PyFlexPt_FromFlexPt(z); \
    }

#define FIXPT_BINOP(name) FIXPT_BINOP_2(name, flexpt_##name(x, y))
FIXPT_BINOP(add)
FIXPT_BINOP(subtract)
FIXPT_BINOP(multiply)
FIXPT_BINOP(divide)


/**
 * Write a decimal representation of the number into the buffer passed.
 * ex. "+4.53"
 */
static void flexpt_to_str(flexpt x, char buf[]) {

    size_t pos;

    /* we *may* need up to frac_bits precision after the decimal but typically
     * that isn't the case.  So we write that, then truncate trailing zeros.
     */
    sprintf(buf, "%+.*f", x.frac_bits < 1 ? 1 : x.frac_bits,
            (double) x.val / bin_pow[x.frac_bits]);
    pos = strlen(buf) - 1;
    while (pos > 0 && buf[pos] == '0' && buf[pos - 1] != '.') {
      buf[pos--] = '\0';
    }
}


static PyObject* pyflexpt_repr(PyObject* self) {
    flexpt x = ((PyFlexPt*)self)->fi;
    char buf[256];
    char val[64];

    flexpt_to_str(x, val);
    snprintf(buf, 256, "flexpt(%s, sign_bit=%d, int_bits=%d, frac_bits=%d, "
             "overflow=%d, rounding=%d)", val, x.sign_bit, x.int_bits,
             x.frac_bits, x.overflow, x.rounding);
    return PyUString_FromString(buf);
}

PyMethodDef module_methods[] = {
    {0}
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "flexpt_dtype",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

static PyNumberMethods pyflexpt_as_number = {
    pyflexpt_add,          /* nb_add */
    pyflexpt_subtract,     /* nb_subtract */
    pyflexpt_multiply,     /* nb_multiply */
    pyflexpt_divide,       /* nb_divide */
    0,    /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power */
    0,     /* nb_negative */
    0,     /* nb_positive */
    0,     /* nb_absolute */
    0,      /* nb_nonzero */
    0,                       /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
    0,                       /* nb_coerce */
    0,          /* nb_int */
    0,          /* nb_long */
    0,        /* nb_float */
    0,                       /* nb_oct */
    0,                       /* nb_hex */

    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
    0,                       /* nb_inplace_divide */
    0,                       /* nb_inplace_remainder */
    0,                       /* nb_inplace_power */
    0,                       /* nb_inplace_lshift */
    0,                       /* nb_inplace_rshift */
    0,                       /* nb_inplace_and */
    0,                       /* nb_inplace_xor */
    0,                       /* nb_inplace_or */

    0, /* nb_floor_divide */
    0,       /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
    0,                       /* nb_index */
};

static PyTypeObject PyFlexPt_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
#else
    PyObject_HEAD_INIT(&PyType_Type)
    0,                                        /* ob_size */
#endif
    "flexpt",                                /* tp_name */
    sizeof(PyFlexPt),                        /* tp_basicsize */
    0,                                        /* tp_itemsize */
    0,                                        /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    pyflexpt_repr,                           /* tp_repr */
    &pyflexpt_as_number,                     /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                                        /* tp_call */
    0,                           /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "flexpoint representation of real numbers",       /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                   /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    pyflexpt_new,                              /* tp_new */
    0,                                        /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_flexpt_dtype(void) {
#else
PyMODINIT_FUNC initflexpt_dtype(void) {
#endif

    PyObject *m;

    import_array();
    if (PyErr_Occurred()) {
#if defined(NPY_PY3K)
        return NULL;
#else
        return;
#endif
    }

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("flexpt_dtype", module_methods);
#endif

    if (!m) {
#if defined(NPY_PY3K)
        return NULL;
#else
        return;
#endif
    }

    /* add the new types */
    Py_INCREF(&PyFlexPt_Type);
    PyModule_AddObject(m, "flexpt", (PyObject*)&PyFlexPt_Type);

#if defined(NPY_PY3K)
    return m;
#endif
}
