#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include "numpy/arrayobject.h"
#include "clipper.hpp"
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif



inline int round_to_int(float r) {
    return (int)lrint(r);
}


static PyObject* c_star_dist (PyObject *self, PyObject *args) {

    PyArrayObject *src = NULL;
    PyArrayObject *dst = NULL;
    int n_rays;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &src, &n_rays))
        return NULL;

    npy_intp *dims = PyArray_DIMS(src);

    npy_intp dims_dst[3];
    dims_dst[0] = dims[0];
    dims_dst[1] = dims[1];
    dims_dst[2] = n_rays;

    dst = (PyArrayObject*)PyArray_SimpleNew(3,dims_dst,NPY_FLOAT32);

    # pragma omp parallel for schedule(dynamic)
    for (int i=0; i<dims[0]; i++) {
        for (int j=0; j<dims[1]; j++) {
            const unsigned short value = *(unsigned short *)PyArray_GETPTR2(src,i,j);
            // background pixel
            if (value == 0) {
                for (int k = 0; k < n_rays; k++) {
                    *(float *)PyArray_GETPTR3(dst,i,j,k) = 0;
                }
            // foreground pixel
            } else {
                const float st_rays = (2*M_PI) / n_rays; // step size for ray angles
                for (int k = 0; k < n_rays; k++) {
                    const float phi = k*st_rays;
                    const float dy = cos(phi);
                    const float dx = sin(phi);
                    float x = 0, y = 0;
                    // move along ray
                    while (1) {
                        x += dx;
                        y += dy;
                        const int ii = round_to_int(i+x), jj = round_to_int(j+y);
                        // stop if out of bounds or reaching a pixel with a different value/id
                        if (ii < 0 || ii >= dims[0] ||
                            jj < 0 || jj >= dims[1] ||
                            value != *(unsigned short *)PyArray_GETPTR2(src,ii,jj))
                        {
                            const float dist = sqrt(x*x + y*y);
                            *(float *)PyArray_GETPTR3(dst,i,j,k) = dist;
                            break;
                        }
                    }

                }
            }

        }
    }

    return PyArray_Return(dst);
}



inline float area_from_path(ClipperLib::Path p) {

    float area = 0;
    const int n = p.size();
    for (int i = 0; i<n; i++) {
        area += p[i].X * p[(i+1)%n].Y -
                p[i].Y * p[(i+1)%n].X;
    }
    area = 0.5 * abs(area);
    return area;
}


// polys.shape = (n_polys, 2, n_rays)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept

static PyObject* c_non_max_suppression_inds (PyObject *self, PyObject *args) {

    PyArrayObject *polys=NULL, *scores=NULL;
    PyArrayObject *result=NULL;
    float threshold;
    bool use_bbox;

    if (!PyArg_ParseTuple(args, "O!O!fp", &PyArray_Type, &polys, &PyArray_Type, &scores, &threshold, &use_bbox))
        return NULL;

    npy_intp *dims = PyArray_DIMS(polys);

    const int n_polys = dims[0];
    const int n_rays = dims[2];

    int * bbox_x1 = new int[n_polys];
    int * bbox_x2 = new int[n_polys];
    int * bbox_y1 = new int[n_polys];
    int * bbox_y2 = new int[n_polys];

    float * areas = new float[n_polys];
    bool * suppressed = new bool[n_polys];
    ClipperLib::Path * poly_paths = new ClipperLib::Path[n_polys];

    //initialize indices
    #pragma omp parallel for
    for (int i=0; i<n_polys; i++) {
        suppressed[i] = false;
    }

    // build polys and areas
    #pragma omp parallel for
    for (int i=0; i<n_polys; i++) {
        ClipperLib::Path clip;
        // build clip poly and bounding boxes
        for (int k =0; k<n_rays; k++) {
            int x = *(int *)PyArray_GETPTR3(polys,i,0,k);
            int y = *(int *)PyArray_GETPTR3(polys,i,1,k);
            if (k==0) {
                bbox_x1[i] = x;
                bbox_x2[i] = x;
                bbox_y1[i] = y;
                bbox_y2[i] = y;
            } else {
                bbox_x1[i] = (x<bbox_x1[i])?x:bbox_x1[i];
                bbox_x2[i] = (x>bbox_x2[i])?x:bbox_x2[i];
                bbox_y1[i] = (y<bbox_y1[i])?y:bbox_y1[i];
                bbox_y2[i] = (y>bbox_y2[i])?y:bbox_y2[i];
            }
            clip<<ClipperLib::IntPoint(x,y);
        }
        poly_paths[i] = clip;
        areas[i] = area_from_path(clip);
    }

    // suppress (double loop)
    for (int i=0; i<n_polys-1; i++) {

        if (suppressed[i])
           continue;

        #pragma omp parallel for schedule(dynamic)
        for (int j=i+1; j<n_polys; j++) {

            if (suppressed[j])
                continue;

            // first check if bounding boxes are intersecting
            if (use_bbox) {
                bool bbox_inter = !( bbox_x1[j] > bbox_x2[i] || bbox_x2[j] < bbox_x1[i] ||
                                     bbox_y1[j] > bbox_y2[i] || bbox_y2[j] < bbox_y1[i]);
                if (!bbox_inter)
                    continue;
            }

            ClipperLib::Clipper c;
            ClipperLib::Paths res;
            c.Clear();

            c.AddPath(poly_paths[i],ClipperLib::ptClip, true);
            c.AddPath(poly_paths[j],ClipperLib::ptSubject, true);
            c.Execute(ClipperLib::ctIntersection, res, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

            float area_inter = 0;
            for (unsigned int r=0; r<res.size(); r++)
                area_inter += area_from_path(res[r]);

            // criterion to suppress
            float overlap = area_inter / fmin( areas[i]+1.e-10, areas[j]+1.e-10 );

            if (overlap > threshold)
                suppressed[j] = true;
        }
    }

    npy_intp dims_result[1];
    dims_result[0] = n_polys;

    result = (PyArrayObject*)PyArray_SimpleNew(1,dims_result,NPY_BOOL);

    for (int i=0; i<n_polys;i++)
        *(bool *)PyArray_GETPTR1(result,i) = !suppressed[i];

    delete [] areas;
    delete [] suppressed;
    delete [] poly_paths;
    delete [] bbox_x1;
    delete [] bbox_x2;
    delete [] bbox_y1;
    delete [] bbox_y2;

    return PyArray_Return(result);
}


//------------------------------------------------------------------------


static struct PyMethodDef methods[] = {
    {"c_non_max_suppression_inds", c_non_max_suppression_inds, METH_VARARGS, "non-maximum suppression"},
    {"c_star_dist",                c_star_dist,                METH_VARARGS, "star dist calculation"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "stardist", /* name of module */
    NULL,       /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods,
    NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_stardist(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
