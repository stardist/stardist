#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>

#include "numpy/arrayobject.h"
#include "clipper.hpp"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif


#ifdef _OPENMP
#include <omp.h>
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
                // small correction as we overshoot the boundary
                const float t_corr = .5f/fmax(fabs(dx),fabs(dy));
                x += (t_corr-1.f)*dx;
                y += (t_corr-1.f)*dy;
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



inline bool bbox_intersect(const int bbox_a_x1, const int bbox_a_x2,
                           const int bbox_a_y1, const int bbox_a_y2,
                           const int bbox_b_x1, const int bbox_b_x2,
                           const int bbox_b_y1, const int bbox_b_y2) {
  // return !( bbox_b_x1 >  bbox_a_x2 || bbox_a_x1 >  bbox_b_x2 || bbox_b_y1 >  bbox_a_y2 || bbox_a_y1 >  bbox_b_y2 );
  return  ( bbox_b_x1 <= bbox_a_x2 && bbox_a_x1 <= bbox_b_x2 && bbox_b_y1 <= bbox_a_y2 && bbox_a_y1 <= bbox_b_y2 );
}



inline float poly_intersection_area(const ClipperLib::Path poly_a_path, const ClipperLib::Path poly_b_path) {
  ClipperLib::Clipper c;
  ClipperLib::Paths res;
  c.Clear();

  c.AddPath(poly_a_path,ClipperLib::ptClip, true);
  c.AddPath(poly_b_path,ClipperLib::ptSubject, true);
  c.Execute(ClipperLib::ctIntersection, res, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

  float area_inter = 0;
  for (unsigned int r=0; r<res.size(); r++)
    area_inter += area_from_path(res[r]);
  return area_inter;
}



// polys.shape = (n_polys, 2, n_rays)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept

static PyObject* c_non_max_suppression_inds (PyObject *self, PyObject *args) {

  PyArrayObject *polys=NULL, *mapping=NULL, *result=NULL;
  float threshold;
  int max_bbox_search, grid_x, grid_y;
  int verbose;

  if (!PyArg_ParseTuple(args, "O!O!fiiii", &PyArray_Type, &polys, &PyArray_Type, &mapping, &threshold, &max_bbox_search, &grid_y, &grid_x, &verbose))
    return NULL;

  npy_intp *img_dims = PyArray_DIMS(mapping);
  const int height = img_dims[0], width = img_dims[1];

  npy_intp *dims = PyArray_DIMS(polys);
  const int n_polys = dims[0];
  const int n_rays = dims[2];

  int * bbox_x1 = new int[n_polys];
  int * bbox_x2 = new int[n_polys];
  int * bbox_y1 = new int[n_polys];
  int * bbox_y2 = new int[n_polys];

  int max_bbox_size_x = 0;
  int max_bbox_size_y = 0;

  float * areas = new float[n_polys];
  bool * suppressed = new bool[n_polys];
  ClipperLib::Path * poly_paths = new ClipperLib::Path[n_polys];

  int count_suppressed = 0;

  //initialize indices
#pragma omp parallel for
  for (int i=0; i<n_polys; i++) {
    suppressed[i] = false;
  }


  if (verbose){
    printf("Non Maximum Suppression (2D) ++++ \n");
    printf("NMS: n_polys  = %d \nNMS: n_rays   = %d  \nNMS: thresh   = %.3f \nNMS: max_bbox_search = %d \n", n_polys, n_rays, threshold, max_bbox_search);
#ifdef _OPENMP
    printf("NMS: using OpenMP with %d thread(s)\n", omp_get_max_threads());
#endif
  }

  // build polys and areas

  // disable OpenMP  for now, as there is still a race condition (segfaults on OSX)
  // #pragma omp parallel for
  for (int i=0; i<n_polys; i++) {
    ClipperLib::Path clip;
    // build clip poly and bounding boxes
    for (int k =0; k<n_rays; k++) {
      int y = *(int *)PyArray_GETPTR3(polys,i,0,k);
      int x = *(int *)PyArray_GETPTR3(polys,i,1,k);
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
    if (max_bbox_search) {
      const int bbox_size_x = bbox_x2[i] - bbox_x1[i];
      const int bbox_size_y = bbox_y2[i] - bbox_y1[i];
      if (bbox_size_x > max_bbox_size_x) {
#pragma omp critical (max_x)
        max_bbox_size_x = bbox_size_x;
      }
      if (bbox_size_y > max_bbox_size_y) {
#pragma omp critical (max_y)
        max_bbox_size_y = bbox_size_y;
      }
    }
    poly_paths[i] = clip;
    areas[i] = area_from_path(clip);
  }

  // printf("max_bbox_size_x = %d, max_bbox_size_y = %d\n", max_bbox_size_x, max_bbox_size_y);
  if (verbose)
    printf("NMS: starting suppression loop\n");

  ProgressBar prog("suppressed");

  if (max_bbox_search) {

    // suppress (double loop)
    for (int i=0; i<n_polys-1; i++) {
      if (suppressed[i]) continue;

      if (verbose)
        prog.update(100.*count_suppressed/n_polys);

      // check signals e.g. such that the loop is interruptible
      if (PyErr_CheckSignals()==-1){
        delete [] areas;
        delete [] suppressed;
        delete [] poly_paths;
        delete [] bbox_x1;
        delete [] bbox_x2;
        delete [] bbox_y1;
        delete [] bbox_y2;
        PyErr_SetString(PyExc_KeyboardInterrupt, "interrupted");
        return Py_None;
      }

      const int xs = std::max((bbox_x1[i]-max_bbox_size_x)/grid_x, 0);
      const int xe = std::min((bbox_x2[i]+max_bbox_size_x)/grid_x, width);
      const int ys = std::max((bbox_y1[i]-max_bbox_size_y)/grid_y, 0);
      const int ye = std::min((bbox_y2[i]+max_bbox_size_y)/grid_y, height);

      // printf("%5d [%03d:%03d,%03d:%03d]",i,bbox_x1[i],bbox_x2[i],bbox_y1[i],bbox_y2[i]);
      // printf(" - search area [%03d:%03d,%03d:%03d]\n",xs,xe,ys,ye);

      // cf. https://github.com/peterwittek/somoclu/issues/111
#ifdef _WIN32
#pragma omp parallel for schedule(dynamic) reduction(+:count_suppressed)
#else
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:count_suppressed)
#endif
      for (int jj=ys; jj<ye; jj++) for (int ii=xs; ii<xe; ii++) {
          // j is the id of the score-sorted polygon at coordinate (ii,jj)
          const int j = *(int *)PyArray_GETPTR2(mapping,jj,ii);
          // if (j<0) continue;  // polygon not even a candidate (check redundant because of next line)
          if (j<=i) continue; // polygon has higher score (i.e. lower id) than "suppressor polygon" i
          if (suppressed[j]) continue;
          // skip if bounding boxes are not even intersecting
          if (!bbox_intersect(bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j], bbox_y1[j], bbox_y2[j]))
            continue;

          const float area_inter = poly_intersection_area(poly_paths[i], poly_paths[j]);
          const float overlap = area_inter / fmin( areas[i]+1.e-10, areas[j]+1.e-10 );
          if (overlap > threshold){
            count_suppressed +=1;
            suppressed[j] = true;
          }
        }
    }

  } else {

    // suppress (double loop)
    for (int i=0; i<n_polys-1; i++) {
      if (suppressed[i]) continue;

      if (verbose)
        prog.update(100.*count_suppressed/n_polys);

      // check signals e.g. such that the loop is interruptible
      if (PyErr_CheckSignals()==-1){
        delete [] areas;
        delete [] suppressed;
        delete [] poly_paths;
        delete [] bbox_x1;
        delete [] bbox_x2;
        delete [] bbox_y1;
        delete [] bbox_y2;
        PyErr_SetString(PyExc_KeyboardInterrupt, "interrupted");
        return Py_None;
      }

      // printf("%5d [%03d:%03d,%03d:%03d]\n",i,bbox_x1[i],bbox_x2[i],bbox_y1[i],bbox_y2[i]);

#pragma omp parallel for schedule(dynamic) reduction(+:count_suppressed)
      for (int j=i+1; j<n_polys; j++) {
        if (suppressed[j]) continue;
        // skip if bounding boxes are not even intersecting
        if (!bbox_intersect(bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j], bbox_y1[j], bbox_y2[j]))
          continue;

        const float area_inter = poly_intersection_area(poly_paths[i], poly_paths[j]);
        const float overlap = area_inter / fmin( areas[i]+1.e-10, areas[j]+1.e-10 );
        if (overlap > threshold){
          count_suppressed +=1;
          suppressed[j] = true;
        }

      }
    }

  }

  if (verbose)
    prog.finish();

  if (verbose){
    printf("NMS: Suppressed polygons:   %8d / %d  (%.2f %%)\n", count_suppressed,n_polys,100*(float)count_suppressed/n_polys);
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
                                       {"c_non_max_suppression_inds",
                                        c_non_max_suppression_inds,
                                        METH_VARARGS, "non-maximum suppression"},
                                       {"c_star_dist",
                                        c_star_dist,
                                        METH_VARARGS, "star dist calculation"},
                                       {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "stardist2d",
                                       NULL,
                                       -1,
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_stardist2d(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
