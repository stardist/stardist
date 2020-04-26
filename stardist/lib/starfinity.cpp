#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include "numpy/arrayobject.h"
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif


inline int round_to_int(float r) {
    return (int)lrint(r);
}



// ----------------------------------------------------------------------
// Affinity functions


static PyObject* c_dist_to_affinity2D(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_weights = NULL;
  PyArrayObject *arr_result=NULL;
  PyArrayObject *arr_result_neg=NULL;

  int verbose;
  int normed;
  float kappa;
  int gridz, gridy, gridx;

  if (!PyArg_ParseTuple(args, "O!O!fiiii",
						&PyArray_Type, &arr_dist,
						&PyArray_Type, &arr_weights,
						&kappa,
						&normed,
						&gridy,
						&gridx,						 
						&verbose))
       return NULL;


   const unsigned long ny      = PyArray_DIMS(arr_dist)[0];
   const unsigned long nx      = PyArray_DIMS(arr_dist)[1];
   const unsigned long n_rays  = PyArray_DIMS(arr_dist)[2];

   const float * const dist = (float*) PyArray_DATA(arr_dist);
   const float * const weights = (float*) PyArray_DATA(arr_weights);

   npy_intp dims_result[3];
   dims_result[0] = ny;
   dims_result[1] = nx;
   dims_result[2] = 2;

   arr_result = (PyArrayObject*)PyArray_ZEROS(3,dims_result,NPY_FLOAT32,0);
   arr_result_neg = (PyArrayObject*)PyArray_ZEROS(3,dims_result,NPY_FLOAT32,0);

   if (verbose>=1){
	 printf("affinities ++++ \n");
	 printf("kappa   %.2f \n", kappa);
	 printf("normed  %d \n", normed);
	 printf("grid    %d %d \n", gridy, gridx);
   }

   
	 // loop over all pixels
#pragma omp parallel for schedule(dynamic)
   for (unsigned long y0 = 0; y0 < ny; ++y0) {
	 for (unsigned long x0 = 0; x0 < nx; ++x0) {

	   unsigned long offset = x0+nx*y0;

	   float weight = weights[offset];
	   if (weight==0)
		 continue;
	   
	   const float st_rays = (2*M_PI) / n_rays;
		 
	   for (int r = 0; r < n_rays; ++r) {

		 const float phi = r*st_rays;
		 const float dy0 = sin(phi);
		 const float dx0 = cos(phi);
		 
		   
		 float aff_inc = weight;

           
		 // grid-correction of distances
		 float dy = dy0/gridy;
		 float dx = dx0/gridx;

		 float _mag = sqrt(dy*dy+dx*dx);
		 dy /= _mag;
		 dx /= _mag;
		 float true_dist = dist[r+n_rays*offset]*_mag;
		 
		 // if ((y0*gridy==10)&&(x0*gridx==10))
		 //   printf("%.4f %.4f\n", _mag, true_dist);
		   
		 int d0 = round_to_int(true_dist);
		 if (d0<1)
		   continue;

         float true_dist_opposite = dist[((r+n_rays/2)%n_rays)+n_rays*offset]*_mag;
         int d0_opposite = round_to_int(true_dist_opposite);
             
		 // loop through all pixels that are have a distance smaller than d0-1
		 // and increment the affinity along the specific direction
		   
		 for(int i=0;i<d0;i++){


		   // if !((0<=x)&&(x<nx)&&(0<=y)&&(y<ny)&&(0<=z)&&(z<nz))
		   //   break;

		   int ry = round_to_int(i*dy);
		   int rx = round_to_int(i*dx);


		   float prefac = exp(-kappa*i); 

		   // if (normed){
		   //   if (kappa>0)
		   //     prefac *= (1.-exp(-kappa))/(exp(-kappa)-exp(-kappa*(d0)));
		   //   else
		   //     prefac *= 1./d0;
		   // }
		   if (normed){
             prefac *= 1+kappa*(d0_opposite+i);
             prefac *= 2./(d0+d0_opposite)/n_rays;
		   }
		   
		   // increment all affinities that are neighboring the two pixels 
		   // int ay = y1 - (dy<0);
		   // int ax = x1 - (dx<0);
		   int ay = y0+ry;
		   int ax = x0+rx;

		   if (!((0<=ax)&&(ax<nx)&&(0<=ay)&&(ay<ny)))
			 break;

		   // weight by the increment vector
			 
#pragma omp atomic
		   *(float *)PyArray_GETPTR3(arr_result,ay,ax,0) +=
			 fabs(dy)*prefac*aff_inc;
#pragma omp atomic
		   *(float *)PyArray_GETPTR3(arr_result,ay,ax,1) +=
			 fabs(dx)*prefac*aff_inc;
		 }

		 // negative boundary
		 int ry = round_to_int(d0*dy);
		 int rx = round_to_int(d0*dx);

		 // int ay = y1 - (dy<0);
		 // int ax = x1 - (dx<0);
		 int ay = y0+ry;
		 int ax = x0+rx;

		 if (!((0<=ax)&&(ax<nx)&&(0<=ay)&&(ay<ny)))
		   continue;
		 // weight by the increment vector

         float prefac = 1;
         if (normed)
           prefac *= 2./(d0+d0_opposite)/n_rays;

         
#pragma omp atomic
		 *(float *)PyArray_GETPTR3(arr_result_neg,ay,ax,0) +=
		   fabs(dy)*prefac*aff_inc;
// -			 aff_inc;			
#pragma omp atomic
		 *(float *)PyArray_GETPTR3(arr_result_neg,ay,ax,1) +=
		   fabs(dx)*prefac*aff_inc;
		 // aff_inc;
	   }
	 }
   }

   PyObject *res = PyTuple_New(2);
   PyTuple_SetItem(res, 0, PyArray_Return(arr_result));
   PyTuple_SetItem(res, 1, PyArray_Return(arr_result_neg));
   return res;	 

	 
   // return PyArray_Return(arr_result);
}




static PyObject* c_dist_to_affinity3D(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_verts=NULL, *arr_weights = NULL;
  PyArrayObject *arr_result=NULL;
  PyArrayObject *arr_result_neg=NULL;

  int verbose;
  int normed;
  float kappa;
  float clip_dist;
  int gridz, gridy, gridx;

   if (!PyArg_ParseTuple(args, "O!O!O!fifiiii",
						 &PyArray_Type, &arr_dist,
						 &PyArray_Type, &arr_verts,
						 &PyArray_Type, &arr_weights,
						 &kappa,
						 &normed,
						 &clip_dist,
						 &gridz,
						 &gridy,
						 &gridx,						 
						 &verbose))
       return NULL;


   const unsigned long nz      = PyArray_DIMS(arr_dist)[0];
   const unsigned long ny      = PyArray_DIMS(arr_dist)[1];
   const unsigned long nx      = PyArray_DIMS(arr_dist)[2];
   const unsigned long n_rays  = PyArray_DIMS(arr_dist)[3];


   const float * const dist    = (float*) PyArray_DATA(arr_dist);
   const float * const verts   = (float*) PyArray_DATA(arr_verts);
   const float * const weights = (float*) PyArray_DATA(arr_weights);

   npy_intp dims_result[4];
   dims_result[0] = nz;
   dims_result[1] = ny;
   dims_result[2] = nx;
   dims_result[3] = 3;

   arr_result = (PyArrayObject*)PyArray_ZEROS(4,dims_result,NPY_FLOAT32,0);
   arr_result_neg = (PyArrayObject*)PyArray_ZEROS(4,dims_result,NPY_FLOAT32,0);

   if (verbose>=1){
	 printf("affinities ++++ \n");
	 
	 printf("kappa   %.2f \n", kappa);
	 printf("normed  %d \n", normed);
	 printf("grid    %d %d %d \n", gridz, gridy, gridx);
	 
   // 	 printf("n_polys  = %d \nn_rays   = %d  \nn_faces  = %d \nshape   = %d %d %d\nrender_mode = %d\n", n_polys, n_rays, n_faces,nz,ny,nx, render_mode);
   }


	 // loop over all pixels
#pragma omp parallel for schedule(dynamic)
   for (unsigned long z0 = 0; z0 < nz; ++z0) {
	 for (unsigned long y0 = 0; y0 < ny; ++y0) {
	   for (unsigned long x0 = 0; x0 < nx; ++x0) {

		 unsigned long offset = x0+nx*y0+nx*ny*z0;

		 float weight = weights[offset];
		 if (weight==0)
		   continue;
		 
		 for (unsigned long r = 0; r < n_rays; ++r) {

		   const float dz0 = verts[0+3*r];
		   const float dy0 = verts[1+3*r];
		   const float dx0 = verts[2+3*r];
		   

		   float dz = dz0/gridz;
		   float dy = dy0/gridy;
		   float dx = dx0/gridx;
		   float _mag = sqrt(dz*dz+dy*dy+dx*dx);
		   dz /= _mag;
		   dy /= _mag;
		   dx /= _mag;

		   const float true_dist = dist[r+n_rays*offset]*_mag;
		   
		   const int d0 = round_to_int(true_dist);
		   if (d0<1)
			 continue;
		   
		   float aff_inc = weight;

		   // loop through all pixels that are have a distance smaller than d0-1
		   // and increment the affinity along the specific direction
		   
		   for(int i=0;i<d0;i++){


			 float prefac = exp(-kappa*i); 

			 if (normed){
			   if (kappa>0)
				 prefac *= (1.-exp(-kappa))/(exp(-kappa)-exp(-kappa*(d0)));
			   else
				 prefac *= 1./d0;
			 }

			 // increment all affinities that are neighboring the two pixels 

			 // int az = z1 - (dz<0);
			 // int ay = y1 - (dy<0);
			 // int ax = x1 - (dx<0);

			 if (d0>=clip_dist)
			   continue;
			   

			 int rz = round_to_int(i*dz);
			 int ry = round_to_int(i*dy);
			 int rx = round_to_int(i*dx);

			 int az = z0+rz;
			 int ay = y0+ry;
			 int ax = x0+rx;

			 // if ((z0==4)&&(y0==10)&&(x0==10)){
			 //   printf("r %d %d %d \n",rz,ry,rx);
			 //   printf("a %d %d %d \n",az,ay,ax);
			 // }
			 
			 if (!((0<=ax)&&(ax<nx)&&(0<=ay)&&(ay<ny)&&(0<=az)&&(az<nz)))
			   break;

			 // weight by the increment vector

#pragma omp atomic
			 *(float *)PyArray_GETPTR4(arr_result,az,ay,ax,0) +=
			   fabs(dz)*prefac*aff_inc;
#pragma omp atomic
			 *(float *)PyArray_GETPTR4(arr_result,az,ay,ax,1) +=
			   fabs(dy)*prefac*aff_inc;
#pragma omp atomic
			 *(float *)PyArray_GETPTR4(arr_result,az,ay,ax,2) +=
			   fabs(dx)*prefac*aff_inc;
		   }


		   int rz = round_to_int(d0*dz);
		   int ry = round_to_int(d0*dy);
		   int rx = round_to_int(d0*dx);

		   int az = z0+rz;
		   int ay = y0+ry;
		   int ax = x0+rx;

		   if (!((0<=ax)&&(ax<nx)&&(0<=ay)&&(ay<ny)&&(0<=az)&&(az<nz)))
			 continue;

		   // weight by the increment vector
			 
#pragma omp atomic
		   *(float *)PyArray_GETPTR4(arr_result_neg,az,ay,ax,0) +=
			 fabs(dz)*aff_inc;
		   
#pragma omp atomic
		   *(float *)PyArray_GETPTR4(arr_result_neg,az,ay,ax,1) +=
			 fabs(dy)*aff_inc;
#pragma omp atomic
		   *(float *)PyArray_GETPTR4(arr_result_neg,az,ay,ax,2) +=
			 fabs(dx)*aff_inc;
		 }
	   }
	 }
   }

   PyObject *res = PyTuple_New(2);
   PyTuple_SetItem(res, 0, PyArray_Return(arr_result));
   PyTuple_SetItem(res, 1, PyArray_Return(arr_result_neg));
   return res;	 

   // return PyArray_Return(arr_result);
}


//------------------------------------------------------------------------


static struct PyMethodDef methods[] = {
//    {"c_non_max_suppression_inds", c_non_max_suppression_inds, METH_VARARGS, "non-maximum suppression"},
    {"c_dist_to_affinity2D",
	 c_dist_to_affinity2D,
	 METH_VARARGS, "star affinity calculation"},
    {"c_dist_to_affinity3D",
	 c_dist_to_affinity3D,
	 METH_VARARGS, "star affinity calculation"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "starfinity", /* name of module */
    NULL,       /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods,
    NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_starfinity(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
