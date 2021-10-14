#include <iostream>
#include "stardist3d_lib.h"


int main()
{

  std::cout << "hallo" << std::endl;


  // Geometry ray definitions
  
  // n_rays, verts, faces are given by the specific Rays object used in the python code
  // e.g, rays = Rays_GoldenSpiral(10)
  // len(rays), rays.vertices, rays.faces
  const int n_rays=10;
  const int n_faces=16;
  
  float verts[3*n_rays] = {-1.0, 0.0, 0.0, -0.78, 0.42, -0.46, -0.56, -0.83, 0.07, -0.33, 0.75, 0.57, -0.11, -0.17, -0.98, 0.11, -0.53, 0.84, 0.33, 0.91, -0.24, 0.56, -0.74, -0.38, 0.78, 0.22, 0.59, 1.0, 0.0, -0.0};
  int faces[3*n_faces] = {9, 6, 4, 2, 4, 0, 0, 4, 1, 1, 4, 6, 1, 3, 0, 6, 3, 1, 9, 4, 7, 4, 2, 7, 8, 6, 9, 8, 3, 6, 3, 8, 5, 5, 2, 0, 0, 3, 5, 5, 7, 2, 9, 7, 5, 5, 8, 9};

  // scores, distances, and center points per object
  const int n_polys=100;
  
  float scores[n_polys];
  float dist[n_polys*n_rays];
  float points[3*n_polys];

  for (int i=0; i < n_polys; ++i) {
    scores[i] = 1.;
    for (int j = 0; j < n_rays; ++j) 
      dist[j+n_rays*i] = 20 *(i+1)/n_polys;
    
    points[3*i] = 10*i;
    points[3*i+1] = 0;
    points[3*i+2] = 0;

  }



  // NMS step
  // result[i] will be set to true/false if object <i> survives/got suppressed 
  bool result[n_polys];
 
  _LIB_non_maximum_suppression_sparse(scores, dist, points, n_polys, n_rays, n_faces,
                                      verts, faces,
                                      0.4, true, true, true, 
                                      result);

  // print NMS result
  for (int i = 0; i < n_polys; ++i)
    std::cout << "keep poly "<< i<< " ? -> "<< result[i] << "\n";

   


  return 0;
}
