#include "stardist3d_lib.h"
 
void _LIB_non_maximum_suppression_sparse(
                    const float* scores, const float* dist, const float* points,
                    const int n_polys, const int n_rays, const int n_faces,
                    const float* verts, const int* faces,
                    const float threshold, const int use_bbox, const int use_kdtree, const int verbose,
                    bool* result
                                         )
{
  _COMMON_non_maximum_suppression_sparse(scores, dist, points,
                                         n_polys, n_rays, n_faces,
                                         verts, faces,
                                         threshold, use_bbox, use_kdtree , verbose, 
                                           result );
}


void _LIB_polyhedron_to_label(const float* dist, const float* points,
                              const float* verts,const int* faces,
                              const int n_polys,
                              const int n_rays,
                              const int n_faces,                                    
                              const int* labels,
                              const int nz, const int  ny, const int nx,
                              const int render_mode,
                              const int verbose,
                              const int use_overlap_label,
                              const int overlap_label,                                
                         int * result)
{
  _COMMON_polyhedron_to_label(dist, points,
                              verts,faces,
                              n_polys,n_rays, n_faces, 
                              labels,
                              nz, ny,nx,
                              render_mode,
                              verbose,
                              use_overlap_label,
                              overlap_label,                                
                      result);
}
