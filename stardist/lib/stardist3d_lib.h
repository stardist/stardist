#ifndef STARDIST3D_LIB_H
#define STARDIST3D_LIB_H


void _COMMON_non_maximum_suppression_sparse(
                                            const float* scores,
                                            const float* dist,
                                            const float* points,
                                            const int n_polys,
                                            const int n_rays,
                                            const int n_faces, 
                                            const float* verts,
                                            const int* faces,
                                            const float threshold,
                                            const int use_bbox,
                                            const int use_kdtree,
                                            const int verbose, 
                                            bool* result
                                            );


void _COMMON_polyhedron_to_label(
                                 const float* dist,
                                 const float* points,
                                 const float* verts,
                                 const int* faces,
                                 const int n_polys,
                                 const int n_rays,
                                 const int n_faces,
                                 const int* labels,
                                 const int nz,
                                 const int  ny,
                                 const int nx,
                                 const int render_mode,
                                 const int verbose,
                                 const int use_overlap_label,
                                 const int overlap_label,
                                 int * result
                                 );

#ifdef __cplusplus
extern "C" {
#endif
 
  /* void _LIB_non_maximum_suppression_sparse( */
  /*                   const float*, const float*, const float*,                     */
  /*                   const int, const int, const int, */
  /*                   const float* , const int* , */
  /*                   const float, const int, const int, const int,  */
  /*                   bool*); */

  void _LIB_non_maximum_suppression_sparse(
                                           const float* scores,
                                           const float* dist,
                                           const float* points,
                                           const int n_polys,
                                           const int n_rays,
                                           const int n_faces, 
                                           const float* verts,
                                           const int* faces,
                                           const float threshold,
                                           const int use_bbox,
                                           const int use_kdtree,
                                           const int verbose, 
                                           bool* result
                                           );


  void  _LIB_polyhedron_to_label(const float* dist, const float* points,
                                 const float* verts,const int* faces,
                                 const int n_polys, const int n_rays,
                                 const int n_faces,
                                 const int* labels,
                                 const int nz, const int  ny,const int nx,
                                 const int render_mode,
                                 const int verbose,
                                 const int use_overlap_label,
                                 const int overlap_label,  
                                 int * result);
  
#ifdef __cplusplus
}
#endif

#endif

