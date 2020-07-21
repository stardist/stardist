#ifndef M_PI
#define M_PI 3.141592653589793
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline int round_to_int(float r) {
    return (int)rint(r);
}


__kernel void stardist3d(read_only image3d_t lbl, __constant float * rays, __global float* dist, const int grid_z, const int grid_y, const int grid_x) {

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const int Nx = get_global_size(0);
  const int Ny = get_global_size(1);
  const int Nz = get_global_size(2);

  const float4 grid = (float4)(grid_x, grid_y, grid_z, 1);
  const float4 origin = (float4)(i,j,k,0) * grid;
  const int value = read_imageui(lbl,sampler,origin).x;

  if (value == 0) {
	// background pixel -> nothing to do, write all zeros
	for (int m = 0; m < N_RAYS; m++) {
	  dist[m + i*N_RAYS + j*N_RAYS*Nx+k*N_RAYS*Nx*Ny] = 0;
	}
	
  }
  else {
	for (int m = 0; m < N_RAYS; m++) {
	
	  const float4 dx = (float4)(rays[3*m+2],rays[3*m+1],rays[3*m],0);
	  // if ((i==Nx/2)&&(j==Ny/2)&(k==Nz/2)){
	  // 	printf("kernel: %.2f %.2f  %.2f  \n",dx.x,dx.y,dx.z);
	  // }
	  float4 x = (float4)(0,0,0,0);

	  // move along ray
	  while (1) {
		x += dx;
		// if ((i==10)&&(j==10)&(k==10)){
		//   printf("kernel run: %.2f %.2f  %.2f value %d \n",x.x,x.y,x.z, read_imageui(lbl,sampler,origin+x).x);
		// }
		
		// to make it equivalent to the cpp version... 
		const float4 x_int = (float4)(round_to_int(x.x),
										 round_to_int(x.y),
										 round_to_int(x.z),
										 0);

		if (value != read_imageui(lbl,sampler,origin+x_int).x){

		  dist[m + i*N_RAYS + j*N_RAYS*Nx+k*N_RAYS*Nx*Ny] = length(x_int);
		  break;		  
		}
	  }
	}
  }
}
