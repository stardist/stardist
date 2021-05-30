#ifndef M_PI
#define M_PI 3.141592653589793
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline float2 pol2cart(const float rho, const float phi) {
    const float x = rho * cos(phi);
    const float y = rho * sin(phi);
    return (float2)(x,y);
}

__kernel void star_dist(__global float* dst, read_only image2d_t src, const int grid_y, const int grid_x) {

    const int i = get_global_id(0), j = get_global_id(1);
    const int Nx = get_global_size(0), Ny = get_global_size(1);
    const float2 grid = (float2)(grid_x, grid_y);

    const float2 origin = (float2)(i,j) * grid;
    const int value = read_imageui(src,sampler,origin).x;

    if (value == 0) {
        // background pixel -> nothing to do, write all zeros
        for (int k = 0; k < N_RAYS; k++) {
            dst[k + i*N_RAYS + j*N_RAYS*Nx] = 0;
        }
    } else {
        float st_rays = (2*M_PI) / N_RAYS; // step size for ray angles
        // for all rays
        for (int k = 0; k < N_RAYS; k++) {
            const float phi = k*st_rays; // current ray angle phi
            const float2 dir = pol2cart(1,phi); // small vector in direction of ray
            float2 offset = 0; // offset vector to be added to origin
            // find radius that leaves current object
            while (1) {
                offset += dir;
                const int offset_value = read_imageui(src,sampler,round(origin+offset)).x;
                if (offset_value != value) {
                  // small correction as we overshoot the boundary
                  const float t_corr = .5f/fmax(fabs(dir.x),fabs(dir.y));
                  offset += (t_corr-1.f)*dir;

                  const float dist = sqrt(offset.x*offset.x + offset.y*offset.y);
                  dst[k + i*N_RAYS + j*N_RAYS*Nx] = dist;
                  break;
                }
            }
        }
    }

}
