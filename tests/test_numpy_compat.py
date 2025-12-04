"""
Tests for NumPy 2.0 compatibility
"""

import numpy as np
import pytest


class TestNumPy2Compatibility:
    """Test suite for NumPy 2.0 compatibility of C extensions"""
    
    def test_numpy_version(self):
        """Verify NumPy version is accessible"""
        assert hasattr(np, '__version__')
        version = tuple(int(x) for x in np.__version__.split('.')[:2])
        # Test should work with both NumPy 1.x (>= 1.20) and 2.x
        assert version >= (1, 20)
    
    def test_stardist2d_import(self):
        """Test that stardist2d C extension can be imported"""
        from stardist.lib import stardist2d
        assert stardist2d is not None
    
    def test_stardist3d_import(self):
        """Test that stardist3d C extension can be imported"""
        from stardist.lib import stardist3d
        assert stardist3d is not None
    
    def test_stardist2d_c_star_dist(self):
        """Test the c_star_dist function from stardist2d"""
        from stardist.lib.stardist2d import c_star_dist
        
        # Create a simple test image with a single object
        img = np.zeros((64, 64), dtype=np.uint16)
        img[20:40, 20:40] = 1  # Simple square object
        
        n_rays = 32
        grid_x = 1
        grid_y = 1
        
        result = c_star_dist(img, n_rays, grid_y, grid_x)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, n_rays)
        assert result.dtype == np.float32
    
    def test_stardist2d_c_non_max_suppression_inds(self):
        """Test the c_non_max_suppression_inds function from stardist2d"""
        from stardist.lib.stardist2d import c_non_max_suppression_inds
        
        # Create simple test data
        n_polys = 10
        n_rays = 32
        
        dist = np.random.rand(n_polys, n_rays).astype(np.float32) * 10
        points = np.random.rand(n_polys, 2).astype(np.float32) * 100
        
        use_kdtree = 0
        use_bbox = 0
        verbose = 0
        threshold = 0.5
        
        result = c_non_max_suppression_inds(dist, points, use_kdtree, use_bbox, verbose, threshold)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (n_polys,)
        assert result.dtype == bool
    
    def test_stardist3d_c_star_dist3d(self):
        """Test the c_star_dist3d function from stardist3d"""
        from stardist.lib.stardist3d import c_star_dist3d
        
        # Create a simple test volume with a single object
        img = np.zeros((32, 32, 32), dtype=np.uint16)
        img[10:20, 10:20, 10:20] = 1  # Simple cube object
        
        n_rays = 96
        grid_x = 1
        grid_y = 1
        grid_z = 1
        
        # Create normalized direction vectors
        pdz = np.random.randn(n_rays).astype(np.float32)
        pdy = np.random.randn(n_rays).astype(np.float32)
        pdx = np.random.randn(n_rays).astype(np.float32)
        # Normalize direction vectors
        norm = np.sqrt(pdx**2 + pdy**2 + pdz**2)
        pdx /= norm
        pdy /= norm
        pdz /= norm
        
        result = c_star_dist3d(img, pdz, pdy, pdx, n_rays, grid_z, grid_y, grid_x)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 32, n_rays)
        assert result.dtype == np.float32
    
    def test_stardist3d_c_non_max_suppression_inds(self):
        """Test the c_non_max_suppression_inds function from stardist3d"""
        from stardist.lib.stardist3d import c_non_max_suppression_inds
        
        # Create simple test data
        n_polys = 10
        n_rays = 96
        n_faces = 192
        
        dist = np.random.rand(n_polys, n_rays).astype(np.float32) * 10
        points = np.random.rand(n_polys, 3).astype(np.float32) * 100
        verts = np.random.rand(n_rays, 3).astype(np.float32)
        faces = np.random.randint(0, n_rays, (n_faces, 3)).astype(np.int32)
        scores = np.random.rand(n_polys).astype(np.float32)
        
        use_bbox = 0
        use_kdtree = 0
        verbose = 0
        threshold = 0.5
        
        result = c_non_max_suppression_inds(dist, points, verts, faces, scores, 
                                            use_bbox, use_kdtree, verbose, threshold)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (n_polys,)
        assert result.dtype == bool
    
    def test_stardist3d_other_functions(self):
        """Test other stardist3d C functions"""
        from stardist.lib.stardist3d import c_polyhedron_to_label, c_dist_to_volume, c_dist_to_centroid
        
        # Test c_dist_to_volume
        nz, ny, nx, n_rays = 16, 16, 16, 96
        n_faces = 192
        dist = np.random.rand(nz, ny, nx, n_rays).astype(np.float32)
        verts = np.random.rand(n_rays, 3).astype(np.float32)
        faces = np.random.randint(0, n_rays, (n_faces, 3)).astype(np.int32)
        
        result = c_dist_to_volume(dist, verts, faces)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (nz, ny, nx)
        assert result.dtype == np.float32
        
        # Test c_dist_to_centroid
        result = c_dist_to_centroid(dist, verts, faces, 0)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (nz, ny, nx, 3)
        assert result.dtype == np.float32
        
        # Test c_polyhedron_to_label
        n_polys = 5
        dist_polys = np.random.rand(n_polys, n_rays).astype(np.float32) * 5
        points = np.random.rand(n_polys, 3).astype(np.float32) * 8 + 4  # Center in volume
        labels = np.arange(1, n_polys + 1).astype(np.int32)
        render_mode = 0
        verbose = 0
        use_overlap_label = 0
        overlap_label = -1
        
        result = c_polyhedron_to_label(dist_polys, points, verts, faces, labels,
                                       render_mode, verbose, use_overlap_label, overlap_label,
                                       (nz, ny, nx))
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (nz, ny, nx)
        assert result.dtype == np.int32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
