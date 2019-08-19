import sys
import numpy as np
import pytest
from stardist.models import Config3D, StarDist3D
from utils import circle_image
import tempfile

def create_data():
    img1 = circle_image(shape = (64,80,96), center = (0,.8,.8), radius = .8)
    img2 = circle_image(shape = (64,80,96), center = (0,-.8,-.8), radius = .8)

    img = img1 + 2*img2
    
    imgs = np.repeat(img[np.newaxis],10, axis = 0)

    X = imgs+.6*np.random.uniform(0,1,imgs.shape)
    Y = imgs.astype(np.uint16)

    return X,Y


@pytest.mark.parametrize('n_rays, grid', [(73,(2,2,2)),(33,(1,2,4))])
def test_model(n_rays, grid):

    X,Y = create_data()
    
    conf = Config3D (
        rays       = n_rays,
        grid       = grid,
        use_gpu    = False,
        use_dist_mask = True, 
        train_epochs     = 1,
        train_steps_per_epoch = 2,
        train_loss_weights = (4,1),
        train_patch_size = (48,64,64),
        n_channel_in = 1)

    with tempfile.TemporaryDirectory() as tmp:
        model = StarDist3D(conf, name='stardist', basedir=tmp)
        model.train(X, Y, validation_data=(X[:2],Y[:2]))



if __name__ == '__main__':
    # test_model(16,(1,1,1))

    from stardist.models import StarDistData3D
    from stardist import Rays_GoldenSpiral
    

    
    n_rays = 32
    rays = Rays_GoldenSpiral(n_rays)
    
    X, Y = create_data()

    # np.random.seed(44)
    # from tifffile import imread
    # X = (imread("/home/mweigert/python/stardist/examples/3D_dist_mask/data/test/images/stack_0027.tif"),)
    # Y = (imread("/home/mweigert/python/stardist/examples/3D_dist_mask/data/test/masks/stack_0027.tif"),)
    # data = StarDistData3D(X,Y, batch_size = 1, rays = rays,
    #                       # patch_size=X[0].shape,
    #                       patch_size=(32,48,48),
    #                       use_dist_mask= True)

    # (x,dist_mask), (prob,dist) = data[0]
    



    conf = Config3D (
        rays       = n_rays,
        grid       = (1,2,2),
        
        use_gpu    = False,
        use_dist_mask = True, 
        use_valid_mask = True, 
        train_epochs     = 1,
        train_steps_per_epoch = 20,
        train_loss_weights = (4,1),
        train_patch_size = (48,80,80),
        n_channel_in = 1)

    # with tempfile.TemporaryDirectory() as tmp:
    model = StarDist3D(conf, name = "test3D", basedir = "_tmp")
    # model.prepare_for_training()
    model.train(X, Y, validation_data=(X[:2],Y[:2]))

    # p,d = model.predict(X[0])
