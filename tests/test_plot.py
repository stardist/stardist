def test_random_label_cmap():
    import matplotlib
    from stardist import random_label_cmap
    cmap = random_label_cmap(1024)
    assert isinstance(cmap, matplotlib.colors.Colormap)


def test_draw_polygons():
    from stardist.data import test_image_nuclei_2d
    from stardist import edt_prob, star_dist
    from stardist import draw_polygons

    from stardist.nms import _non_maximum_suppression_old as non_maximum_suppression
    from stardist.geometry import _dist_to_coord_old as dist_to_coord

    lbl = test_image_nuclei_2d(return_mask=True)[1]
    prob = edt_prob(lbl)
    dist = star_dist(lbl, 32)
    coord = dist_to_coord(dist)
    points = non_maximum_suppression(coord, prob, prob_thresh=0.5)
    draw_polygons(coord,prob,points,show_dist=True)


def test_render_label():
    from stardist.data import test_image_nuclei_2d
    from stardist import render_label, render_label_pred

    x, y = test_image_nuclei_2d(return_mask=True)
    r = render_label(y, img=x, alpha=0.123, cmap=(0.9,0.1,0.7))
    assert r.shape == y.shape + (4,)

    r = render_label_pred(y, y, img=None)
    assert r.shape == y.shape + (4,)