import numpy as np
from stardist.nms import non_maximum_suppression, non_maximum_suppression_sparse
from stardist.geometry.geom2d import polygons_to_label, dist_to_coord
from typing import Optional, Dict, Tuple, Any


def stardist_postprocessing(
        img_shape: Tuple[int, ...],
        prob: np.ndarray,
        dist: np.ndarray,
        grid: Tuple[int, ...] = (2, 2),
        points: Optional[np.ndarray] = None,
        prob_class: Optional[np.ndarray] = None,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        overlap_label: Optional[Any] = None,
        return_labels: bool = True,
        scale: Optional[Dict[str, float]] = None,
        **nms_kwargs
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Apply StarDist post-processing to probability maps and distance predictions.
    
    Args:
        img_shape: Shape of the input image (H, W)
        prob: Probability map from StarDist prediction
        dist: Distance predictions from StarDist
        grid: Network output scaling factor, default (2,2)
        points: Optional points for sparse prediction
        prob_class: Optional class probabilities for multi-class prediction
        prob_thresh: Probability threshold for detection
        nms_thresh: Non-maximum suppression threshold
        overlap_label: Not supported for 2D
        return_labels: Whether to return instance labels
        scale: Optional scaling dictionary with 'X' and 'Y' entries
        **nms_kwargs: Additional arguments for NMS
        
    Returns:
        tuple: (labels, details_dict)
            - labels: Instance segmentation mask (if return_labels=True)
            - details_dict: Dictionary with 'coord', 'points', 'prob' and optional class info
    """
    if overlap_label is not None: raise NotImplementedError("overlap_label not supported for 2D yet!")

    # sparse prediction
    if points is not None:
        points, probi, disti, indsi = non_maximum_suppression_sparse(dist, prob, points, nms_thresh=nms_thresh, **nms_kwargs)
        if prob_class is not None:
            prob_class = prob_class[indsi]

    # dense prediction
    else:
        points, probi, disti = non_maximum_suppression(dist,
                                                       prob,
                                                       grid=grid,
                                                       prob_thresh=prob_thresh,
                                                       nms_thresh=nms_thresh,
                                                       **nms_kwargs
                                                       )
        if prob_class is not None:
            inds = tuple(p//g for p,g in zip(points.T, grid))
            prob_class = prob_class[inds]

    if scale is not None:
        # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5):
        #   1. re-scale points (origins of polygons)
        #   2. re-scale coordinates (computed from distances) of (zero-origin) polygons
        if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale):
            raise ValueError("scale must be a dictionary with entries for 'X' and 'Y'")
        rescale = (1/scale['Y'],1/scale['X'])
        points = points * np.array(rescale).reshape(1,2)
    else:
        rescale = (1,1)

    if return_labels:
        labels = polygons_to_label(disti, points, prob=probi, shape=img_shape, scale_dist=rescale)
    else:
        labels = None

    coord = dist_to_coord(disti, points, scale_dist=rescale)
    res_dict = dict(coord=coord, points=points, prob=probi)

    # multi class prediction
    if prob_class is not None:
        prob_class = np.asarray(prob_class)
        class_id = np.argmax(prob_class, axis=-1)
        res_dict.update(dict(class_prob=prob_class, class_id=class_id))

    return labels, res_dict