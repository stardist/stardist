import numpy as np
from stardist.nms import non_maximum_suppression, non_maximum_suppression_sparse, non_maximum_suppression_3d, non_maximum_suppression_3d_sparse
from stardist.geometry.geom2d import polygons_to_label, dist_to_coord
from stardist.geometry.geom3d import polyhedron_to_label
from stardist.rays3d import rays_from_json, Rays_GoldenSpiral
from stardist.matching import relabel_sequential
from typing import Optional, Dict, Tuple, Any, Union


def stardist_postprocessing_2D(
        img_shape: Tuple[int, ...],
        prob: np.ndarray,
        dist: np.ndarray,
        grid: Tuple[int, ...] = (1, 1),
        points: Optional[np.ndarray] = None,
        prob_class: Optional[np.ndarray] = None,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
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


def stardist_postprocessing_3D(
        img_shape: Tuple[int, ...],
        prob: np.ndarray,
        dist: np.ndarray,
        rays: Optional[Union[str, Any, int]] = None,
        anisotropy: Optional[Tuple[float, float, float]] = None,
        grid: Tuple[int, ...] = (1, 1, 1),
        points: Optional[np.ndarray] = None,
        prob_class: Optional[np.ndarray] = None,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        overlap_label: Optional[Any] = None,
        return_labels: bool = True,
        scale: Optional[Dict[str, float]] = None,
        **nms_kwargs
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Apply StarDist 3D post-processing to probability maps and distance predictions.
        
        Args:
            img_shape: Shape of the input image (D, H, W)
            prob: Probability map from StarDist 3D prediction
            dist: Distance predictions from StarDist 3D
            rays: Rays configuration (JSON string, rays object, or None for default 64 rays)
            anisotropy: Anisotropy tuple (Z, Y, X) for rays generation if rays is int or None
            grid: Network output scaling factor, default (1,1,1)
            points: Optional points for sparse prediction
            prob_class: Optional class probabilities for multi-class prediction
            prob_thresh: Probability threshold for detection
            nms_thresh: Non-maximum suppression threshold
            overlap_label: Label for overlapping regions
            return_labels: Whether to return instance labels
            scale: Optional scaling dictionary with 'X', 'Y', and 'Z' entries
            **nms_kwargs: Additional arguments for NMS
            
        Returns:
            tuple: (labels, details_dict)
                - labels: Instance segmentation mask (if return_labels=True)
                - details_dict: Dictionary with 'dist', 'points', 'prob', 'rays', etc.
        """
        # Handle rays configuration
        if rays is None:
            rays = Rays_GoldenSpiral(64, anisotropy=anisotropy)
        elif isinstance(rays, str):
            rays = rays_from_json(rays)
            # ignore anisotropy parameter when rays_json is provided
            if anisotropy is not None:
                import warnings
                warnings.warn("anisotropy parameter ignored when rays is provided as JSON string")
        elif isinstance(rays, int):
            # fallback to default anisotropy if None
            if anisotropy is None:
                import warnings
                warnings.warn("Using default isotropic rays (1,1,1) as anisotropy is None."
                              "Consider providing anisotropy tuple or the full rays_json configuration for accurate 3D reconstruction.")
                anisotropy = (1,1,1)
            rays = Rays_GoldenSpiral(rays, anisotropy=anisotropy)
        else:
            rays = rays_from_json(rays)

        # sparse prediction
        if points is not None:
            points, probi, disti, indsi = non_maximum_suppression_3d_sparse(dist, prob, points, rays, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                prob_class = prob_class[indsi]

        # dense prediction
        else:
            points, probi, disti = non_maximum_suppression_3d(dist, prob, rays, grid=grid,
                                                              prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, grid))
                prob_class = prob_class[inds]

        verbose = nms_kwargs.get('verbose',False)
        verbose and print("render polygons...")

        if scale is not None:
            # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5,Z=1.0):
            #   1. re-scale points (origins of polyhedra)
            #   2. re-scale vectors of rays object (computed from distances)
            if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale and 'Z' in scale):
                raise ValueError("scale must be a dictionary with entries for 'X', 'Y', and 'Z'")
            rescale = (1/scale['Z'],1/scale['Y'],1/scale['X'])
            points = points * np.array(rescale).reshape(1,3)
            rays = rays.copy(scale=rescale)
        else:
            rescale = (1,1,1)

        if return_labels:
            labels = polyhedron_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label=overlap_label, verbose=verbose)

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _,_ = relabel_sequential(labels)
                # print(np.unique(labels))
        else:
            labels = None

        res_dict = dict(dist=disti, points=points, prob=probi, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            # build the list of class ids per label via majority vote
            # zoom prob_class to img_shape
            # prob_class_up = zoom(prob_class,
            #                      tuple(s2/s1 for s1, s2 in zip(prob_class.shape[:3], img_shape))+(1,),
            #                      order=0)
            # class_id, label_ids = [], []
            # for reg in regionprops(labels):
            #     m = labels[reg.slice]==reg.label
            #     cls_id = np.argmax(np.mean(prob_class_up[reg.slice][m], axis = 0))
            #     class_id.append(cls_id)
            #     label_ids.append(reg.label)
            # # just a sanity check whether labels where in sorted order
            # assert all(x <= y for x,y in zip(label_ids, label_ids[1:]))
            # res_dict.update(dict(classes = class_id))
            # res_dict.update(dict(labels = label_ids))
            # self.p = prob_class_up

            prob_class = np.asarray(prob_class)
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict
