import numpy as np
import tensorflow as tf


def wrap_stardistdata_as_tfdata(
    data, 
    shuffle=True, 
    num_parallel_calls=1, 
    verbose=False
):
    """
    wrap a stardist data generator (whose batchsize needs to be 1) into a tf dataset 

    Stardist data generator data return values: 
    
    single-class:

        img, (prob, dist) = data[i] 

    multiclass: 
        
        img, (prob, dist, prob_class) = data[i] 

    """

    if not data.batch_size == 1:
        raise ValueError("Can only convert stardist data generator with batchsize 1 to tf data!")

    def _gen_idx():
        for i in range(len(data)):
            yield i

    def _pre_map(idx):
        """ for a given idx, return data[idx] as flattened list"""
        if verbose:
            print(f"fetching stardist item {idx:4d}   ({data.data_size})")
        a, b = data[idx]
        return tuple(_x[0].astype(np.float32, copy=False) for _x in a) + tuple(_x[0].astype(np.float32, copy=False) for _x in b)

    def _id_map(idx):
        """ defines the input and outputs tensors (4 outputs if multiclass else 3)"""
        return tf.numpy_function(
            func=_pre_map,
            inp=[idx],
            Tout=[tf.float32] * (3 if data.n_classes is None else 4),
        )

    def _post_map(*args):
        """ reformat tuples such that they get accepted by model.fit"""
        return (args[0],), args[1:]

    # generate ids
    data_tf = tf.data.Dataset.from_generator(
        _gen_idx, output_types=tf.int32, output_shapes=()
    )

    # shuffle if necessary
    if shuffle:
        data_tf = data_tf.shuffle(data.data_size)

    # parallelize  
    data_tf = data_tf.map(_id_map, num_parallel_calls=num_parallel_calls)

    # reformat tuple output
    data_tf = data_tf.map(_post_map)

    return data_tf
