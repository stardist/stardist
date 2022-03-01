import tensorflow as tf

def wrap_stardistdata_as_tfdata(data, shuffle=True, num_parallel_calls=1, verbose=False):
    assert data.batch_size==1
    
    def _gen_idx():
        for i in range(len(data)):
            yield i
    def _pre_map(idx):
        if verbose: print(f'fetching stardist item {idx:4d}   ({data.data_size})')
        a, b = data[idx]
        return tuple(_x[0] for _x in a) + tuple(_x[0] for _x in b)

    def _id_map(idx):
        return tf.numpy_function(func=_pre_map,
                    inp=[idx],
                    Tout=[tf.float32]* (3 if data.n_classes is None else 4))
    def _post_map(*args):
        return (args[0],), args[1:]

    # generate ids
    data_tf = tf.data.Dataset.from_generator(_gen_idx,
                        output_types=tf.int32, output_shapes = ())
    if shuffle:
        data_tf = data_tf.shuffle(data.data_size)

    print(f'{num_parallel_calls=}')
    data_tf = data_tf.map(_id_map, num_parallel_calls=num_parallel_calls)
    
    data_tf = data_tf.map(_post_map)
    
    return data_tf
