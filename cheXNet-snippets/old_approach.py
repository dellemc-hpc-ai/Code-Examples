"""
This snippet demonstrates a non optimized tf data pipeline. 
This is not full code, this is merely a snippet.

1. Gets the absolute list of filenames.  
2. Builds a dataset from the list of filenames using TFRecordDataset()  
3. Create a new dataset that loads and formats images by preprocessing them.  
4. Shard the dataset.  
5. Shuffle the dataset when training.  
6. Repeat the dataset.  
7. Batch the dataset.  
8. Prefetch the dataset for the batch_size.

"""

def prepare_training_ds_once():  
    """  
    Build training ds.  
    """  
    # build a list of absolute paths to locate the training images  
    training_ds = build_image_paths('train')  

    # build a dataset from image paths  
    training_ds = tensorflow.data.TFRecordDataset(training_ds)  

    # Create a new dataset that loads and formats images on the fly by  
    # mapping preprocess_image over the dataset of paths.  
    training_ds = training_ds.map(lambda value: record_parser(value), num_parallel_calls=FLAGS.parallel_calls)  
    training_ds = training_ds.shard(hvd.size(),hvd.rank())  
    training_ds = training_ds.shuffle(buffer_size=500)  
    return training_ds  

  

def build_training_dataset(training_ds):  
    """  
    Build a training tf dataset.  
    Sets up the dataset from image files in the format (image,labels).  
    :return: a tf dataset with required parameters specified in the FLAGS.  
    """  
    training_ds = training_ds.repeat().  
    training_ds  = training_ds.batch(FLAGS.batch_size)  
    training_ds = training_ds.prefetch(buffer_size=FLAGS.batch_size)  
    return training_ds  

  

  
# do the required preprocessing once and keep the dataset ready as opposed to doing it every epoch.   
training_prep = prepare_training_ds_once()  
  

# in your model.fit() function  
model.fit(  
        build_training_dataset(training_prep).make_one_shot_iterator(),  
        steps_per_epoch=steps_per_epoch // hvd.size(),  
        epochs=FLAGS.epochs,  
        callbacks=callbacks,  
        verbose=verbose)  