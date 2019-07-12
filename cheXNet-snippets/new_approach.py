"""
This script demonstrates an optimized pipeline.
This is not full code, this is merely a snippet.

1. Gets the absolute list of filenames.  
2. Builds a dataset from the list of filenames using from_tensor_slices()  
3. Sharding is done ahead of time.  
4. The dataset is shuffled during training.  
5. The dataset is then parallelly interleaved, which is basically interleaving and processing multiple files (defined by cycle_length) to transform them to create TFRecord dataset.  
6. The dataset is then prefetched. The buffer_size defines how many records are prefetched, which is usually the mini batch_size of the job.  
7. The dataset is again shuffled. Details of the shuffle is controlled by buffer_size.  
8. The dataset is repeated.  It repeats the dataset until num_epochs to train.  
9. The dataset is subjected to simultaneous map_and_batch() which parses the tf record files, which in turn preprocesses the image and batches them.  
10. The preprocessed image is ready as a dataset and prefetched again.  

"""

def process_record_dataset(dataset,  
                           is_training,  
                           batch_size,  
                           shuffle_buffer,  
                           parse_record_fn,  
                           num_epochs=1,  
                           dtype=tf.float32,  
                           num_parallel_batches=1,  
                           ):  

    """Given a Dataset with raw records, return an iterator over the records.  
   
        Args:  
          dataset: A Dataset representing raw records  
          is_training: A boolean denoting whether the input is for training.  
          batch_size: The number of samples per batch.  
          shuffle_buffer: The buffer size to use when shuffling records. A larger  
          value results in better randomness, but smaller values reduce startup  
          time and use less memory.  
          parse_record_fn: A function that takes a raw record and returns the  
          corresponding (image, label) pair.  
          num_epochs: The number of epochs to repeat the dataset.  
          dtype: Data type to use for images/features.  
          num_parallel_batches: Number of parallel batches for tf.data.  

        Returns:  
          Dataset of (image, label) pairs ready for iteration.  
    """  

    # Prefetches a batch at a time to smooth out the time taken to load input  
    # files for shuffling and processing.  
    dataset = dataset.prefetch(buffer_size=batch_size)  
    if is_training:  
        # Shuffles records before repeating to respect epoch boundaries.  
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)  
        steps_per_epoch =_NUM_TRAINING_IMAGES//batch_size  
    # take care while building validation dataset  
    else:  
        steps_per_epoch =_NUM_VAL_IMAGES//batch_size  
    # Repeats the dataset for the number of epochs to train.  
    # Multiplying by the factor steps_per_epoc // hvd.size() to  
    # prevent running out of data problem.  
    dataset = dataset.repeat(num_epochs*steps_per_epoch//hvd.size())  


    # Parses the raw records into images and labels.  
    dataset = dataset.apply(  
        tf.data.experimental.map_and_batch(  
            lambda value: parse_record_fn(value, is_training, dtype),  
            batch_size=batch_size,  
            num_parallel_batches=num_parallel_batches,  
            drop_remainder=True  
            ))  

def input_fn(is_training,  
             data_dir,  
             batch_size,  
             num_epochs=1,  
             dtype=tf.float32,  
             num_parallel_batches=5,  
             ):  

    """Input function which provides batches for train or eval.  
  

    Args:  
      is_training: A boolean denoting whether the input is for training.  
      data_dir: The directory containing the input data.  
      batch_size: The number of samples per batch.  
      num_epochs: The number of epochs to repeat the dataset.  
      dtype: Data type to use for images/features  
      num_parallel_batches: Number of parallel batches for tf.data.  
      parse_record_fn: Function to use for parsing the records.  
    Returns:  
      A dataset that can be used for iteration.  
    """  
    filenames = get_filenames(is_training, data_dir)  
    dataset = tf.data.Dataset.from_tensor_slices(filenames)  

    # shard the dataset  
    print('Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (  
                hvd.rank(), hvd.size()))  
    dataset = dataset.shard(hvd.size(), hvd.rank())  

  
    if is_training:  
        # Shuffle the input files  
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)  
    # Convert to individual records.  
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.  
    # This number is low enough to not cause too much contention on small systems  
    # but high enough to provide the benefits of parallelization. You may want  
    # to increase this number if you have a large number of CPU cores.  
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(  
       tf.data.TFRecordDataset, cycle_length=10))  

     return process_record_dataset(  
        dataset=dataset,  
        is_training=is_training,  
        batch_size=batch_size,  
        shuffle_buffer=_SHUFFLE_BUFFER,  
        parse_record_fn=record_parser,  
        num_epochs=num_epochs,  
        dtype=dtype,  
        num_parallel_batches=num_parallel_batches  
    )  

  
train_input_dataset = input_fn(  
        is_training=True,  
        data_dir=FLAGS.data_dir,  
        batch_size=FLAGS.batch_size,  
        num_epochs=FLAGS.epochs,  
        dtype=tf.float32  
    )  


# in your model.fit() function  
model.fit(  
        train_input_dataset,  
        steps_per_epoch= (_NUM_TRAINING_IMAGES//FLAGS.batch_size)// hvd.size(),  
        epochs=FLAGS.epochs,  
        callbacks=callbacks,  
        verbose=verbose)  