
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


import numpy as np


def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def readfile(file):

    for serialized_example in tf.python_io.tf_record_iterator(file):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        image = example.features.feature['image_raw'].float_list.value
        #label = example.features.feature['label_raw'].float_list.value
        label=[0]


        
        example = tf.train.Example(features = tf.train.Features(feature={
            'label_raw':tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                'image_raw': tf.train.Feature(float_list=tf.train.FloatList(value=image))}))             

        writer.write(example.SerializeToString())
        
    writer.close()


#readfile(r'g:\python\data-change\evt_1000_1.bin.tfrecords',r'g:\python\data-change\evt_1000_129.bin.tfrecords')
tf.app.flags.DEFINE_string('fname', 'evt.root',
                           'Name of root file with muon events stored in tree')
tf.app.flags.DEFINE_integer('startcounter', 51,
                            'Counter of file names, ')
FLAGS = tf.app.flags.FLAGS
def main(argv):
    #fname='g:\python\data-change'
    #readfile('g:\python\junonn-classifer\evt_1000_50.bin.tfrecords')
    filenames = [os.path.join(FLAGS.fname, 'evt_1000_%d.bin.tfrecords' % i) for i in xrange(50)]
    for i in range(len(filenames)) :
        child = filenames[i]
        writer = tf.python_io.TFRecordWriter('evt_1000_'+str(50)+'.bin'+'.tfrecords')
        
        readfile(child,writer)
        




if __name__ == '__main__':
    tf.app.run()