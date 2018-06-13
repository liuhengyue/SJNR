import tensorflow as tf
import numpy as np
from model import Model
from PIL import Image
import sys, os
from fnmatch import fnmatch
import itertools
import matplotlib.pyplot as plt
from pprint import pprint
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.app.flags.DEFINE_string('path', None, 'Path to image test folder')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
FLAGS = tf.app.flags.FLAGS


def main(_):
    path = FLAGS.path
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    files = getTestFiles(path)
    num = 0
    correct = 0
    image_paths = []
    number_predictions = []
    labels = []
    for key in files:
        print key
        # print files['1']
        class_num = len(files[key])
        num = num + class_num
        images = [tf.image.decode_jpeg(tf.read_file(image_dir), channels=3) for image_dir in files[key]]
        # print images[0], images[1]
        # image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
        for i in range(0, len(files[key])):
            im = Image.open(files[key][i])
            image_paths += [files[key][i]]
            width, height = im.size
            # resize method 1
            # images[i] = tf.image.crop_to_bounding_box(images[i], height / 5, width / 2 - 32, 64, 64)
            # resize method 2 (float image)
            # images[i] = tf.image.resize_image_with_crop_or_pad(images[i], int(min(width, height)/1.5), int(min(width, height)/1.5))
            # images[i] = tf.image.resize_images(images[i], [64, 64])/255.0
            # images[i] = tf.image.resize_images(images[i], [64, 64])
            # resize method 3 for cropped images
            images[i] = tf.image.resize_image_with_crop_or_pad(images[i], 54, 54)
	    # print str(i) + ' ' + files[key][i]
        # stack multiple images
        images = tf.stack(images)
        # image = tf.image.resize_images(image, [64, 64])
        # images = tf.reshape(images, [-1, 64, 64, 3])
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.multiply(tf.subtract(images, 0.5), 2)
        #images = tf.image.resize_images(images, [54, 54])
        images = tf.reshape(images, [-1, 54, 54, 3])
	#display(images[124])
        #figures = {'img' + str(i): images[i] for i in range(len(files[key]))}
        # print figures
        #plot_figures(figures)
        # images = tf.unstack(images)
        # print images
        # for image in images:
        length_logits, digits_logits = Model.inference(images, drop_rate=0.0)
        length_predictions = tf.argmax(length_logits, axis=1)
        digits_predictions = tf.argmax(digits_logits, axis=2)
        digits_predictions_string = tf.reduce_join(tf.as_string(digits_predictions), axis=1)

        with tf.Session() as sess:
            restorer = tf.train.Saver()
            restorer.restore(sess, path_to_restore_checkpoint_file)

            length_predictions_val, digits_predictions_string_val = sess.run(
                [length_predictions, digits_predictions_string])
            # print 'length: ', length_predictions_val
            # print 'digits: ', digits_predictions_string_val
            predictions = [int(d[:l]) for l, d in itertools.izip(length_predictions_val, digits_predictions_string_val)]
            # print predictions
            number_predictions += predictions
            labels += [int(key)] * class_num
            # class_correct = sum([1.0 for p in predictions if p == key])
            # correct = correct + class_correct
            # print 'Accuracy: ', class_correct / class_num
        # break
    acc, acc_op = tf.metrics.accuracy(labels = tf.Variable(labels), predictions = tf.Variable(number_predictions))
    # auc, auc_op = tf.metrics.auc(labels = tf.Variable(labels), predictions = tf.Variable(number_predictions))
    pre, pre_op = tf.metrics.precision(labels = tf.Variable(labels), predictions = tf.Variable(number_predictions))
    recall, recall_op = tf.metrics.recall(labels = tf.Variable(labels), predictions = tf.Variable(number_predictions))
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    sess.run(acc_op)
    #sess.run(auc_op)
    sess.run(pre_op)
    sess.run(recall_op)
    print(labels)
    print(number_predictions)
    # sess.run(init)    
    print("Number of test images: " + str(num) + ".")
    print("Accuracy: " + str(sess.run(acc)))
    print("Precision: " + str(sess.run(pre)))
    print("Recall: " + str(sess.run(recall)))
    confusion_matrix(image_paths, labels, number_predictions)
    # print("Area Under the Curve (AUC): " + str(sess.run(auc)))
    # print '%d total Accuracy: %f' % (num, correct / num)


# length_prediction_val = length_predictions_val[0]
# digits_prediction_string_val = digits_predictions_string_val[0]
# print 'length: %d' % length_prediction_val
# print 'digits: %s' % digits_prediction_string_val

def confusion_matrix(image_paths, labels, predictions):
    file = open("data.txt", "w")
    lines = [str(l)+" "+str(p)+" "+str(d)+"\n" for l, p, d in itertools.izip(labels, predictions, image_paths)]
    file.writelines(lines)
    file.close()

def display(images):
    sess = tf.Session()
    plt.imshow(np.asarray(images.eval(session=sess)))
    plt.imshow(sess.run(images))
    plt.show()


def getTestFiles(root):
    test_set = {}
    subdirs = next(os.walk(root))[1]
    if not subdirs:
        test_set['test'] = [os.path.join(root, name) for name in next(os.walk(root))[2] if fnmatch(name, "*.png")]
        return test_set
    for subdir in subdirs:
        # print subdir
        path = os.path.join(root, subdir)
        # print path
        subfiles = [os.path.join(path, name) for name in next(os.walk(path))[2] if fnmatch(name, "*.png")]
        test_set[subdir] = subfiles
    return test_set

def plot_figures(figures):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    sess = tf.Session()
    ncols = 5
    nrows = len(figures)/ncols + 1
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        # axeslist.ravel()[ind].imshow(sess.run(figures[title]), cmap=plt.jet())

        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
        fig.add_subplot(nrows, ncols, ind + 1)
        plt.axis('off')
        plt.imshow(sess.run(figures[title]))
    # plt.tight_layout() # optional
    plt.show()

if __name__ == '__main__':
    tf.app.run(main=main)
