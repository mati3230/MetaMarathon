'''
Created on 21.05.2018

@author: modalg
'''
import tensorflow as tf

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            # input image
            # 210: width of image of the game
            # 160: height of image of the game
            # 3: rgb of image of the game
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            # convert image to grayscale
            # TODO 1 use tf.image module to convert input to grayscale
            # extract only relevant game window
            # TODO 2 crop image to bounding box y1: 34, x1: 0, y2: 160, x2: 160
            # resize image to quadratic form expected by convolutional neural network
            # TODO 3 resize image to 84 with NEAREST_NEIGHBOR method
            self.output = tf.squeeze(self.output)
            
    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })