'''
Created on 21.05.2018

@author: modalg
'''
import tensorflow as tf
import os

class Estimator():
    """Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """
    def __init__(self, scope="estimator", summaries_dir=None, valid_actions=None):
        self.valid_actions=valid_actions
        self.scope = scope
        # writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # build the graph
            self._build_model()
            # manage directory for summaries
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
                
    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        # placeholders for our input
        # our input are 4 RGB frames of shape 84, 84 each
        # TODO 1 create a placeholder for self.X_pl with shape [None, 84, 84, 4] with dtype tf.uint8
        # the TD target value
        # TODO 2 create a placeholder for self.y_pl for shape [None] with dtype tf.float32
        # integer id of which action was selected
        # TODO 3 create a placeholder for self.actions_pl for shape [None] with dtype tf.int32
        # remove the pixel scale from the images
        # TODO 4 convert self.X_pl to float and divide by 255.0
        # first dimension of X_pl is the batch size
        # TODO 5 extract the batch size
        
        # three convolutional layers
        
        # TODO 6 create a conv net with tf.contrib.layers. conv2d with following features:
        # 32 feature maps, kernel_size=8, stride=4, Rectified Linear (ReLu) activation function
        # 64 feature maps, kernel_size=4, stride=2, Rectified Linear (ReLu) activation function
        # 64 feature maps, kernel_size=3, stride=1, Rectified Linear (ReLu) activation function
        
        # TODO 7 flatten the last convolutional layer
        # TODO 8 create a fully connected layer for with 512 neurons
        
        # TODO 9 create a fully connected layer with length of possible actions and assign to self.predictions
        
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        # calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        # calculate the mean loss of the batch
        self.loss = tf.reduce_mean(self.losses)
        # optimizer Parameters from original paper
        # TODO 10 assign self.optimizer to tf.train.RMSPropOptimizer with following features: 
        # learning_rate=0.00025
        # decay=0.99
        # momentum=0.0
        # epsilon=1e-6
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        # summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])
        
    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })
    
    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.
        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]
        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss