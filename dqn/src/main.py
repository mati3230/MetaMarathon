'''
Created on 21.05.2018
24 TODOs
@author: modalg
'''
import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from state_processor import StateProcessor
from estimator import Estimator

if "../" not in sys.path:
    sys.path.append("../")
#from collections import deque, namedtuple
from collections import namedtuple

# hyperparameter
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_episodes", 10000, """Number of episodes to run for training""")
tf.app.flags.DEFINE_string("experiment_dir", "./experiments/", """Directory to save Tensorflow summaries""")
tf.app.flags.DEFINE_integer("replay_memory_size", 500000, """Size of the replay memory""")
tf.app.flags.DEFINE_integer("replay_memory_init_size", 50000, """Number of random experiences to sampel when initializing the reply memory""")
tf.app.flags.DEFINE_integer("update_target_estimator_every", 10000, """Copy parameters from the Q estimator to the target estimator every N steps""")
tf.app.flags.DEFINE_float("discount_factor", 0.99, """Lambda time discount factor""")
tf.app.flags.DEFINE_float("epsilon_start", 1.0, """Chance to sample a random action when taking an action. Epsilon is decayed over time and this is the start value""")
tf.app.flags.DEFINE_float("epsilon_end", 0.1, """The final minimum value of epsilon after decaying is done""")
tf.app.flags.DEFINE_integer("epsilon_decay_steps", 500000, """Number of steps to decay epsilon over""")
tf.app.flags.DEFINE_integer("batch_size", 32, """Size of batches to sample from the replay memory""")
tf.app.flags.DEFINE_boolean("record_video", False, """Decide if video should be recorded while training""")
tf.app.flags.DEFINE_integer("record_video_every", 50, """Record a video every N episodes""")
tf.app.flags.DEFINE_boolean("train", False, """if True: autonomous agent will play game""")
tf.app.flags.DEFINE_string("environment", "Breakout-v0", """Environment that should be adapted - you may have to change StateProcessor and Estimator image sizes when you change the environment""")

# dictionary 
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    # get all variables from estimator 1
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    # sort parameter as key value pairs
    e1_params = sorted(e1_params, key=lambda v: v.name)
    
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    
    # list of update operations
    update_ops = []
    # iterate over all variables of estimator 1 and estimator 2
    for e1_v, e2_v in zip(e1_params, e2_params):
        # plan assignment operation e2_v = e1_v
        op = e2_v.assign(e1_v)
        # append operation in the list
        update_ops.append(op)
    # execute all assignments
    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        # predict best action in this state (observation)
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        # return the indice of the highest q_value -> best action
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(sess=None,
                    env=None,
                    q_estimator=None,
                    target_estimator=None,
                    state_processor=None,
                    valid_actions=None):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # all the variables outputted by the environment will be stored in a transition
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # the replay memory
    replay_memory = []
    # keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(FLAGS.num_episodes), # how many steps did the model play in one episode
        episode_rewards=np.zeros(FLAGS.num_episodes)) # rewards collected while playing in one episode
    
    
    # create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(FLAGS.experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(FLAGS.experiment_dir, "monitor")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
        
        
    saver = tf.train.Saver()
    # load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())
    # the epsilon decay schedule 
    epsilons = np.linspace(FLAGS.epsilon_start, FLAGS.epsilon_end, FLAGS.epsilon_decay_steps)
    # the policy we are following
    # assignment of the make_epsilon_greedy_policy function
    policy = make_epsilon_greedy_policy(q_estimator,len(valid_actions))
    # populate the replay memory with initial experience
    # fill the replay memory buffer
    print("Populating replay memory...")
    # reset the environment to an initial state
    # TODO 1 reset the environment and get state
    # TODO 2 preprocess the game image with state_processor and assign to state
    # combine 4 images to a state 
    state = np.stack([state] * 4, axis=2)
    for i in range(FLAGS.replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, FLAGS.epsilon_decay_steps-1)])
        # choose a random action with probability of action_probs
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        # TODO 3 apply the action to make a step and get next_state
        # TODO 4 preprocess the next_state image with state_preprocessor
        # append the next state image on last position in stack
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        # save the transition in memory
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done: # game over
            # TODO 5 reset the environment and get state
            # TODO 6 apply preprocessing to state with state_preprocessor
            state = np.stack([state] * 4, axis=2)
        else:
            # the next state will be the current state in the next iteration
            # TODO 7 set the state to next_state
            print("set the state to next_state")
    # record videos
    if FLAGS.record_video:
        # video recording - add env Monitor wrapper
        env = Monitor(env, directory=monitor_path, video_callable=lambda count: count % FLAGS.record_video_every == 0, resume=True)
    # play num_episodes games
    for i_episode in range(FLAGS.num_episodes):
        saver.save(tf.get_default_session(), checkpoint_path)
        # TODO 8 reset the environment and get state
        # TODO 9 apply preprocessing to state with state_preprocessor
        state = np.stack([state] * 4, axis=2)
        loss = None
        # one step in the environment
        # t will count up till game is lost
        for t in itertools.count():
            # epsilon for this time step
            epsilon = epsilons[min(total_t, FLAGS.epsilon_decay_steps-1)]
            
            # add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            # add episode_summary to q_estimator.summary_writer
            q_estimator.summary_writer.add_summary(episode_summary, total_t)
            
            # maybe update the target estimator when total_t reaches FLAGS.update_target_estimator_every
            if total_t % FLAGS.update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")
            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, FLAGS.num_episodes, loss), end="")
            # write everything in the buffer to the terminal, even if normally it would wait before doing so
            sys.stdout.flush()
            
            # take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # TODO 10 apply the action to make a step and get next_state
            # TODO 11 preprocess the next_state image with state_preprocessor
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            
            # if our replay memory is full, pop the first element
            if len(replay_memory) == FLAGS.replay_memory_size:
                replay_memory.pop(0)
            
            # save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   
            
            # update statistics - save total reward and timestep in this episode
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # sample a minibatch from the replay memory
            samples = random.sample(replay_memory, FLAGS.batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            
            # calculate q values and targets
            q_values_next = target_estimator.predict(sess, next_states_batch)
            # best actions in next state (bellman equation)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * FLAGS.discount_factor * np.amax(q_values_next, axis=1)
            
            # perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            if done:
                break
            state = next_state
            total_t += 1
        
        # episode over
        # add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()
        # save stats of this episode
        yield total_t, EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])
    return stats

def bot_play(sess=None,
                env=None,
                q_estimator=None,
                target_estimator=None,
                state_processor=None,
                num_episodes=10,
                valid_actions=None):
    """
    Test runs with rendering and logger.infos the total score
    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
    """
    checkpoint_dir = os.path.join(FLAGS.experiment_dir, "checkpoints")
    saver = tf.train.Saver()
    saver.restore(sess, "./experiments/{0}/checkpoints/model".format(env.spec.id))
    """
    # load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, "./experiments/Breakout-v0/checkpoints/model")
    """
    # TODO 12 reset the environment and get state
    # TODO 13 apply preprocessing to state with state_preprocessor
    # combine 4 images to a state 
    state = np.stack([state] * 4, axis=2)
    sum_reward = 0
    
    
    # the policy we are following
    # assignment of the make_epsilon_greedy_policy function
    policy = make_epsilon_greedy_policy(q_estimator,len(valid_actions))
    
    for episode in range(num_episodes):
        while(True):
            # TODO render a frame from the environment
            # calculate q values and targets
            action_probs = policy(sess, state, 0.1)
            # choose a random action with probability of action_probs (Hint: use np.random.choice)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # TODO 14 apply the action to make a step and get next_state
            # TODO 15 preprocess the next_state image with state_preprocessor
            sum_reward += reward
            # append the next state image on last position in stack
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            
            if done: # game over
                # TODO 16 reset the environment and get state
                # TODO 17 apply preprocessing to state with state_preprocessor
                state = np.stack([state] * 4, axis=2)
                print("episode: {0}, reward: {1}".format(episode, sum_reward))
                break
            else:
                # the next state will be the current state in the next iteration
                # TODO 18 set the state to next_state
                print("set the state to next_state")

def main(argv=None):
    env = gym.make(FLAGS.environment)
    valid_actions = np.arange(env.action_space.n)
    tf.reset_default_graph()
    # where we save our checkpoints and graphs
    FLAGS.experiment_dir = os.path.abspath("{0}{1}".format(FLAGS.experiment_dir, env.spec.id))
    # create a global step variable
    global_step = tf.Variable(0, name="global_step", trainable=False)
        
    # create estimators - constructors will build the graph
    q_estimator = Estimator(scope="q", summaries_dir=FLAGS.experiment_dir, valid_actions=valid_actions)
    target_estimator = Estimator(scope="target_q", valid_actions=valid_actions)
    # state processor - component which preprocesses images
    state_processor = StateProcessor()
    # run the graph in training process
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.train:
            for t, stats in deep_q_learning(sess,
                                            env,
                                            q_estimator=q_estimator,
                                            target_estimator=target_estimator,
                                            state_processor=state_processor,
                                            valid_actions=valid_actions):
                print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
        else:
            bot_play(sess,
                    env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    state_processor=state_processor,
                    num_episodes=2,
                    valid_actions=valid_actions
                    )
    env.close()
    print("done")

# run main function
if __name__ == "__main__":
    tf.app.run()