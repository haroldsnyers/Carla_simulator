import numpy as np
import time
import random

from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from TensorBoardClass import ModifiedTensorBoard
import tensorflow as tf

IM_WIDTH = 640
IM_HEIGHT = 480

REPLAY_MEMORY_SIZE = 5_000 # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000 # Minimum number of steps in a memory to start training  
MINIBATCH_SIZE = 16 # How many steps (samples) to use for training
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4

UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = "Xception"

MIN_REWARD = -200

DISCOUNT = 0.99

# how many choices (actions) -> number of neurons at output
ACTION_SPACE = 3


class DQNAgent:
    def __init__(self):
        # we are using those two models to have some sort of consistency in those .predict, because at the same 
        # time this model is trying to fit at the same time with initially some random places which leads the model
        # to go all over the place as it tries to figure things out 
        # stability in the training will be set by the batch (replay memory) and in the prediction with the target model

        # main model -> model that we are fitting every single step, so that we train every single step
        self.model = self.create_model()

        # Target model -> model that we predict every single step
        # this model is than updated every number of episodes/steps which is why we put the weights of target model equal 
        # to the weights of the model 
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # the model is taking prediction consistent with model over time
        # avoid overfitting with only one sample, this replay memory acts like a batch
        # so we take randomly selected data from this replay memory which will in turn from the batch
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0  # will track when it's time to update the target model, so that our predictions are reliable/stable 
        self.graph = tf.get_default_graph()

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self):
        # could use another one
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        # adding to out layer
        x = GlobalAveragePooling2D()(x)

        # adding the 3 neuron output (each possible action for the agent to take)
        predictions = Dense(ACTION_SPACE, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # grab our current and future q values
        # Recall a transition is: transition = (current_state, action, reward, new_state, done)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        # create our input X and output Y
        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here (not takin ginto account old states)
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # fitting
        # Fit on all samples as one batch, log only on terminal state
        with self.graph.as_default():
            # we're setting the tensorboard callback, only if log_this_step is true
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        # Update target network counter every episode
        if log_this_step:
            self.target_update_counter += 1

        # look if it's time to update model
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # method to get q values, thus make a prediction
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

