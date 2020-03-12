# How is code build
All information coming from pythonprogramming.net 

- car environment 
- DQNagent with keras using tensorflow

## car environment is some part of program that is made up of 2 main functions 

- step
- reset

The step function is a function whose role is to do something with the action it has been given and depending on the result of this last return the next observation, a certain reward based on how well it has performed during this step and finally a done flag.

The reset function is a function put at the beginning of the environment or whenever you want to rerun a new episode

Before I dive into the DNQ agent, I like to mention that this reinforcement learning is a difficult as our subject has to perform at the same time as he learns. This means we have to provide learning but the next prediction as well. 

## DNQ agent 

### Q learning
What is Q learning ? The idea is to have these Q values for every actions that you are going to take given a state.
It is a model-free form of machine learning, which means that it doesn't need to know or have the model of the environment that it will be in. 
For a given environment, everything is broken down into "states" and "actions." The states are observations and samplings that we pull from the environment, and the actions are the choices the agent has made based on the observation.

The way Q-Learning works is there's a "Q" value per action possible per state. This creates a table. In order to figure out all of the possible states, we can either query the environment (if it is kind enough to us to tell us)...or we just simply have to engage in the environment for a while to figure it out.

To update the q's in the Q-table , the formula is the following 
new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
where 
    - DISCOUNT : is a measure on how much we want to care about the FUTURE reward tha, the immediate reward (so distinction between logn term gains and short term gains) -> most of the time high because in q learning learning we strive to have a positive outcome at the end
    - max_future_q : grabbed after we've performed our action already, and then we update our previous values based partially on the next-step's best Q value 

### DQN agent 
With DQNs, instead of a Q Table to look up values, you have a model that you inference (make predictions from), and rather than updating the Q table, you fit (train) your model.
The DQN neural network model is a regression model, which typically will output values for each of our possible actions. These values will be continuous float values, and they are directly our Q values.

"where our Q-Learning algorithm could learn something in minutes, it will take our DQN hours. We will want to learn DQNs, however, because they will be able to solve things that Q-learning simply cannot...and it doesn't take long at all to exhaust Q-Learning's potentials."

https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/own-environment-q-learning-reinforcement-learning-python-tutorial/ 