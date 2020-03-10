# Carla_simulator

## Where it comes from? 
code up to this point coming from :

https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/?fbclid=IwAR3r0_QOVQBMQAiiluF5MtEprn_uaPmQVLkEsLXcRRzEu0_X0EOvtFbpwqY

additional comments have been put and code separated into multiple classes

## What you need
Before being able to go further and run the code, it is necessary that you download everything needed to run carla 
https://carla.readthedocs.io/en/latest/start_quickstart/

Secondly, you need to download python 3.7 as Carla latest version for windows is only available for python 3.7. 

## Steps to gets started 
You may follow the videos 1 and 2 of the first link to get started as they show you how to start the simulation, how to spawn cars and even ride your own car in the environment. 

When you have got acquanted with it, follow the third video that explains you how to spawn your own car (make it go straight for 5 seconds and then dissapear. When you are done, you will be able to follow the reinforcement learning. 

As said before, I have already written all the code that has been written during the videos, however I have separated them to have a better visibility. 

We have 3 classes and 1 script to view our model. 
1. Car environment
2. DQN agent
3. self made tensorboard

To run the code, first you need to place this folder where you have installed your carla application. More precisely, in the folder **PythonAPI**, same folder as carla's examples. 

Next, you need to install the needed libraries to run your code, especially the following 
tqdm==4.43
tensorflow==1.14 (not higher otherwise will not work, or you will have to change multiple things)
Keras==2.2.5

To install them on your python 3.7, you can use the following command

```bash
py -3.7 -m pip install Keras=2.2.5
```

Next, open a command line and go into the folder you just added with the different scipts (an easy way to that is by going into the folder in _file explorer_ and type in navigation bar **cmd**. 

As you are in the command line type the following 

```bash
py -3.7 car_environment.py
```
 
if everything goes well, you should see a progress bar after a few seconds depending on your machine show the progress on the episodes of your machine learning. 

finally, when the learning is done, you can view your model graphics with tensorflow. 

To this, first you will need to add the tensorflow application to your path. Next, you can type the following command to open tensorflow web server

```bash
tensorboard --logdir==logs/
```

If everything goes well, a message should appear telling you where you can go to view you tensorboard. 

On your tensorboard 

