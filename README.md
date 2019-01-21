
# DQN with Prioritized Experience Replay, Dual and Double Network on Atari Ms Pacman

<p align="center"><a href="https://gym.openai.com/envs/MsPacman-v0/">
 <img width="342" height="450" src="https://github.com/chihoxtra/dqn_ms_pacman/blob/master/mspacman.png"></a>
</p>

### Project Summary

This repository summarized humble learning experience on using a deep Q network to train an agent to play the classic atari Ms Pacman game. This project used the following techniques:
* Prioritized Experience Replay
* Double DQN
* Dual Network

### About OpenAI Ms Pacman

[Open AI](https://gym.openai.com/envs/#classic_control) provides many interesting environments for
developers to work on: Ms Pacman is one of them. There are 2 versions of Ms Pacman environments:
* provide the ram status (1 dimensional array) as observations
* provide the raw screenshot (210 x 160 x 3) as observations <br>
This project used the second environment.

### Required Packages

Apart from standard python 3.6, numpy, pytorch, you will also need the OpenAI gym package to run the environment, you can follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

### Implementation Summary
The following implementation were made according to the paper published by [Google Deep Mind](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

#### Data Preparation:
The observations inputs provided by the environment to the agent is in the form of
a screenshot image (210, 160, 3) where 3 is the RGB channel. The following procedures were made
in the 'Wrapper.py' files to pre-process the inputs:
<ul>
<li>Frame Skipping: The original game was a video game and so the FPS was relatively high. However chances between consecutive frames are not really that big. Thus here we are only taking input on every 4 frames. This will dramatically reduce number of inputs without diluting important signals from the inputs.
<li>The image is converted to grayscale.
<li>The image is rescaled to 64 x 64 to so that it can be trained faster in GPU as squared images utilize GPU resource better.
<li>Here odd number frame are super imposed with even number frame. The reason for this is that
at the time Atari was created, there is a strict limitation on RAM. To save RAM, the game
does not display all sprites in every frame. Take a look at the 2 consecutive frames below and you
can notice the bottom left/bottom right power dots are not consistently existing in these frames. To
avoid confusion to agent, we superimposed odd frame with even frame by taking the max values to create one "combined frame".
<div align="center"><img width="402" height="258" src="https://github.com/chihoxtra/dqn_ms_pacman/blob/master/oddevenframes.png"></img></div><br>
<li>To allow the agent to understand the sequential relationship between frames, a stack of 4
'combined frames' are stacked together as an input. That is first layer would be combined frame
of t3 and t4, second layer would be t7 and t8 and so on. The final input size is 64 x 64 x 4.
</ul>

#### Technique Used and Key Learnings:
- DQN: The input state is technically a discrete input as there are finite combinations of states.
However since the number of combination are huge, it is almost impossible to use traditional
temporal difference method to solve the problem. While there are some other methods like discretization
that can be used to 'discretize' the input state, here we use a neutral network as a functional approximator of the Q value. The DQN is composed of layers of convolutional inputs. The inputs are then flattened and then are connected by a fully connected layers. <br>
<B>Key Learnings: Pooling is not used. Unlike other image classification tasks where pooling is often used to send a 'summarized' signal of a certain region of the input to the next layer, here pooling might hinder the performance as it forces the network to send only representative signal to the next layer. Also while in most other cases a deeper and more complicated network will work better, here over complication can sometimes cause poor training performance. It is recommended that we choose a simpler network as simple as possible</B>
- Prioritized Experience Replay: The ordinary experience replay randomly pick same from experience and learn from it. Here we we give those experience with higher td error a priority of higher probability of being chosen. The probability is normalized td error magnitude across all entries in the experience method.
<p align="center">
 <img width="201" height="105" src="https://github.com/chihoxtra/dqn_ms_pacman/blob/master/per.png">
</p>
Here I tried to use 2 different data storage structure to store the memory:
* deque from the collection packages. Pros: easy to understand and visualize. Cons. Could be slow when memory<br>
<B>- Key Learnings: To avoid zero probability for some experience, a small value is added to all td error so that they wont end up having a zero value. I personally find that this value cannot be too small. A reasonable value could be 1e-3. </B>
