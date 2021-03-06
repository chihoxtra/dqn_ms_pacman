
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
<ul>
<li>DQN: The input state is technically a discrete input as there are finite combinations of states.
However since the number of combination are huge, it is almost impossible to use traditional
temporal difference method to solve the problem. While there are some other methods like discretization
that can be used to 'discretize' the input state, here we use a neutral network as a functional approximator of the Q value. The DQN used here composed of 3 layers of convolutional layers followed by relu activation. The inputs are then flattened and then are connected by a fully connected layer to produce the action output (size of 9 in this case). <br>
  <ul>Key Learnings:
  <li><B>Pooling is not used. Unlike other image classification tasks where pooling is often used to send a 'summarized' characteristics of a certain region of the input to the next layer, here pooling might hinder the performance as it forces the network to send only representative signal to the next layer. </B>
  <li><B>Also while in most other cases a deeper and more complicated network will work better, here over complication can sometimes cause poor training performance. It is recommended that we choose a simpler network as simple as possible</B>
  </ul>
<li>Prioritized Experience Replay: The ordinary experience replay randomly pick samples from experience and learn from it. The problem is that not all experience are equally important to learning. Therefore Here we we give those experience with higher td error (as calculated by the difference between td target and td current) a higher priority. The probability is of samples being chosen are basically the td error magnitude normalized summation of all td errors magnitude across the entire experience memory. Note that 2 hyperparameters alpha and beta are applied here to adjust the 'degree of reliance on prioritized replay' and over weight adjustment when it comes to backward propagation.
<div align="center"><img width="201" height="105" src="https://github.com/chihoxtra/dqn_ms_pacman/blob/master/per.png"></div>
  <ul>Key Learnings:
  <li>The calculation of weight adjustment is very tricky. And that beta is supposed to gradually increase to 1.0 when training converges.
  <li>Here I tried to use 2 different data storage structure to store the memory:
  <ul>
  <li>deque from the collection packages. Pros: easy to understand and visualize. Cons. Could be slow when memory.
  <li>TreeSum. Using tree is much faster but it requires extreme care when doing the implementation. Here a data structure Tree is used to store and sort and add td errors. Another data structure called data is used to store object of experience. TreeSum works faster.
  </ul>
  <li><B>To avoid zero probability for some experience, a small value is added to all td error so that they wont end up having a zero value. I personally find that this value cannot be too small. A reasonable value could be 1e-3. </B>
  <li><b>Regarding TreeSum implementation, one key thing to notice is that earlier attempts of using recursion caused the programme to run exceeding the max recursion limit. Hence it is strongly suggested that the treesum function should be implemented using while loop instead of recursion.</b>
  <li><b>Since accessing the middle elements in deque is very slow, rotation functions are used to pop and append elements when needed. I am also using another array to store all td values for quicker calculation.</b>
  </ul>
<li>Double DQN: Here 50% of time we will use target network as choice of action and 50% of time we will use local network.
<li>Duel Network: A special attention has to be made on dimension of reduction of action. 
</ul>
