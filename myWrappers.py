import gym
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from torchvision import transforms

class StackEnv():
    """ Virtual Wrapper """

    def __init__(self, env, input_shape=(84,84), action_space=5, depth=4,
                 skipframes=4, mergeframes=4, cropframe=True, initial_skipframe=50):

        self.env = env #the parent class of env
        self.input_shape = input_shape #dimension of the output
        self.action_space = action_space #valid action space
        self.depth = depth #how many frames per stack
        self.max_skipframes = skipframes #how many frame to skip
        self.mergeframes = mergeframes
        self.cropframe = cropframe
        self.life_count = -1 #storing game info like lives
        self.lostlife_reward = -50 #if die
        self.initial_skipframe = 50 #skip initial idle frames

        # changing the size and and gray scale the input
        self.transform_step1 = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Grayscale(num_output_channels=1)
                               ])
        self.transform_step3 = transforms.Compose([
                               transforms.Resize(self.input_shape),
                               transforms.ToTensor()
                               ])

        # deque for final states output
        self.states_stack = deque([np.zeros(self.input_shape) for z in range(self.depth)],
                            maxlen=self.depth)

        #previous frames for merging; to be merged with recent frame
        self.states_buffer = np.zeros((self.mergeframes,) + env.observation_space.shape,
                             dtype=np.uint8)

        self.last_merged_states_max = None

        assert(skipframes >= mergeframes)

    def encode_state(self, s):
        s = self.transform_step1(s)
        if self.cropframe:
            s = transforms.functional.crop(s, 0, 0, 174, 160) #this is step 2
        s = self.transform_step3(s)
        return s

    def states2stackedInputs(self, current_buffer):

        encoded_stacks = [self.encode_state(s) for s in current_buffer]

        stacked_states = np.vstack(encoded_stacks)
        merged_states_max = np.max(stacked_states, axis=0)

        self.states_stack.append(merged_states_max) #last in the stack and first is out hence
        self.last_merged_states_max = merged_states_max #for rendering

        #convert to tensor for input
        state_inputs = np.array(self.states_stack) #4,84,84

        return state_inputs

    def reset(self):
        # reset the parent env
        initial_state = self.env.reset()
        self.life_count = -1
        # skip the initial idle frames before game starts
        for _ in range(self.initial_skipframe):
            a = np.random.choice(self.action_space)
            _, _, _, _ = self.env.step(a)

        state_inputs, _, _, _ = self.step(a)

        return state_inputs

    def step(self, action):
        #skip thru frames and sum rewards; get last states and done
        reward_total = 0
        life_lost = False
        startframe = self.max_skipframes - self.mergeframes #4-2 = 2

        for i in range(self.max_skipframes):

            next_state, reward, done, info = self.env.step(action)

            # handle lost life reward change
            if self.life_count != -1:
                #current life count < previous life count -> dead
                if info['ale.lives'] < self.life_count:
                    self.life_count = info['ale.lives']
                    reward += self.lostlife_reward
                    life_lost = True
            else: #new
                self.life_count = info['ale.lives']

            # accummulate merge frames
            if i >= startframe:
                self.states_buffer[i-startframe,:,:,:] = next_state

            #add awards
            reward_total += reward
        #print(self.life_count, reward_total)
        next_state_inputs = self.states2stackedInputs(self.states_buffer)

        return next_state_inputs, reward_total, done, life_lost

    def render(self, proccessed=False):
        if proccessed:
            return np.array(self.last_merged_states_max * 255, dtype = np.uint8)
        else:
            return self.env.render(mode='rgb_array')

    def close(self):
        self.env.close()
