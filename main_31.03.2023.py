import os
os.environ['CUDA_VISIBLE_DEVICES']=""

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class DQN_agent():
    def __init__(self, env,learning_rate,epsilon,temp,policy,gamma):
        self.env = env       
        self.gamma =gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.temp = temp
        self.total_states = self.env.observation_space.shape[0]
        self.total_actions = self.env.action_space.n
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.policy=policy
        
    def softmax(self,x, temp):
        ''' Computes the softmax of vector x with temperature parameter 'temp' '''
        x = x / temp # scale by temperature
        z = x - max(x) # substract max to prevent overflow of softmax 
        return np.exp(z)/np.sum(np.exp(z)) # compute softmax

    def argmax(self,x):
        ''' Own variant of np.argmax with random tie breaking '''
        try:
            return np.random.choice(np.where(x == np.max(x))[0])
        except:
            return np.argmax(x)
        
    def build_model(self):
        ''' only 2 layers for now '''
        model = Sequential()
        model.add(Dense(24, input_dim=self.total_states, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.total_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        

    def select_action(self, state):
        ''' action selection function '''
        if self.policy == 'egreedy':
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")
                
           
            else:
                if np.random.uniform() < self.epsilon:
                    action= np.random.randint(self.total_actions)
                else:
                    Q_values= self.model.predict(state)
                    action = np.argmax(Q_values[0]) 
            
        elif self.policy == 'softmax':
            if self.temp is None:
                raise KeyError("Provide a temperature")
            
            else:
                # exp_values = np.exp(self.Q_sa[s]  / temp)
                # probs = exp_values / np.sum(exp_values)
                Q_values = self.model.predict(state)
                probs=self.softmax(Q_values ,self.temp)
                action = np.random.choice(self.total_actions, p=probs)
                
                       
        return action
    
    def remember(self, state, action, reward, new_state, done):
        ''' function to append in the memory'''
        self.memory.append([state, action, reward, new_state, done])
    
    def replay_without_tn(self, batch_size):
        ''' without target netowrk '''
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
               
                target = (reward + self.gamma * np.max(self.model.predict(next_state)[0]))
            Q_values = self.model.predict(state)
            Q_values[0][action] = target
            self.model.fit(state, Q_values, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def replay_with_tn(self, batch_size):
        ''' with target network'''
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            Q_values = self.model.predict(state)
            if done:
                Q_values[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                Q_values[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state,  Q_values, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        

def average_over_repetitions(n_repetitions, episodes, agent, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False):

    reward_results = np.empty([n_repetitions,episodes]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        rewards=train_model(episodes, agent, ER, TN,episode_len, batch_size = 32)

        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def train_model(episodes, agent, ER, TN,episode_len, batch_size = 32):
    env = agent.env
    gamma = agent.gamma
    score_ar=[]
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.total_states])
        done = False
        score = 0
        
        for t in range(episode_len):
            # Update the Q-value for the chosen action with replay buffer
            if ER==True:
                
                # Choose an action based on epsilon-greedy policy
                action=agent.select_action(state)
                
                # Take the chosen action and observe the next state, reward, and done flag
                next_state, reward, done, _ ,_= env.step(action)
                # print(next_state)
                next_state = np.reshape(next_state, [1, agent.total_states])
                agent.remember(state, action, reward, next_state, done)
                state=next_state
                score += reward
                if len(agent.memory) > batch_size:
                    if TN==True:
                        agent.replay_with_tn(batch_size)
                        agent.update_target_model()
                    else:
                        agent.replay_without_tn(batch_size)
                if done:
                    break
                        
     
            # Update the Q-value for the chosen action without replay buffer
            else:
                action=agent.select_action(state)
                
                # Take the chosen action and observe the next state, reward, and done flag
                next_state, reward, done, _ ,_= env.step(action)
                # print(next_state)
                next_state = np.reshape(next_state, [1, agent.total_states])
                #with target network
                if TN==True:
                    Q_values = agent.model.predict(state)
                    
                    if done:
                        Q_values[0][action] = reward
                    else:
                        Q_values[0][action] = reward + gamma * np.max(agent.target_model.predict(next_state)[0])
                    # Train the DQN model with the updated Q-value
                    agent.model.fit(state, Q_values, verbose=0)
                    
                    # Update the score, state, and epsilon
                    score += reward
                    state = next_state
                    # epsilon *= epsilon_decay
                    # epsilon = max(epsilon, min_epsilon)
                    if done:
                        break
                    
                    agent.update_target_model()
                #without target network    
                else:
                    Q_values = agent.model.predict(state)
                    # Q_values[0][action] = reward + gamma * np.max(model.predict(next_state)[0])
                    if done:
                        Q_values[0][action] = reward
                    else:
                        Q_values[0][action] = reward + gamma * np.max(agent.model.predict(next_state)[0])
                    # Train the DQN model with the updated Q-value
                    agent.model.fit(state, Q_values, verbose=0)
                    
                    # Update the score, state, and epsilon
                    score += reward
                    state = next_state
                    # epsilon *= epsilon_decay
                    # epsilon = max(epsilon, min_epsilon)
                    if done:
                        break
                    
        # if episode % 10 == 0:
        #     agent.update_target_model()
        if episode % 50 == 0:
            agent.save("cartpole-dqn.h5")
            
        # Print the score at the end of each episode
        print("Episode:", episode, " Score:", score)
        score_ar.append(score)
    # Close the environment
    env.close()
    return score_ar
    
def main():
    #%% comparison of DQN with others
    env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
    learning_rate=0.005
    epsilon=0.05
    temp=1
    policy='egreedy'
    gamma=0.95
    n_repetitions=2
    episodes=500
    agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
    learning_curve=average_over_repetitions(n_repetitions, episodes, agent, ER=False, TN=False,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
    agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
    learning_curve_er=average_over_repetitions(n_repetitions, episodes, agent, ER=True, TN=False,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
    agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
    learning_curve_tn=average_over_repetitions(n_repetitions, episodes, agent, ER=False, TN=True,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
    agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
    learning_curve_er_tn=average_over_repetitions(n_repetitions, episodes, agent, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
    
    Plot = LearningCurvePlot(title = 'DQN vs DQN-ER') 
    Plot.add_curve(learning_curve,label='DQN')
    Plot.add_curve(learning_curve_er,label='DQN-ER')
    Plot.save('dqn_vs_dqn-er.png')
    
    Plot = LearningCurvePlot(title = 'DQN vs DQN-TN') 
    Plot.add_curve(learning_curve,label='DQN')
    Plot.add_curve(learning_curve_tn,label='DQN-TN')
    Plot.save('dqn_vs_dqn-tn.png')
    
    Plot = LearningCurvePlot(title = 'DQN vs DQN-ER-TN') 
    Plot.add_curve(learning_curve,label='DQN')
    Plot.add_curve(learning_curve_er_tn,label='DQN-ER-TN')
    Plot.save('dqn_vs_dqn-er-tn.png')
    
    
    #%% different policy
    env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
    policies=['egreedy','softmax']
    learning_rate=0.005
    epsilon=0.05
    temp=1
    policy='egreedy'
    gamma=0.8
    
    n_repetitions=2
    episodes=500
   
    Plot = LearningCurvePlot(title = 'Different learning rates')    
    
    for policy in policies:
        agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
        learning_curve = average_over_repetitions(n_repetitions, episodes, agent, ER=False, TN=False,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
        Plot.add_curve(learning_curve,label=r'policy = {} '.format(policy))
   
    Plot.save('policies.png')
    
    
    
   
    #%% different learning rate
    env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
    learning_rates=[0.001,0.005,0.1,0.5]
    epsilon=0.05
    temp=1
    policy='egreedy'
    gamma=0.8
    
    n_repetitions=2
    episodes=500
   
    Plot = LearningCurvePlot(title = 'Different learning rates')    
    
    for learning_rate in learning_rates:
        agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
        learning_curve = average_over_repetitions(n_repetitions, episodes, agent, ER=False, TN=False,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
        Plot.add_curve(learning_curve,label=r'learning_rate = {} '.format(learning_rate))
   
    Plot.save('learning_rates.png')
    
    #%% different gamma
    env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
    gammas=[0.5,0.8,1]
    learning_rate=0.005
    epsilon=0.05
    temp=1
    policy='egreedy'
    
    
    n_repetitions=2
    episodes=500
   
    Plot = LearningCurvePlot(title = 'Different gamma')    
    
    for gamma in gammas:
        agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
        learning_curve = average_over_repetitions(n_repetitions, episodes, agent, ER=False, TN=False,episode_len = 500, batch_size = 32, smoothing_window=10, plot=False)
        Plot.add_curve(learning_curve,label=r'$\gamma$ = {} '.format(gamma))
   
    Plot.save('gamma.png')
    
    #%%
if __name__ == '__main__':
    main()
