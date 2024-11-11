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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from keras import backend as K

import argparse
class DQN_agent():
    def __init__(self, env,learning_rate,epsilon,temp,policy,gamma,architecture_type):
        self.env = env       
        self.gamma =gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.temp = temp
        self.total_states = self.env.observation_space.shape[0]
        self.total_actions = self.env.action_space.n
        self.architecture_type=architecture_type
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.policy=policy
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        

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
        if self.architecture_type==1:
            model = Sequential()
            model.add(Dense(24, input_dim=self.total_states, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.total_actions, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        if self.architecture_type==2:
            print('hello2')
            model = Sequential()
            model.add(Dense(24, input_dim=self.total_states, activation='relu'))
            
            model.add(Dense(self.total_actions, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        if self.architecture_type==3:
            model = Sequential()
            model.add(Dense(48, input_dim=self.total_states, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.total_actions, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        if self.architecture_type==4:
            model = Sequential()
            model.add(Dense(24, input_dim=self.total_states, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(48, activation='relu'))
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
                probs=self.softmax(Q_values[0] ,self.temp)
                action = np.random.choice(self.total_actions, p=probs)
                
                       
        return action
    
    def remember(self, state, action, reward, new_state, done):
        ''' function to append in the memory'''
        self.memory.append([state, action, reward, new_state, done])
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    # def replay_without_tn(self, batch_size):
    #     ''' without target netowrk '''
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
               
    #             target = (reward + self.gamma * np.max(self.model.predict(next_state)[0]))
    #         Q_values = self.model.predict(state)
    #         Q_values[0][action] = target
    #         self.model.fit(state, Q_values, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    # def replay_with_tn(self, batch_size):
    #     ''' with target network'''
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         Q_values = self.model.predict(state)
    #         if done:
    #             Q_values[0][action] = reward
    #         else:
    #             # a = self.model.predict(next_state)[0]
    #             t = self.target_model.predict(next_state)[0]
    #             Q_values[0][action] = reward + self.gamma * np.amax(t)
    #             # target[0][action] = reward + self.gamma * t[np.argmax(a)]
    #         self.model.fit(state,  Q_values, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
            
    def replay_without_tn(self,batch_s):
        
        # Randomly sample minibatch from the memory
        m_batch = random.sample(self.memory, min(len(self.memory), batch_s))

        state = np.zeros((batch_s, self.total_states))
        next_state = np.zeros((batch_s, self.total_states))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_s):
            state[i] = m_batch[i][0]
            action.append(m_batch[i][1])
            reward.append(m_batch[i][2])
            next_state[i] = m_batch[i][3]
            done.append(m_batch[i][4])

        # do batch prediction to save speed
        Q_values = self.model.predict(state)
        t= self.model.predict(next_state)

        for i in range(batch_s):
            # correction on the Q value for the action used
            if done[i]:
                Q_values[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                Q_values[i][action[i]] = reward[i] + self.gamma * (np.amax(t[i]))

        # Train the Neural Network with batches
        self.model.fit(state, Q_values, batch_size=batch_s, verbose=0)
        
        
    def replay_with_tn(self,batch_s):
        
        # Randomly sample minibatch from the memory
        m_batch = random.sample(self.memory, min(len(self.memory), batch_s))

        state = np.zeros((batch_s, self.total_states))
        next_state = np.zeros((batch_s, self.total_states))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_s):
            state[i] = m_batch[i][0]
            action.append(m_batch[i][1])
            reward.append(m_batch[i][2])
            next_state[i] = m_batch[i][3]
            done.append(m_batch[i][4])

        # do batch prediction to save speed
        Q_values = self.model.predict(state)
        t= self.target_model.predict(next_state)

        for i in range(batch_s):
            # correction on the Q value for the action used
            if done[i]:
                Q_values[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                Q_values[i][action[i]] = reward[i] + self.gamma * (np.amax(t[i]))

        # Train the Neural Network with batches
        self.model.fit(state, Q_values, batch_size=batch_s, verbose=0)
        
        
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        # TAU=0.01
        # model_weights = self.model.get_weights()
        # target_model_weights = self.target_model.get_weights()
        # for i in range(len(model_weights)):
        #     target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        # self.target_model.set_weights(target_model_weights)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        

def average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False):

    reward_results = np.empty([n_repetitions,episodes]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        name=str(rep)+'_'+name            
        
        if 'architecture' in name:
            if '1' in name:
                agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma,1)
            if '2' in name:
                agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma,2)
                print('hello')
            if '3' in name:
                agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma,3)
            if '4' in name:
                agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma,4)
                
        else:
            agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma,1)
            
        rewards=train_model(episodes, agent, ER, TN,episode_len, name,batch_size = 32)

        reward_results[rep] = rewards[0:episodes]
        del agent
        K.clear_session()
        
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

def train_model(episodes, agent, ER, TN,episode_len, name,batch_size = 32):
    env = agent.env
    gamma = agent.gamma

    if Path(name+".npy").is_file():
        score_ar=list(np.load(name+".npy"))
        # print('yes')
        print(name)
    
    else:
        score_ar=[]
    
    if Path(name+"_cartpole-dqn.h5").is_file():
        agent.load(name+"_cartpole-dqn.h5")
        # print('yes')
   
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
                        # agent.update_target_model()
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
                    agent.model.fit(state, Q_values, verbose=0,batch_size=32)
                    
                    # Update the score, state, and epsilon
                    score += reward
                    state = next_state
                    agent.epsilon *= agent.epsilon_decay
                    agent.epsilon = max(agent.epsilon, agent.epsilon_min)
                    if done:
                        break
                    
                    # agent.update_target_model()
                #without target network    
                else:
                    Q_values = agent.model.predict(state)
                    # Q_values[0][action] = reward + gamma * np.max(model.predict(next_state)[0])
                    if done:
                        Q_values[0][action] = reward
                    else:
                        Q_values[0][action] = reward + gamma * np.max(agent.model.predict(next_state)[0])
                    # Train the DQN model with the updated Q-value
                    agent.model.fit(state, Q_values, verbose=0,batch_size=32)
                    
                    # Update the score, state, and epsilon
                    score += reward
                    state = next_state
                    agent.epsilon *= agent.epsilon_decay
                    agent.epsilon = max(agent.epsilon, agent.epsilon_min)
                    if done:
                        break
                    
        if episode % 10 == 0:
            agent.update_target_model()
        if episode % 50 == 0:
            agent.save(name+"_"+str(episode)+"_cartpole-dqn.h5")
            agent.save(name+"_cartpole-dqn.h5")
            np.save(name,score_ar)
            
        # Print the score at the end of each episode
        print("Episode:", episode, " Score:", score)
        score_ar.append(score)
    # Close the environment
    env.close()
    return score_ar


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--exp_type', type=int, default=0)
parser.add_argument('--ER', default=False)
parser.add_argument('--TN', default=False)
parser.add_argument('--episodes', default=250)

def main(exp_type,ER,TN,episodes):
   # %% simple training with fixed parameters
    if exp_type==1:
        print(0)
        env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
        learning_rate=0.001
        epsilon=1
        temp=1
        name='train'
        policy='egreedy'
        gamma=0.95
        episode_len=3
        n_repetitions=1
        agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma,1)
        print(ER)
        print(TN)
        rewards=train_model(episodes, agent, ER, TN,episode_len, name,batch_size = 32)
        mean=np.mean(rewards)
        print('mean_reards:'+str(mean))
   # %% comparison of DQN with others
    if exp_type==2:
         
        env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
        learning_rate=0.001
        epsilon=1
        temp=1
        policy='egreedy'
        gamma=0.95
        n_repetitions=2
        # episodes=500
       
        name='conf1'
        learning_curve=average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=False, TN=False,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
        
        name='conf2'
        learning_curve_er=average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=False,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
        
        name='conf3'
        learning_curve_tn=average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=False, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
     
        name='conf4'
        learning_curve_er_tn=average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
        
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
    if exp_type==3:
        
        env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
        policies=['egreedy','softmax']
        learning_rate=0.001
        epsilon=1
        temp=1
        
        gamma=0.95
        
        n_repetitions=2
        # episodes=500
       
        Plot = LearningCurvePlot(title = 'Different learning rates')    
        
        for policy in policies:
            # agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
            name=str(policy)
            learning_curve = average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
            Plot.add_curve(learning_curve,label=r'policy = {} '.format(policy))
       
        Plot.save('policies.png')
    
    
    
   
    #%% different learning rate
    if exp_type==4:
        env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
        learning_rates=[0.001,0.005,0.1]
        epsilon=1
        temp=1
        policy='egreedy'
        gamma=0.95
        
        n_repetitions=2
        # episodes=200
       
        Plot = LearningCurvePlot(title = 'Different learning rates')    
        
        for learning_rate in learning_rates:
            name=str('lr_')+str(learning_rate)
            # agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
            learning_curve = average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
            Plot.add_curve(learning_curve,label=r'learning_rate = {} '.format(learning_rate))
       
        Plot.save('learning_rates.png')
    
    #%% different gamma
    if exp_type==5:
        env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
        gammas=[0.3,0.6,0.95]
        learning_rate=0.001
        epsilon=1
        temp=1
        policy='egreedy'
        
        
        n_repetitions=2
        # episodes=500
       
        Plot = LearningCurvePlot(title = 'Different gamma')    
        
        for gamma in gammas:
            name=str('gamma_')+str(gamma)
            # agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
            learning_curve = average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
            Plot.add_curve(learning_curve,label=r'$\gamma$ = {} '.format(gamma))
       
        Plot.save('gamma.png')
        
    if exp_type==6:
        env = gym.make('CartPole-v1') #Add render_mode = "human" for rendering the GUI
        arch_type=[1,2,3,4]
        learning_rate=0.001
        epsilon=1
        temp=1
        policy='egreedy'
        gamma=0.05
        
        n_repetitions=2
        # episodes=500
       
        Plot = LearningCurvePlot(title = 'Different neural architecture')    
        
        for t in arch_type:
            name=str('architecture_type_')+str(t)
            # agent=DQN_agent(env,learning_rate,epsilon,temp,policy,gamma)
            learning_curve = average_over_repetitions(n_repetitions, episodes, name,env,learning_rate,epsilon,temp,policy,gamma, ER=True, TN=True,episode_len = 500, batch_size = 32, smoothing_window=51, plot=False)
            Plot.add_curve(learning_curve,label=r'architecture_type = {} '.format(t))
       
        Plot.save('Different neural architecture.png')
       
        
    #%%
if __name__ == '__main__':
    args = parser.parse_args()
    conf = vars(args)
    main(**conf)

