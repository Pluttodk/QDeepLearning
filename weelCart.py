import numpy as np
import gym
import time
env = gym.make("MountainCar-v0").env
print(env.observation_space)
#Learning parameters
alpha = 0.7
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episode = 10000
test_episode = 1000
max_steps = 100

#Method simply for showing game
def visualise_env():
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

#visualise_env()

#Converts the observation to states
def obs_to_state(obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low)/40
    a = int((obs[0]-env_low[0])/env_dx[0])
    b = int((obs[1]-env_low[1])/env_dx[1])
    return a,b

#Initialize Q-table
Q = np.zeros((40,40, env.action_space.n))

#Keeping track of rewards and epsilon values
training_rewards = []
epsilons = []

for episode in range(train_episode):
    #Reseting environment state
    state = env.reset()
    
    total_training_rewards = 0

    for step in range(100):
        #Choosing between exploit or explore
        tradeoff = np.random.uniform(0,1)
        a,b = obs_to_state(state)
        #Exploitation
        if tradeoff > epsilon:
            action = np.argmax(Q[a,b,:])
        #Exploration -> Choose state at random
        else:
            action = env.action_space.sample()

        #Calculating the reward of taking that step
        new_state, reward, is_done, info = env.step(action)
        
        n_a, n_b = obs_to_state(new_state)
        #Update Q table:
        Q[a,b,action] = Q[a,b, action] + alpha*(reward+discount_factor*np.max(Q[n_a,n_b, :])-Q[a,b,action])

        #Update total reward
        total_training_rewards += reward
        state = new_state

        #Check if successful
        if is_done:
            break

    #important that we scale down the epsilon to favor exploitation
    epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay*episode)

    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)

print(f"Training score over time: {sum(training_rewards)/train_episode}")

is_done = False
state = env.reset()
while(not is_done):
    a,b = obs_to_state(state)
    action = np.argmax(Q[a,b,:])
    env.render()
    state, _, is_done, _ = env.step(action)
env.close()
