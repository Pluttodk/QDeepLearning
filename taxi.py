import numpy as np
import gym
import time
env = gym.make("Taxi-v3").env
print(env.observation_space)
#Learning parameters
alpha = 0.7
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episode = 2000
test_episode = 1000
max_steps = 100

#Method simply for showing game
def visualise_env():
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

#visualise_env()

#Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

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

        #Exploitation
        if tradeoff > epsilon:
            action = np.argmax(Q[state,:])
        #Exploration -> Choose state at random
        else:
            action = env.action_space.sample()

        #Calculating the reward of taking that step
        new_state, reward, is_done, info = env.step(action)

        #Update Q table:
        Q[state,action] = Q[state, action] + alpha*(reward+discount_factor*np.max(Q[new_state, :])-Q[state,action])

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
    action = np.argmax(Q[state,:])
    env.render()
    time.sleep(1)
    state, _, is_done, _ = env.step(action)
env.close()
