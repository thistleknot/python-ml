#from __future__ import print_function

import multiprocessing
import os
import pickle
import gym
import numpy as np
import neat
import random

#import cart_pole
#import visualize

runs_per_net = 2 
simulation_seconds = 60.0

#env = gym.make("Taxi-v3").env
# reset environment to a new, random state

total_rewards, total_epochs, total_penalties = 0, 0, 0
episodes = 100

# Hyperparameters
delta = .5
alpha = 1 - delta
gamma = 1 - alpha
#alpha = 0.1
#gamma = 1-.5
#gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make("Taxi-v3").env
        
        q_table = np.zeros([env.observation_space.n, env.action_space.n])
        #sim = cart_pole.CartPole()
        state = env.reset()
        #observation = env.reset()

        # Run the given simulation for up to num_steps time steps.

        rewards = 0
        epochs = 0
        penalties, reward = 0, 0
        fitness = 0.0
        '''
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t
        '''
        done = False

        #frames = []        
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                #action = np.argmax(q_table[state]) # Exploit learned values
                action = np.argmax(net.activate(q_table[state]))
            
            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            #print(observation)

            if reward == -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            '''
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
                }
            )
            '''

            rewards += reward
            epochs += 1 
        print(rewards)
            
        fitness = penalties/epochs
        print(fitness)
    
        #total_rewards +- rewards
        #total_penalties += penalties
        #total_epochs += epochs            
        
        #print("Action Space {}".format(env.action_space))
        #print("Observation Space {}".format(env.observation_space))   
        #print(f"Average rewards per episode: {total_rewards / runs_per_net}")
        #print(f"Average timesteps per episode: {total_epochs / runs_per_net}")
        #print(f"Average penalties per episode: {total_penalties / runs_per_net}")        

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(os.getcwd())
    config_path = os.path.join(local_dir, 'code\config-feedforward.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()