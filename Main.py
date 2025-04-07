import numpy as np
import os
import scipy.io
import Classes.Environment_Platoon as ENV
from ddpg_torch import Agent  # Update this to match your file name
import time
import matplotlib.pyplot as plt

'''
---------------------------------------------------------------------------------------
Simulation code with PPO clipping added to MADDPG framework
---------------------------------------------------------------------------------------
'''
start = time.time()
# ################## SETTINGS ######################
# [Your existing lane definitions and other settings remain the same]
# ################## SETTINGS ######################
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
print('------------- lanes are -------------')
print('up_lanes :', up_lanes)
print('down_lanes :', down_lanes)
print('left_lanes :', left_lanes)
print('right_lanes :', right_lanes)
print('------------------------------------')
width = 750 / 2
height = 1298 / 2
IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN
label = 'marl_model'
# ------------------------------------------------------------------------------------------------------------------ #
# simulation parameters:
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
size_platoon = 5
n_veh = 20  # n_platoon * size_platoon
n_platoon = int(n_veh / size_platoon)  # number of platoons
n_RB = 3  # number of resource blocks
n_S = 2  # decision parameter
Gap = 25 # meter
max_power = 30  # platoon leader maximum power in dbm ---> watt = 10^[(dbm - 30)/10]
V2I_min = 540  # minimum required data rate for V2I Communication = 3bps/Hz
bandwidth = int(180000)
V2V_size = int((4000) * 8)
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# PPO hyperparameters
# Increase clip parameter to allow larger policy updates
ppo_clip = 0.2  # Increased from 0.1

# Increase entropy coefficient to encourage more exploration
entropy_coef = 0.01  # Doubled from 0.005

# Adjust learning rates
ppo_alpha = 0.0001  # Increased actor learning rate
ppo_beta = 0.001    # Increased critic learning rate

# More updates per batch to utilize experience better
ppo_epochs = 5  # Increased from 3

# Reduce batch size for more frequent updates
batch_size = 64  # Reduced from 64

max_grad_norm = 0.5  # Gradient clipping threshold

# DDPG parameters

memory_size = 100000
gamma = 0.99
alpha = 0.0001
beta = 0.001


# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 1024
A_fc2_dims = 512
# ------------------------------

tau = 0.005  # Consider reducing tau for more stable updates (0.001-0.005)
env = ENV.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, size_platoon, n_RB,
                  V2I_min, bandwidth, V2V_size, Gap)
env.new_random_game()  # initialize parameters in env

n_episode = 500
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
def get_state(env, idx):
    """ Get state from the environment """

    V2I_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0

    V2V_abs = (env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60)/60.0

    V2I_fast = (env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[
        idx * size_platoon] + 10) / 35

    V2V_fast = (env.V2V_channels_with_fastfading[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1)), :]
                - env.V2V_channels_abs[idx * size_platoon, idx * size_platoon +
                                       (1 + np.arange(size_platoon - 1))].reshape(size_platoon - 1, 1) + 10) / 35

    Interference = (-env.Interference_all[idx] - 60) / 60

    AoI_levels = env.AoI[idx] / (int(env.time_slow / env.time_fast))

    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])

    # time_remaining = np.asarray([env.individual_time_limit[idx] / env.time_slow])

    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                           np.reshape(V2V_fast, -1), np.reshape(Interference, -1), np.reshape(AoI_levels, -1), V2V_load_remaining), axis=0)
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(get_state(env=env, idx=0))
n_output = 3  # channel selection, mode selection, power
# --------------------------------------------------------------
# Create directories for learning curves
current_dir = os.path.dirname(os.path.realpath(__file__))
os.makedirs(os.path.join(current_dir, "model/ppo_debug"), exist_ok=True)

# Create arrays to store learning metrics
value_losses = []
policy_losses = []
clip_fractions = []
episode_rewards = []

# [Your existing get_state function remains the same]

n_input = len(get_state(env=env, idx=0))
n_output = 3  # channel selection, mode selection, power
# --------------------------------------------------------------
agent = Agent(ppo_alpha, ppo_beta, n_input, tau, n_output, gamma, memory_size, C_fc1_dims, C_fc2_dims, C_fc3_dims,
              A_fc1_dims, A_fc2_dims, batch_size, n_platoon,
              clip_param=ppo_clip, ppo_epochs=ppo_epochs, entropy_coef=entropy_coef, max_grad_norm=max_grad_norm)

# Add a debug flag and gradient norm tracking
debug_mode = True  # Set to True for verbose outputs
total_grad_norm = []
AoI_evolution = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
Demand_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2I_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2V_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
power_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)

AoI_total = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_reward_ = np.zeros([n_episode], dtype=np.float16)
per_total_user_ = np.zeros([n_platoon, n_episode], dtype=np.float16)
if IS_TRAIN:
    # Optionally load pretrained models to fine-tune with PPO
    # agent.load_models()

    for i_episode in range(n_episode):
        done = False
        print("-------------------------------------------------------------------------------------------------------")
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)
        record_AoI = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
        per_total_user = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)

        env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
        env.individual_time_limit = env.time_slow * np.ones(n_platoon, dtype=np.float16)
        env.active_links = np.ones((int(env.n_Veh / env.size_platoon)), dtype='bool')
        if i_episode == 0:
            env.AoI = np.ones(int(n_platoon)) * 100

        # Renew environment more frequently for better exploration
        if i_episode % 10 == 0:  # Changed from 20 to 10
            env.renew_positions()  # update vehicle position
            env.renew_channel(n_veh, size_platoon)  # update channel slow fading
            env.renew_channels_fastfading()  # update channel fast fading

        state_old_all = []
        for i in range(n_platoon):
            state = get_state(env=env, idx=i)
            state_old_all.append(state)

        # Store episode metrics
        episode_reward = 0

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_platoon, n_output], dtype=int)

            # Choose actions with exploration noise
            action = agent.choose_action(np.asarray(state_old_all).flatten())
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)

            for i in range(n_platoon):
                action_all_training[i, 0] = ((action[0+i*n_output]+1)/2) * n_RB  # chosen RB
                action_all_training[i, 1] = ((action[1+i*n_output]+1)/2) * n_S  # Inter/Intra platoon mode
                action_all_training[i, 2] = np.round(np.clip(((action[2+i*n_output]+1)/2) * max_power, 1, max_power))  # power selected by PL

            # Execute actions in environment
            action_temp = action_all_training.copy()
            training_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand_R, V2V_success = \
                env.act_for_training(action_temp)

            record_reward[i_step] = global_reward.copy()
            episode_reward += global_reward

            for i in range(n_platoon):
                per_total_user[i, i_step] = training_reward[i]
                record_AoI[i, i_step] = env.AoI[i]

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            # Get new states
            for i in range(n_platoon):
                state_new = get_state(env, i)
                state_new_all.append(state_new)

            if i_step == n_step_per_episode - 1:
                done = True

            # Store experience in replay buffer
            agent.remember(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                           global_reward, np.asarray(state_new_all).flatten(), done)

            # Learn from experience
            agent.learn()

            # Store learning metrics
            if len(value_losses) < 10000:  # Limit storage to prevent memory issues
                value_losses.append(agent.value_loss)
                policy_losses.append(agent.policy_loss)
                clip_fractions.append(agent.policy_clip_fraction)

            # Update old states
            for i in range(n_platoon):
                state_old_all[i] = state_new_all[i]

            if debug_mode and i_step % 10 == 0:
                print("-----------------------------------")
                print(f'Episode: {i_episode}, Step: {i_step}')
                print(f'Global reward: {global_reward}')
                if hasattr(agent, 'policy_loss'):
                    print(f'Policy loss: {agent.policy_loss:.6f}')
                if hasattr(agent, 'value_loss'):
                    print(f'Value loss: {agent.value_loss:.6f}')
                if hasattr(agent, 'policy_clip_fraction'):
                    print(f'Clip fraction: {agent.policy_clip_fraction:.4f}')

        # Store episode reward
        episode_rewards.append(episode_reward / n_step_per_episode)
        record_reward_[i_episode] = np.mean(record_reward)
        per_total_user_[:, i_episode] = np.mean(per_total_user, axis=1)
        AoI_total[:, i_episode] = np.mean(record_AoI, axis=1)

        # Plot learning curves periodically
        if i_episode % 10 == 0 and i_episode > 0:
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')

            plt.subplot(2, 2, 2)
            plt.plot(value_losses[-1000:])  # Show recent losses
            plt.title('Critic Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')

            plt.subplot(2, 2, 3)
            plt.plot(policy_losses[-1000:])  # Show recent losses
            plt.title('Actor Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')

            plt.subplot(2, 2, 4)
            plt.plot(clip_fractions[-1000:])  # Show recent fractions
            plt.title('PPO Clip Fraction')
            plt.xlabel('Update Step')
            plt.ylabel('Fraction')

            plt.tight_layout()
            plt.savefig(os.path.join(current_dir, f"model/ppo_debug/learning_curves_{i_episode}.png"))
            plt.close()

        # Save models more frequently with PPO
        if i_episode % 20 == 0:
            agent.save_models()

            # Save learning metrics
            np.save(os.path.join(current_dir, "model/ppo_debug/value_losses.npy"), np.array(value_losses))
            np.save(os.path.join(current_dir, "model/ppo_debug/policy_losses.npy"), np.array(policy_losses))
            np.save(os.path.join(current_dir, "model/ppo_debug/clip_fractions.npy"), np.array(clip_fractions))
            np.save(os.path.join(current_dir, "model/ppo_debug/episode_rewards.npy"), np.array(episode_rewards))

    print('Training Done. Saving models...')
    agent.save_models()

    # Save all learning metrics and rewards as in your original code
    # [Your existing save code]

end = time.time()
print("simulation took this much time ... ", end - start)
