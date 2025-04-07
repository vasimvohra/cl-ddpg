import os
import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import ActorNetwork, CriticNetwork
from Classes.noise import OUActionNoise
from Classes.buffer import ReplayBuffer


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma,
                 max_size, C_fc1_dims, C_fc2_dims, C_fc3_dims, A_fc1_dims, A_fc2_dims, batch_size, n_agents,
                 clip_param=0.2, ppo_epochs=5, entropy_coef=0.01, max_grad_norm=0.5, clip_value_loss=1.0):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions

        # PPO parameters
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss

        # Learning tracking
        self.learn_step_counter = 0
        self.policy_clip_fraction = 0
        self.value_loss = 0
        self.policy_loss = 0

        self.memory = ReplayBuffer(max_size, input_dims, n_actions, n_agents)
        self.noise = OUActionNoise(mu=np.zeros(n_actions * n_agents))

        self.actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                  n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                    n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                         n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                           n_actions=n_actions, name='target_critic')

        # Store the old actor for PPO comparison
        self.old_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                      n_actions=n_actions, name='old_actor')

        # Initialize target networks with current parameters
        self.update_network_parameters(tau=1)
        self.update_old_actor()

        # Track experience for importance sampling
        self.current_states = None
        self.current_actions = None
        self.action_probs = None

        # Adaptive learning parameters
        self.adaptive_entropy = entropy_coef
        self.min_entropy = 0.001
        self.success_counter = 0

    def update_old_actor(self):
        """Copy current actor parameters to old actor for PPO ratio calculation"""
        for target_param, param in zip(self.old_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)

        if evaluate:
            self.actor.train()
            return mu.cpu().detach().numpy()[0]

        # Store current state for easier importance sampling calculation
        self.current_states = state

        # Add noise with decaying scale based on learning progress
        if self.learn_step_counter > 5000:
            noise_scale = max(0.5, 1.0 - (self.learn_step_counter - 5000) / 10000)
        else:
            noise_scale = 1.0

        noise = T.tensor(self.noise() * noise_scale, dtype=T.float).to(self.actor.device)
        mu_prime = mu + noise

        # Store chosen action for PPO
        self.current_actions = mu_prime

        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

        # Adaptive entropy adjustment
        if reward > 0:
            self.success_counter += 1
            if self.success_counter >= 10:
                self.success_counter = 0
                # Gradually reduce entropy as we succeed
                self.adaptive_entropy = max(self.min_entropy, self.adaptive_entropy * 0.95)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.old_actor.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.old_actor.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.learn_step_counter += 1

        states, actions, rewards, states_, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        # Update critic more aggressively
        for _ in range(2):  # More critic updates than actor updates
            with T.no_grad():
                self.target_actor.eval()
                self.target_critic.eval()

                target_actions = self.target_actor.forward(states_)
                critic_value_ = self.target_critic.forward(states_, target_actions)
                critic_value_[done] = 0.0
                critic_value_ = critic_value_.view(-1)

                # Calculate Q-target with reward scaling to improve signal
                target = rewards + self.gamma * critic_value_
                target = target.view(self.batch_size, 1)

            # Update critic network
            self.critic.train()
            self.critic.optimizer.zero_grad()

            current_critic_value = self.critic.forward(states, actions)
            critic_loss = F.mse_loss(current_critic_value, target)

            # Apply value loss clipping if needed
            if self.clip_value_loss > 0:
                critic_loss = T.clamp(critic_loss, 0, self.clip_value_loss)

            critic_loss.backward()
            T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            self.value_loss = critic_loss.item()

        # PPO actor update
        clip_fractions = []
        actor_losses = []

        for epoch in range(self.ppo_epochs):
            self.actor.train()
            self.old_actor.eval()

            # Get current actions and old actions
            curr_actions = self.actor.forward(states)
            with T.no_grad():
                old_actions = self.old_actor.forward(states)

            # Calculate advantages
            with T.no_grad():
                old_value = self.critic.forward(states, old_actions)
                current_value = self.critic.forward(states, curr_actions)
                advantage = current_value - old_value
                # Normalize advantages
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Calculate action probabilities (approximated for continuous actions)
            curr_log_probs = -0.5 * ((curr_actions - actions) ** 2).sum(dim=1)
            old_log_probs = -0.5 * ((old_actions - actions) ** 2).sum(dim=1)

            # Importance sampling ratio
            ratio = T.exp(curr_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantage.detach()
            surr2 = T.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage.detach()

            # Track clip fraction
            clip_fraction = (T.abs(ratio - 1.0) > self.clip_param).float().mean().item()
            clip_fractions.append(clip_fraction)

            # Actor loss with entropy bonus
            action_std = T.std(curr_actions, dim=0).mean()
            entropy_bonus = self.adaptive_entropy * action_std

            actor_loss = -T.min(surr1, surr2).mean() - entropy_bonus
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor.optimizer.step()

        # Store metrics for monitoring
        self.policy_loss = np.mean(actor_losses)
        self.policy_clip_fraction = np.mean(clip_fractions)

        # Adaptively update the old actor - more frequently early in training
        if self.learn_step_counter % max(1, min(20, self.learn_step_counter // 500)) == 0:
            self.update_old_actor()

        # Update target networks
        self.update_network_parameters()

        # Print learning stats
        if self.learn_step_counter % 100 == 0:
            print(f"Step: {self.learn_step_counter}, "
                  f"Value Loss: {self.value_loss:.4f}, "
                  f"Policy Loss: {self.policy_loss:.4f}, "
                  f"Clip Fraction: {self.policy_clip_fraction:.4f}, "
                  f"Entropy Coef: {self.adaptive_entropy:.6f}")

            # Reset entropy coefficient occasionally if learning stagnates
            if self.policy_clip_fraction < 0.05 and self.learn_step_counter > 2000:
                print("Resetting entropy coefficient to encourage exploration")
                self.adaptive_entropy = self.entropy_coef  # Reset to initial value

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
