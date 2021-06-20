import os
from time import sleep
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import gym
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from my_mpo.actor import Actor
from my_mpo.critic import Critic
from my_mpo.replaybuffer import ReplayBuffer


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_det = Σi.det()  # (B,)
    Σ_det = Σ.det()  # (B,)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    Σi_det = torch.clamp_min(Σi_det, 1e-6)
    Σ_det = torch.clamp_min(Σ_det, 1e-6)
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ_det / Σi_det) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ, torch.mean(Σi_det), torch.mean(Σ_det)


class MPO(object):
    def __init__(self, env, args):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.device = args.device
        self.ε_dual = args.dual_constraint
        self.εμ = args.kl_mean_constraint
        self.εΣ = args.kl_var_constraint
        self.γ = args.discount_factor
        self.α_μ_scale = args.alpha_mean_scale # scale Largrangian multiplier
        self.α_Σ_scale = args.alpha_var_scale  # scale Largrangian multiplier
        self.α_μ_max = args.alpha_mean_max
        self.α_Σ_max = args.alpha_var_max

        self.sample_episode_num = args.sample_episode_num
        self.sample_episode_maxstep = args.sample_episode_maxstep
        self.sample_action_num = args.sample_action_num
        self.batch_size = args.batch_size
        self.episode_rerun_num = args.episode_rerun_num
        self.mstep_iteration_num = args.mstep_iteration_num
        self.evaluate_period = args.evaluate_period
        self.evaluate_episode_num = args.evaluate_episode_num
        self.evaluate_episode_maxstep = args.evaluate_episode_maxstep

        self.actor = Actor(env).to(self.device)
        self.critic = Critic(env).to(self.device)
        self.target_actor = Actor(env).to(self.device)
        self.target_critic = Critic(env).to(self.device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.norm_loss_q = nn.MSELoss() #nn.SmoothL1Loss()

        self.replaybuffer = ReplayBuffer()

        self.η = np.random.rand()
        self.η_μ = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.η_Σ = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.max_return_eval = -np.inf
        self.start_iteration = 1
        self.render = False

    def train(self, iteration_num=1000, log_dir='log', model_save_period=50, render=False):

        self.render = render

        model_save_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.start_iteration, iteration_num + 1):
            self.__sample_trajectory(self.sample_episode_num)
            buffer_size = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            max_kl_μ = []
            max_kl_Σ = []
            max_kl = []
            mean_Σ_det = []

            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(SubsetRandomSampler(range(buffer_size)), self.batch_size, drop_last=True),
                        desc='training {}/{}'.format(r+1, self.episode_rerun_num)):
                        
                    K = len(indices)  # the sample number of states
                    N = self.sample_action_num  # the sample number of actions per state

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)  # (K, state_dim)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)  # (K, action_dim) or (K,)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)  # (K, state_dim)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)  # (K,)

                    # Policy Evaluation
                    # [2] 3 Policy Evaluation (Step 1)
                    loss_q, q = self.__update_critic_td( state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # E-Step of Policy Improvement
                    # [2] 4.1 Finding action weights (Step 2)
                    with torch.no_grad():
                        # sample N actions per state
                        b_μ, b_A = self.target_actor.forward(state_batch)  # (K,)
                        b = MultivariateNormal(b_μ, scale_tril=b_A)  # (K,)
                        sampled_actions = b.sample((N,))  # (N, K, action_dim)
                        expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, state_dim)
                        target_q = self.target_critic.forward(
                            expanded_states.reshape(-1, self.state_dim), sampled_actions.reshape(-1, self.action_dim)  # (N * K, action_dim)
                        ).reshape(N, K)  # (N, K)
                        target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, N)
                        
                    # https://arxiv.org/pdf/1812.02256.pdf
                    # [2] 4.1 Finding action weights (Step 2)
                    #   Using an exponential transformation of the Q-values
                    def dual(η):
                        """
                        dual function of the non-parametric variational
                        Q = target_q_np  (K, N)
                        g(η) = η*ε + η*mean(log(mean(exp(Q(s, a)/η), along=a)), along=s)
                        For numerical stabilization, this can be modified to
                        Qj = max(Q(s, a), along=a)
                        g(η) = η*ε + mean(Qj, along=j) + η*mean(log(mean(exp((Q(s, a)-Qj)/η), along=a)), along=s)
                        """
                        max_q = np.max(target_q_np, 1)
                        return η * self.ε_dual + np.mean(max_q) \
                            + η * np.mean(np.log(np.mean(np.exp((target_q_np - max_q[:, None]) / η), axis=1)))
                    
                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    # normalize
                    qij = torch.softmax(target_q / self.η, dim=0)  # (N, K) or (action_dim, K)

                    # M-Step of Policy Improvement
                    # [2] 4.2 Fitting an improved policy (Step 3)
                    for _ in range(self.mstep_iteration_num):
                        μ, A = self.actor.forward(state_batch)
                        # First term of last eq of [2] p.5
                        # see also [2] 4.2.1 Fitting an improved Gaussian policy
                        π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (K,)
                        π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (K,)
                        loss_p = torch.mean(
                            qij * (
                                π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                                + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                            )
                        )
                        mean_loss_p.append((-loss_p).item())

                        Cμ, CΣ, Σi_det, Σ_det = gaussian_kl( μi=b_μ, μ=μ, Ai=b_A, A=A)
                        max_kl_μ.append(Cμ.item())
                        max_kl_Σ.append(CΣ.item())
                        mean_Σ_det.append(Σ_det.item())

                        if np.isnan(Cμ.item()):  # This should not happen
                            raise RuntimeError('Cμ is nan')
                        if np.isnan(CΣ.item()):  # This should not happen
                            raise RuntimeError('CΣ is nan')

                        # Update lagrange multipliers by gradient descent
                        # this equation is derived from last eq of [2] p.5,
                        # just differentiate with respect to α
                        # and update α so that the equation is to be minimized.
                        self.η_μ -= self.α_μ_scale * (self.εμ - Cμ).detach().item()
                        self.η_Σ -= self.α_Σ_scale * (self.εΣ - CΣ).detach().item()

                        self.η_μ = np.clip(0.0, self.η_μ, self.α_μ_max)
                        self.η_Σ = np.clip(0.0, self.η_Σ, self.α_Σ_max)

                        self.actor_optimizer.zero_grad()
                        # last eq of [2] p.5
                        loss_l = -( loss_p + self.η_μ * (self.εμ - Cμ) + self.η_Σ * (self.εΣ - CΣ))
                        mean_loss_l.append(loss_l.item())
                        loss_l.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.1)
                        self.actor_optimizer.step()
                    
            self.update_target_actor_critic()

            
            self.save_model(it, os.path.join(model_save_dir, 'model_latest.pt'))
            if it % model_save_period == 0:
                self.save_model(it, os.path.join(model_save_dir, 'model_{}.pt'.format(it)))

            ################################### evaluate and save logs #########################################
            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)
            max_kl_μ = np.max(max_kl_μ)
            max_kl_Σ = np.max(max_kl_Σ)
            mean_Σ_det = np.mean(mean_Σ_det)

            print('iteration :', it)
            if it % self.evaluate_period == 0:
                self.actor.eval()
                return_eval = self.__evaluate()
                self.actor.train()
                self.max_return_eval = max(self.max_return_eval, return_eval)
                print('  max_return_eval :', self.max_return_eval)
                print('  return_eval :', return_eval)
                writer.add_scalar('max_return_eval', self.max_return_eval, it)
                writer.add_scalar('return_eval', return_eval, it)
            print('  mean return :', mean_return)
            print('  mean reward :', mean_reward)
            print('  mean loss_q :', mean_loss_q)
            print('  mean loss_p :', mean_loss_p)
            print('  mean loss_l :', mean_loss_l)
            writer.add_scalar('mean_return', mean_return, it)
            writer.add_scalar('mean_reward', mean_reward, it)
            writer.add_scalar('loss_q', mean_loss_q, it)
            writer.add_scalar('loss_p', mean_loss_p, it)
            writer.add_scalar('loss_l', mean_loss_l, it)
            writer.add_scalar('mean_q', mean_est_q, it)
            writer.add_scalar('η', self.η, it)
            writer.add_scalar('max_kl_μ', max_kl_μ, it)
            writer.add_scalar('max_kl_Σ', max_kl_Σ, it)
            writer.add_scalar('mean_Σ_det', mean_Σ_det, it)
            writer.add_scalar('α_μ', self.η_μ, it)
            writer.add_scalar('α_Σ', self.η_Σ, it)

            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def load_model(self, path=None):
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.start_iteration = checkpoint['iteration'] + 1
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()

    def save_model(self, it, path=None):
        data = {
            'iteration': it,
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(data, path)

    def __sample_trajectory_worker(self, i):
        buff = []
        state = self.env.reset()
        for steps in range(self.sample_episode_maxstep):
            action = self.target_actor.action( torch.from_numpy(state).type(torch.float32).to(self.device)).cpu().numpy()
            next_state, reward, done, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))
            if self.render and i == 0:
                self.env.render(mode='human')
                sleep(0.01)
            if done:
                break
            else:
                state = next_state
        return buff

    def __sample_trajectory(self, sample_episode_num):
        self.replaybuffer.clear()
        episodes = [self.__sample_trajectory_worker(i)
                    for i in tqdm(range(sample_episode_num), desc='sample_trajectory')]
        self.replaybuffer.store_episodes(episodes)

    def __evaluate(self):
        with torch.no_grad():
            total_rewards = []
            for e in tqdm(range(self.evaluate_episode_num), desc='evaluating'):
                total_reward = 0.0
                state = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):
                    action = self.actor.action(
                        torch.from_numpy(state).type(torch.float32).to(self.device)
                    ).cpu().numpy()
                    state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
                total_rewards.append(total_reward)
            return np.mean(total_rewards)


    def __update_critic_td(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=64):
        B = state_batch.size(0)
        with torch.no_grad():
            r = reward_batch  # (B,)

            ## get mean, cholesky from target actor --> to sample from Gaussian
            π_μ, π_A = self.target_actor.forward(next_state_batch)  # (B,)
            π = MultivariateNormal(π_μ, scale_tril=π_A)  # (B,)
            sampled_next_actions = π.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, action_dim)
            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, state_dim)
            
            ## get expected Q value from target critic
            expected_next_q = self.target_critic.forward(
                expanded_next_states.reshape(-1, self.state_dim),  # (B * sample_num, state_dim)
                sampled_next_actions.reshape(-1, self.action_dim)  # (B * sample_num, action_dim)
            ).reshape(B, sample_num).mean(dim=1)  # (B,)
            
            y = r + self.γ * expected_next_q
        self.critic_optimizer.zero_grad()
        t = self.critic( state_batch, action_batch).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, y

    def update_target_actor_critic(self):
        # param(target_actor) <-- param(actor)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # param(target_critic) <-- param(critic)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
