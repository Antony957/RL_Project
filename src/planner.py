"""
Lightweight model-based planner using SPR's learned transition model.

For each candidate first action, rolls out N steps greedily (picking the
Q-argmax at each subsequent step), accumulates discounted predicted rewards,
and evaluates the leaf state with the Q-head.  Returns the first action
whose rollout yields the highest total value.

Complexity per decision: O(A * N) forward passes through the transition model,
where A = number of actions, N = planning horizon.
"""

import torch
import torch.nn.functional as F
from src.models import from_categorical, renormalize


class ModelBasedPlanner:
    def __init__(self,
                 model,
                 horizon=5,
                 discount=0.99,
                 reward_weight=1.0,
                 value_weight=1.0,
                 top_k=None):
        """
        Args:
            model:        trained SPRCatDqnModel (with conv, head, dynamics_model)
            horizon:      number of steps to roll out (should be <= jumps used in training)
            discount:     gamma for discounting future rewards and leaf Q
            reward_weight: weight on accumulated predicted rewards
            value_weight:  weight on leaf-node Q value
            top_k:        只对 Q 值最高的 k 个 action 做 rollout，
                          None 表示全部 action 都规划
        """
        self.model = model
        self.horizon = horizon
        self.discount = discount
        self.reward_weight = reward_weight
        self.value_weight = value_weight
        self.num_actions = model.num_actions
        self.top_k = top_k if top_k is not None else self.num_actions

    @torch.no_grad()
    def get_q_values(self, latent):
        """从 latent 算出每个 action 的 Q 值标量 [B, A]。"""
        p = self.model.head(latent)  # [B, A, n_atoms] or [B, A]
        if self.model.distributional:
            q = from_categorical(p, logits=False, limit=10)  # [B, A]
        else:
            q = p.squeeze(-1)
        return q

    @torch.no_grad()
    def predict_reward_scalar(self, reward_logits):
        """把 reward 分布转成标量 [B]。"""
        return from_categorical(F.softmax(reward_logits, -1),
                                logits=False, limit=1)

    @torch.no_grad()
    def rollout_from(self, latent, first_action):
        """
        给定初始 latent 和第一步 action，贪心 rollout horizon 步。

        Returns:
            total_value: [B] 累积折扣 reward + 折扣后的叶子 Q 值
        """
        B = latent.shape[0]
        device = latent.device
        cumulative_reward = torch.zeros(B, device=device)
        gamma_t = 1.0

        state = latent
        action = first_action  # [B]

        for step in range(self.horizon):
            # transition model: state + action -> next_state, reward_logits
            next_state, reward_logits = self.model.dynamics_model(state, action)

            # 累积折扣 reward
            r = self.predict_reward_scalar(reward_logits)
            cumulative_reward += gamma_t * r
            gamma_t *= self.discount

            state = next_state

            # 后续步：Q argmax（在所有 action 里贪心即可，这里开销小）
            if step < self.horizon - 1:
                q = self.get_q_values(state)
                action = q.argmax(-1)  # [B]

        # 叶子节点 Q 值
        leaf_q = self.get_q_values(state)
        leaf_value = leaf_q.max(-1).values

        total_value = (self.reward_weight * cumulative_reward
                       + self.value_weight * gamma_t * leaf_value)
        return total_value

    @torch.no_grad()
    def plan(self, observation):
        """
        主入口：给定 observation [B, C, H, W]，返回最优 action [B]。

        流程:
          1. encode observation -> latent
          2. Q head 算出所有 action 的 Q 值，取 top_k
          3. 只对这 k 个 candidate action 做 greedy rollout
          4. 选 total_value 最高的 action
        """
        self.model.eval()

        # encode: merge framestack and channel dims [B,F,C,H,W] -> [B,F*C,H,W]
        if observation.dim() == 5:
            observation = observation.flatten(-4, -3)
        obs = self.model.transform(observation, augment=False)
        latent = self.model.conv(obs)
        if self.model.renormalize:
            latent = renormalize(latent, -3)

        B = latent.shape[0]
        device = latent.device

        # Step 1: Q 值粗筛，选 top_k 个候选 action
        q_values = self.get_q_values(latent)                    # [B, A]
        k = min(self.top_k, self.num_actions)
        top_q, top_indices = q_values.topk(k, dim=1)            # [B, k]

        # Step 2: 只对 top_k action 做 rollout
        all_values = []
        for i in range(k):
            first_action = top_indices[:, i]                     # [B]
            value = self.rollout_from(latent, first_action)      # [B]
            all_values.append(value)

        all_values = torch.stack(all_values, dim=1)              # [B, k]

        # Step 3: 在 top_k 里选最优
        best_idx_in_k = all_values.argmax(dim=1)                 # [B]
        best_action = top_indices.gather(1, best_idx_in_k.unsqueeze(1)).squeeze(1)  # [B]
        return best_action

    @torch.no_grad()
    def plan_with_info(self, observation):
        """和 plan() 一样，但额外返回 debug 信息。"""
        self.model.eval()

        if observation.dim() == 5:
            observation = observation.flatten(-4, -3)
        obs = self.model.transform(observation, augment=False)
        latent = self.model.conv(obs)
        if self.model.renormalize:
            latent = renormalize(latent, -3)

        B = latent.shape[0]
        device = latent.device

        q_values = self.get_q_values(latent)                     # [B, A]
        k = min(self.top_k, self.num_actions)
        top_q, top_indices = q_values.topk(k, dim=1)             # [B, k]

        all_values = []
        for i in range(k):
            first_action = top_indices[:, i]
            value = self.rollout_from(latent, first_action)
            all_values.append(value)

        all_values = torch.stack(all_values, dim=1)              # [B, k]
        best_idx_in_k = all_values.argmax(dim=1)
        best_action = top_indices.gather(1, best_idx_in_k.unsqueeze(1)).squeeze(1)

        mf_action = q_values.argmax(dim=1)

        info = {
            "planning_values": all_values,              # [B, k]
            "candidate_actions": top_indices,            # [B, k]  被选中参与规划的 action
            "candidate_q": top_q,                        # [B, k]  它们的 Q 值
            "best_action": best_action,                  # [B]
            "model_free_q": q_values,                    # [B, A]
            "model_free_action": mf_action,              # [B]
            "agreement": (best_action == mf_action).float().mean().item(),
            "q_rank_of_chosen": None,  # planning 选的 action 在 Q 排名中的位置
        }

        # planning 选的 action 在原始 Q 排名中排第几
        q_ranks = q_values.argsort(dim=1, descending=True).argsort(dim=1)  # [B, A]
        info["q_rank_of_chosen"] = q_ranks.gather(1, best_action.unsqueeze(1)).float().mean().item()

        return best_action, info