import torch
from src.agent import SPRAgent
from src.planner import ModelBasedPlanner


class PlanningSPRAgent(SPRAgent):
    """
    SPRAgent that uses the learned transition model for planning during eval.
    Training sampling stays unchanged (noisy nets exploration).
    """

    def __init__(self,
                 planning_horizon=3,
                 planning_top_k=5,
                 planning_discount=0.99,
                 planning_reward_weight=1.0,
                 planning_value_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._planning_horizon = planning_horizon
        self._planning_top_k = planning_top_k
        self._planning_discount = planning_discount
        self._planning_reward_weight = planning_reward_weight
        self._planning_value_weight = planning_value_weight

        self._planner = None
        self._planning_active = False

    def initialize(self, *args, **kwargs):
        """agent 初始化后创建 planner（此时 model 已经构建好了）。"""
        super().initialize(*args, **kwargs)
        model = getattr(self.model, "module", self.model)  # handle DDP
        self._planner = ModelBasedPlanner(
            model=model,
            horizon=self._planning_horizon,
            top_k=self._planning_top_k,
            discount=self._planning_discount,
            reward_weight=self._planning_reward_weight,
            value_weight=self._planning_value_weight,
        )

    def sample_mode(self, itr):
        """训练采样：关闭 planning，走 noisy nets 探索。"""
        super().sample_mode(itr)
        self._planning_active = False

    def eval_mode(self, itr):
        """Eval：启用 planning。"""
        super().eval_mode(itr)
        self._planning_active = True

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        rlpyt 每步调这个方法选 action。
        eval 时用 planner，训练时用原来的 Q-based。
        """
        # 先让 parent 跑一遍，拿到格式正确的 agent_info（rlpyt collector 需要）
        original_action, agent_info = super().step(observation, prev_action, prev_reward)

        if self._planning_active and self._planner is not None:
            obs = observation.to(self.device)
            if obs.dim() == 5:
                obs = obs.flatten(-4, -3)     # [B, F*C, H, W]
            elif obs.dim() == 4:
                obs = obs.flatten(-4, -3)     # [F*C, H, W]
                obs = obs.unsqueeze(0)        # [1, F*C, H, W]
            best_action = self._planner.plan(obs)
            action = best_action.view_as(original_action)
            return action, agent_info
        else:
            return original_action, agent_info