"""
PlanningSPRAgent: drop-in replacement for SPRAgent.

把原来 train.py 里的:
    from src.agent import SPRAgent
    agent = SPRAgent(ModelCls=SPRCatDqnModel, model_kwargs=config["model"], **config["agent"])

改成:
    from src.planner_integration import PlanningSPRAgent
    agent = PlanningSPRAgent(
        ModelCls=SPRCatDqnModel,
        model_kwargs=config["model"],
        planning_horizon=3,          # rollout 步数, <= jumps
        planning_top_k=5,            # Q 值粗筛保留几个 action
        planning_warmup_itrs=5000,   # 前 5000 itr 不用 planning（transition model 还没学好）
        use_planning_in_train=False, # 训练采样用原来的 noisy nets，eval 才用 planning
        **config["agent"],
    )

其余代码（sampler, algo, runner）完全不用动。
"""

import torch
from collections import namedtuple
from src.agent import SPRAgent
from src.planner import ModelBasedPlanner


AgentInfo = namedtuple("AgentInfo", ["p"])


class PlanningSPRAgent(SPRAgent):
    """
    SPRAgent that uses the learned transition model for planning.
    Training sampling stays unchanged (noisy nets exploration).
    Planning activates during eval after a warmup period.
    """

    def __init__(self,
                 planning_horizon=3,
                 planning_top_k=5,
                 planning_warmup_itrs=5000,
                 use_planning_in_train=False,
                 planning_discount=0.99,
                 planning_reward_weight=1.0,
                 planning_value_weight=1.0,
                 **kwargs):
        """
        Args:
            planning_horizon:       rollout 步数，建议 <= 训练时的 jumps
            planning_top_k:         先用 Q 值筛出 top_k 个 action 再 rollout
            planning_warmup_itrs:   前这么多 itr，即使 eval 也不用 planning
                                    （因为 transition model 还没学好）
            use_planning_in_train:  训练采样时是否也用 planning（一般 False）
            planning_discount:      rollout 时的折扣因子
            planning_reward_weight: 累积 reward 的权重
            planning_value_weight:  叶子节点 Q 值的权重
        """
        super().__init__(**kwargs)
        self._planning_horizon = planning_horizon
        self._planning_top_k = planning_top_k
        self._planning_warmup = planning_warmup_itrs
        self._use_planning_in_train = use_planning_in_train
        self._planning_discount = planning_discount
        self._planning_reward_weight = planning_reward_weight
        self._planning_value_weight = planning_value_weight

        self._planner = None
        self._planning_active = False
        self._current_itr = 0

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
        """训练采样阶段：一般关闭 planning，走 noisy nets 探索。"""
        super().sample_mode(itr)
        self._current_itr = itr
        self._planning_active = self._use_planning_in_train and self._past_warmup()

    def eval_mode(self, itr):
        """Eval 阶段：warmup 之后启用 planning。"""
        super().eval_mode(itr)
        self._current_itr = itr
        self._planning_active = self._past_warmup()

    def _past_warmup(self):
        return self._current_itr >= self._planning_warmup

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        rlpyt 每步调这个方法选 action。
        planning_active 时用 planner，否则用原来的 Q-based 选择。
        """
        if self._planning_active and self._planner is not None:
            obs = observation.to(self.device)
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            best_action = self._planner.plan(obs)           # [B]
            action = best_action.squeeze(0)                 # scalar tensor
            agent_info = AgentInfo(p=torch.zeros(1))
            return action, agent_info
        else:
            return super().step(observation, prev_action, prev_reward)