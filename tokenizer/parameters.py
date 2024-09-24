from pydantic import BaseModel


class InputParameters(BaseModel):
    num_agents: int = 13
    num_previous_actions: int = 5
    agents_radius: int = 5
    cost2go_value_limit: int = 20
    cost2go_radius: int = 5
    context_size: int = 256
    mask_greed_action: bool = False
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
