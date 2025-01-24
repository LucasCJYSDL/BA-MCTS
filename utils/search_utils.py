import numpy as np

def select_action(visit_counts, temperature, deterministic):
    """
    Overview:
        Select action from visit counts of the root node.
    Arguments:
        - visit_counts (:obj:`np.ndarray`): The visit counts of the root node.
        - temperature (:obj:`float`): The temperature used to adjust the sampling distribution.
        - deterministic (:obj:`bool`):  Whether to enable deterministic mode in action selection. True means to \
            select the argmax result, False indicates to sample action from the distribution.
    Returns:
        - action_pos (:obj:`np.int64`): The selected action position (index).
    """
    if deterministic:
        action_pos = np.argmax(visit_counts)
        return action_pos
    
    exp = 1.0 / temperature
    visit_counts = np.power(np.array(visit_counts), exp)
    action_probs = visit_counts / visit_counts.sum()
    action_pos = np.random.choice(len(visit_counts), p=action_probs)
    return action_pos

def visit_count_temperature(trained_steps, threshold_training_steps_for_final_lr_temperature):
    if trained_steps < 0.5 * threshold_training_steps_for_final_lr_temperature:
            return 1.0
    elif trained_steps < 0.75 * threshold_training_steps_for_final_lr_temperature:
        return 0.5
    else:
        return 0.25

class LinearParameter:
    def __init__(self, start=1.0, end=0.1, num_steps=10):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_decrement = (start - end) / float(num_steps)
        self.value = start
        self.current_step = 0

    def decrease(self):
        """Decreases the parameter linearly for one step and ensures it doesn't go below the minimum."""
        if self.current_step < self.num_steps:
            self.value -= self.step_decrement
            self.value = max(self.value, self.end)
            self.current_step += 1