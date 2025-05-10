import torch
import random
import pandas as pd
from config.constants import MAIN_TEXT, QUESTION, SEED, N_ITERS

torch.cuda.manual_seed_all(SEED)


# Run one MAB experiment
def MAB(pipe, hint, bandit):
    instruction = MAIN_TEXT + "\n\nYour hint for this round is: " + hint + "\n"
    print(instruction)

    arm_counts = [0] * bandit.narms
    arm_rewards = [0] * bandit.narms
    total_rewards = 0
    hist = []
    history_instruction = ""

    for step in range(N_ITERS):
        input_text = instruction + history_instruction + QUESTION
        choice = int(pipe(input_text)[0]["generated_text"][len(input_text) :])

        if 1 <= choice <= bandit.narms:
            hist.append(choice)
            chosen_idx = choice - 1

        else:
            hist.append(None)
            chosen_idx = random.randint(0, bandit.narms - 1)

        reward = bandit.get_reward(chosen_idx)
        arm_counts[chosen_idx] += 1
        arm_rewards[chosen_idx] += reward
        total_rewards += reward

        history_instruction += f"Trial {step}: You chose arm <<{choice}>> and received a reward of {reward:.2f}.\n"

        print(
            f"Trial {step}: You chose arm <<{choice}>> and received a reward of {reward:.2f}."
        )

    print(f"Total Arm Counts :" + str(arm_counts))
    print(f"Actual Arm Means: {bandit.means}")

    return hist
