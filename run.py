import json
import time
import argparse
from unsloth import FastLanguageModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from rotator import BanditArmsRotator
from bandits import stationary_MAB, drifting_MAB, stepwise_MAB, two_context_MAB
from bandits import three_context_MAB, moving_avg_MAB, time_delayed_MAB
from mab import MAB
from analysis.plot import plot_results


def run_experiment(pipe, config):
    results_df = pd.DataFrame(
        columns=["bandit", "og_arms", "og_hints", "arms", "hint", "history"]
    )
    bandit = eval(config["bandit"] + "_MAB()")

    bandit.set_narms(config["narms"])
    bandit.set_stds(config["arm_stds"])

    for a in config["arm_means"]:
        rotator = BanditArmsRotator(a, config["hints"])

        while rotator.current_index < bandit.narms:
            bandit.means, rhints = rotator.next()
            for ogh, h in zip(config["hints"], rhints):
                for t in range(config["ntrials"]):
                    hist = MAB(pipe, h, bandit)
                    results_df = results_df._append(
                        {
                            "bandit": config["bandit"],
                            "og_arms": a,
                            "og_hints": ogh,
                            "arms": bandit.means,
                            "hint": h,
                            "history": hist,
                        },
                        ignore_index=True,
                    )

    return results_df


def load_model(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=4096,
        # max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
        force_download=True,
    )
    FastLanguageModel.for_inference(model)

    # model, tokenizer = AutoModelForCausalLM.from_pretrained(config["model_name"]), AutoTokenizer.from_pretrained(config["model_name"])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=config["temperature"],
        max_new_tokens=1,
    )
    return model, tokenizer, pipe


def run_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/stationary.json")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    _, _, pipe = load_model(config)
    results_df = run_experiment(pipe, config)

    results_df.to_parquet(
        config["bandit"] + f"_{time.strftime('%Y%m%d-%H%M%S')}.parquet",
        engine="pyarrow",
        index=False,
    )
    print(results_df)
    # plot_results(args.config_file, config["bandit"] + f'_{time.strftime("%Y%m%d-%H%M%S")}.parquet')


if __name__ == "__main__":
    run_exp()
