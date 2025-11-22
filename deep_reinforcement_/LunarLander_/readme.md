LunarLander-v2 — PPO Reinforcement Learning Agent

This project trains a PPO (Proximal Policy Optimization) agent to solve the LunarLander-v2 environment from Gymnasium.
The trained agent, code, evaluation, and video are all included in this repository.

📘 Project Overview

Algorithm: PPO (Stable-Baselines3)

Environment: LunarLander-v2

Training Steps: 1,000,000

Frameworks: Gymnasium, SB3, PyVirtualDisplay, HuggingFace Hub

Output: Trained model + evaluation video + notebook

📂 Files in This Repository

lunarlander_training.ipynb — complete training & upload pipeline

video/ — agent gameplay video

🤗 Trained Model on Hugging Face- https://huggingface.co/Gauravsinhasinha/ppo-LunarLander-v2/tree/main/ppo-LunarLander-v2

to load model:
from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("Gauravsinhasinha/ppo-LunarLander-v2")
env = gym.make("LunarLander-v2", render_mode="human")

obs, info = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
🧠 Key Training Hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1)


    

