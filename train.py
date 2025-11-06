# train.py
import argparse
import os
import numpy as np
from environments.arcade_game import SimpleArcadeGame
from agents.dqn_agent import DQNAgent

def train_dqn(episodes=500, max_steps=300, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = SimpleArcadeGame(max_steps=max_steps)
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=len(env.get_valid_actions()),
        lr=1e-3, gamma=0.99,
        epsilon=1.0, eps_decay=0.995, eps_min=0.01,
        buffer_size=100_000, batch_size=64,
        target_update=1000
    )

    rewards = []
    best_avg = -1e9
    for ep in range(1, episodes + 1):
        s = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = agent.act(s, env.get_valid_actions())
            ns, r, done, _ = env.step(a)
            agent.remember(s, a, r, ns, float(done))
            agent.train_step()
            s = ns
            total += r
            if done:
                break
        rewards.append(total)
        if ep % 25 == 0:
            avg = float(np.mean(rewards[-25:]))
            print(f"Episode {ep}/{episodes} | avg_reward(25): {avg:.3f} | epsilon: {agent.epsilon:.3f}")
            if avg > best_avg:
                best_avg = avg
                agent.save(os.path.join(checkpoint_dir, "dqn_arcade_best.pth"))
        if ep % 100 == 0:
            agent.save(os.path.join(checkpoint_dir, f"dqn_arcade_ep{ep}.pth"))

    agent.save(os.path.join(checkpoint_dir, "dqn_arcade_final.pth"))
    print("Training complete. Checkpoints saved in", checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()
    train_dqn(episodes=args.episodes, max_steps=args.max_steps)
