import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.table import Table

class CliffWalkEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start_pos = (rows-1, 0)  # 左下角
        self.goal_pos = (rows-1, cols-1)  # 右下角
        self.cliff_positions = [(rows-1, j) for j in range(1, cols-1)]  # 底部中间的悬崖
        
        # 动作空间: 0:上, 1:右, 2:下, 3:左
        self.actions = [0, 1, 2, 3]
        self.action_effects = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        self.reset()
    
    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        row, col = self.current_pos
        drow, dcol = self.action_effects[action]
        
        # 计算新位置
        new_row = max(0, min(self.rows-1, row + drow))
        new_col = max(0, min(self.cols-1, col + dcol))
        new_pos = (new_row, new_col)
        
        # 检查是否掉下悬崖
        if new_pos in self.cliff_positions:
            reward = -100
            done = False
            new_pos = self.start_pos  # 回到起点
        elif new_pos == self.goal_pos:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        
        self.current_pos = new_pos
        return new_pos, reward, done

def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(len(Q[0]))
    else:
        return np.argmax(Q[state])

def sarsa(env, episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1):
    Q = np.zeros((env.rows * env.cols, len(env.actions)))
    
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        state_idx = state[0] * env.cols + state[1]
        
        # 选择初始动作
        action = epsilon_greedy_policy(Q, state_idx, epsilon)
        
        total_reward = 0
        done = False
        
        while not done:
            # 执行动作
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * env.cols + next_state[1]
            
            # 选择下一个动作
            next_action = epsilon_greedy_policy(Q, next_state_idx, epsilon)
            
            # SARSA更新
            if done:
                target = reward
            else:
                target = reward + gamma * Q[next_state_idx][next_action]
            
            Q[state_idx][action] += alpha * (target - Q[state_idx][action])
            
            state_idx = next_state_idx
            action = next_action
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode

def q_learning(env, episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1):
    Q = np.zeros((env.rows * env.cols, len(env.actions)))
    
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        state_idx = state[0] * env.cols + state[1]
        
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = epsilon_greedy_policy(Q, state_idx, epsilon)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * env.cols + next_state[1]
            
            # Q-learning更新
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state_idx])
            
            Q[state_idx][action] += alpha * (target - Q[state_idx][action])
            
            state_idx = next_state_idx
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode

def plot_results(sarsa_rewards, qlearning_rewards, window=10):
    plt.figure(figsize=(12, 8))
    
    # 平滑曲线
    def smooth_rewards(rewards, window):
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))
        return smoothed
    
    sarsa_smooth = smooth_rewards(sarsa_rewards, window)
    qlearning_smooth = smooth_rewards(qlearning_rewards, window)
    
    plt.plot(sarsa_smooth, label='SARSA', alpha=0.8)
    plt.plot(qlearning_smooth, label='Q-learning', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA vs Q-learning on Cliff Walk')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_policy(env, Q, title):
    policy = np.argmax(Q, axis=1)
    action_symbols = ['↑', '→', '↓', '←']
    
    grid = np.empty((env.rows, env.cols), dtype=object)
    
    for i in range(env.rows):
        for j in range(env.cols):
            state_idx = i * env.cols + j
            pos = (i, j)
            
            if pos == env.start_pos:
                grid[i, j] = 'S'
            elif pos == env.goal_pos:
                grid[i, j] = 'G'
            elif pos in env.cliff_positions:
                grid[i, j] = 'C'
            else:
                grid[i, j] = action_symbols[policy[state_idx]]
    
    plt.figure(figsize=(12, 3))
    plt.imshow(np.ones((env.rows, env.cols)), cmap='Pastel1')
    
    for i in range(env.rows):
        for j in range(env.cols):
            color = 'white'
            if (i, j) == env.start_pos:
                color = 'lightgreen'
            elif (i, j) == env.goal_pos:
                color = 'lightcoral'
            elif (i, j) in env.cliff_positions:
                color = 'lightgray'
            
            plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, 
                                            facecolor=color, edgecolor='black'))
            plt.text(j, i, grid[i, j], ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    
    plt.xlim(-0.5, env.cols-0.5)
    plt.ylim(env.rows-0.5, -0.5)
    plt.title(f'{title} Policy')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 创建Cliff Walk环境
    env = CliffWalkEnv()
    
    print("开始训练SARSA算法...")
    Q_sarsa, sarsa_rewards = sarsa(env, episodes=1000, alpha=0.1, epsilon=0.1)
    
    print("开始训练Q-learning算法...")
    Q_qlearning, qlearning_rewards = q_learning(env, episodes=1000, alpha=0.1, epsilon=0.1)
    
    # 绘制结果比较
    plot_results(sarsa_rewards, qlearning_rewards)
    
    # 可视化学习到的策略
    visualize_policy(env, Q_sarsa, "SARSA")
    visualize_policy(env, Q_qlearning, "Q-learning")
    
    # 打印性能统计
    print(f"\n性能统计 (最后100回合平均奖励):")
    print(f"SARSA: {np.mean(sarsa_rewards[-100:]):.2f}")
    print(f"Q-learning: {np.mean(qlearning_rewards[-100:]):.2f}")
    