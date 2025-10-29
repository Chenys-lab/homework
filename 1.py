import numpy as np
import matplotlib.pyplot as plt

class MazeMDP:
    def __init__(self, maze_layout, rewards, actions, start, goal):
        """
        初始化迷宫MDP
        
        参数:
        maze_layout: 2D数组，0表示可通行，1表示障碍物
        rewards: 每个时间步的奖励
        actions: 可执行的动作
        start: 起始位置
        goal: 目标位置
        """
        self.maze = maze_layout
        self.rows, self.cols = maze_layout.shape
        self.reward_per_step = rewards
        self.actions = actions  # ['N', 'E', 'S', 'W']
        self.start = start
        self.goal = goal
        self.actions_dict = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}
        
        # 状态空间：所有可通行的位置
        self.states = []
        for i in range(self.rows):
            for j in range(self.cols):
                if maze_layout[i, j] == 0:
                    self.states.append((i, j))
        
        self.n_states = len(self.states)
        
    def get_next_state(self, state, action):
        """获取执行动作后的下一个状态"""
        i, j = state
        di, dj = self.actions_dict[action]
        next_i, next_j = i + di, j + dj
        
        # 检查边界和障碍物
        if (0 <= next_i < self.rows and 0 <= next_j < self.cols and 
            self.maze[next_i, next_j] == 0):
            return (next_i, next_j)
        else:
            return state  # 撞墙或出界，保持原地
    
    def is_terminal(self, state):
        """检查是否到达终止状态"""
        return state == self.goal

class MazeSolver:
    def __init__(self, maze_mdp, gamma=0.9, theta=1e-6):
        self.mdp = maze_mdp
        self.gamma = gamma  # 折扣因子
        self.theta = theta  # 收敛阈值
        
        # 初始化价值函数和策略
        self.V = {state: 0 for state in self.mdp.states}
        self.policy = {state: np.random.choice(self.mdp.actions) 
                      for state in self.mdp.states if not self.mdp.is_terminal(state)}
        
    def value_iteration(self):
        """价值迭代算法"""
        print("开始价值迭代...")
        iteration = 0
        while True:
            delta = 0
            new_V = self.V.copy()
            
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue
                    
                # 计算每个动作的价值
                action_values = []
                for action in self.mdp.actions:
                    next_state = self.mdp.get_next_state(state, action)
                    reward = self.mdp.reward_per_step
                    if self.mdp.is_terminal(next_state):
                        reward = 0  # 到达目标，无额外奖励
                    value = reward + self.gamma * self.V[next_state]
                    action_values.append(value)
                
                # 更新价值函数（取最大值）
                new_V[state] = max(action_values)
                delta = max(delta, abs(new_V[state] - self.V[state]))
            
            self.V = new_V
            iteration += 1
            
            if delta < self.theta:
                break
        
        # 从价值函数导出最优策略
        self.extract_policy()
        print(f"价值迭代完成，共{iteration}次迭代")
        return self.V, self.policy
    
    def policy_evaluation(self):
        """策略评估"""
        while True:
            delta = 0
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue
                
                action = self.policy[state]
                next_state = self.mdp.get_next_state(state, action)
                reward = self.mdp.reward_per_step
                if self.mdp.is_terminal(next_state):
                    reward = 0
                
                old_value = self.V[state]
                self.V[state] = reward + self.gamma * self.V[next_state]
                delta = max(delta, abs(old_value - self.V[state]))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        """策略改进"""
        policy_stable = True
        
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue
            
            old_action = self.policy[state]
            
            # 找出最优动作
            action_values = []
            for action in self.mdp.actions:
                next_state = self.mdp.get_next_state(state, action)
                reward = self.mdp.reward_per_step
                if self.mdp.is_terminal(next_state):
                    reward = 0
                value = reward + self.gamma * self.V[next_state]
                action_values.append((value, action))
            
            # 选择价值最大的动作
            best_value, best_action = max(action_values, key=lambda x: x[0])
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self):
        """策略迭代算法"""
        print("开始策略迭代...")
        iteration = 0
        
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            iteration += 1
            
            if policy_stable:
                break
        
        print(f"策略迭代完成，共{iteration}次迭代")
        return self.V, self.policy
    
    def extract_policy(self):
        """从价值函数中提取策略"""
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue
            
            action_values = []
            for action in self.mdp.actions:
                next_state = self.mdp.get_next_state(state, action)
                reward = self.mdp.reward_per_step
                if self.mdp.is_terminal(next_state):
                    reward = 0
                value = reward + self.gamma * self.V[next_state]
                action_values.append((value, action))
            
            best_value, best_action = max(action_values, key=lambda x: x[0])
            self.policy[state] = best_action
    
    def get_optimal_path(self, start):
        """获取最优路径"""
        path = [start]
        current = start
        
        max_steps = 100  # 防止无限循环
        steps = 0
        
        while not self.mdp.is_terminal(current) and steps < max_steps:
            action = self.policy[current]
            current = self.mdp.get_next_state(current, action)
            path.append(current)
            steps += 1
        
        return path
    
    def visualize_results(self, method_name):
        """可视化结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制价值函数热力图
        value_grid = np.full(self.mdp.maze.shape, np.nan)
        for (i, j), value in self.V.items():
            value_grid[i, j] = value
        
        im1 = ax1.imshow(value_grid, cmap='viridis')
        ax1.set_title(f'{method_name} - Value Function')
        plt.colorbar(im1, ax=ax1)
        
        # 绘制策略
        policy_grid = np.full(self.mdp.maze.shape, '', dtype=object)
        arrow_dict = {'N': '↑', 'E': '→', 'S': '↓', 'W': '←'}
        
        for (i, j), action in self.policy.items():
            policy_grid[i, j] = arrow_dict[action]
        
        # 绘制迷宫和策略
        ax2.imshow(self.mdp.maze, cmap='binary')
        ax2.set_title(f'{method_name} - Optimal Policy')
        
        for i in range(self.mdp.rows):
            for j in range(self.mdp.cols):
                if self.mdp.maze[i, j] == 0:  # 可通行格子
                    ax2.text(j, i, policy_grid[i, j], ha='center', va='center', 
                            fontsize=12, fontweight='bold')
        
        # 标记起点和终点
        start_i, start_j = self.mdp.start
        goal_i, goal_j = self.mdp.goal
        ax2.text(start_j, start_i, 'S', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='red')
        ax2.text(goal_j, goal_i, 'G', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.show()

# 示例使用
def main():
    # 定义迷宫布局 (0=可通行, 1=障碍物)
    maze_layout = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ])
    
    rewards = -1  # 每步奖励
    actions = ['N', 'E', 'S', 'W']
    start = (2, 0)  # 起点
    goal = (6, 7)   # 终点
    
    # 创建MDP环境
    maze_mdp = MazeMDP(maze_layout, rewards, actions, start, goal)
    
    print("=== 价值迭代 ===")
    vi_solver = MazeSolver(maze_mdp)
    vi_values, vi_policy = vi_solver.value_iteration()
    vi_path = vi_solver.get_optimal_path(start)
    print(f"最优路径: {vi_path}")
    vi_solver.visualize_results("Value Iteration")
    
    print("\n=== 策略迭代 ===")
    pi_solver = MazeSolver(maze_mdp)
    pi_values, pi_policy = pi_solver.policy_iteration()
    pi_path = pi_solver.get_optimal_path(start)
    print(f"最优路径: {pi_path}")
    pi_solver.visualize_results("Policy Iteration")
    
    # 比较结果
    print(f"\n=== 结果比较 ===")
    print(f"价值迭代路径长度: {len(vi_path)}")
    print(f"策略迭代路径长度: {len(pi_path)}")
    
    # 检查策略一致性
    policy_match = all(vi_policy[state] == pi_policy[state] 
                      for state in vi_policy.keys())
    print(f"策略一致性: {policy_match}")

if __name__ == "__main__":
    main()