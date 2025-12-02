import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib.pyplot as plt
from IPython import display

# 设置参数
plt.ion()

# 游戏参数
BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT
SPEED = 100  # 游戏速度，值越小越快

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 方向常量
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 初始化蛇的位置（在中间）
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.score = 0
        self.food = self._place_food()
        self.game_over = False
        return self.get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food
    
    def get_state(self):
        # 状态表示：蛇头周围8个方向的距离、食物相对位置、蛇头方向
        head_x, head_y = self.snake[0]
        
        # 计算蛇头到障碍物的距离（墙或自身）
        point_l = (head_x - 1, head_y)
        point_r = (head_x + 1, head_y)
        point_u = (head_x, head_y - 1)
        point_d = (head_x, head_y + 1)
        
        # 方向
        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN
        
        # 危险情况
        danger_straight = (
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d))
        )
        
        danger_right = (
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d))
        )
        
        danger_left = (
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d))
        )
        
        # 食物位置
        food_left = self.food[0] < head_x
        food_right = self.food[0] > head_x
        food_up = self.food[1] < head_y
        food_down = self.food[1] > head_y
        
        state = [
            # 危险情况
            danger_straight, danger_right, danger_left,
            # 移动方向
            dir_l, dir_r, dir_u, dir_d,
            # 食物位置
            food_left, food_right, food_up, food_down
        ]
        
        return np.array(state, dtype=int)
    
    def _is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        # 检查是否撞墙
        if point[0] < 0 or point[0] >= GRID_WIDTH or point[1] < 0 or point[1] >= GRID_HEIGHT:
            return True
        # 检查是否撞到自己
        if point in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
        # 根据动作更新方向
        if action == 0:  # 直行
            new_direction = self.direction
        elif action == 1:  # 右转
            new_direction = (self.direction + 1) % 4
        else:  # 左转
            new_direction = (self.direction - 1) % 4
        
        self.direction = new_direction
        
        # 根据方向移动蛇头
        head_x, head_y = self.snake[0]
        if self.direction == RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.direction == LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == DOWN:
            new_head = (head_x, head_y + 1)
        
        # 检查游戏是否结束
        if self._is_collision(new_head):
            self.game_over = True
            reward = -50
            return self.get_state(), reward, self.game_over
        
        # 移动蛇
        self.snake.insert(0, new_head)
        
        # 检查是否吃到食物
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            # 如果没有吃到食物，移除蛇尾
            self.snake.pop()
            reward = 0
        
        return self.get_state(), reward, self.game_over

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=10000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([experience[0] for experience in minibatch])
        actions = torch.LongTensor([experience[1] for experience in minibatch])
        rewards = torch.FloatTensor([experience[2] for experience in minibatch])
        next_states = torch.FloatTensor([experience[3] for experience in minibatch])
        dones = torch.BoolTensor([experience[4] for experience in minibatch])
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn():
    # 初始化游戏和智能体
    game = SnakeGame()
    state_size = len(game.get_state())
    action_size = 3  # 直行、右转、左转
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    batch_size = 32
    episodes = 200
    scores = []
    avg_scores = []
    
    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while not game.game_over:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done = game.step(action)
            
            # 记住经验
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 经验回放
            agent.replay(batch_size)
        
        scores.append(game.score)
        avg_score = np.mean(scores[-10:])  # 最近100局的平均分
        avg_scores.append(avg_score)
        
        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}, Score: {game.score}, Epsilon: {agent.epsilon:.2f}, Avg Score: {avg_score:.2f}")
    
    # 绘制训练结果
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.6, label='Score per episode')
    plt.plot(avg_scores, label='Average score (last 10)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title('DQN Training Progress')
    plt.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('dqn_training_results.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return agent

def play_game(agent, episodes=5):
    # 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake with DQN')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 20)
    
    for e in range(episodes):
        game = SnakeGame()
        state = game.reset()
        
        while not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # 使用训练好的模型选择动作
            action = agent.act(state)
            state, reward, done = game.step(action)
            
            # 绘制游戏
            screen.fill(BLACK)
            
            # 绘制蛇
            for i, (x, y) in enumerate(game.snake):
                color = GREEN if i == 0 else BLUE  # 蛇头绿色，蛇身蓝色
                pygame.draw.rect(screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            
            # 绘制食物
            pygame.draw.rect(screen, RED, (game.food[0] * BLOCK_SIZE, game.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            
            # 显示分数
            score_text = font.render(f'Score: {game.score}', True, WHITE)
            screen.blit(score_text, (5, 5))
            
            pygame.display.flip()
            clock.tick(SPEED)
        
        print(f"Game {e+1}: Score = {game.score}")
        
        # 等待一下再开始下一局
        pygame.time.wait(1000)
    
    pygame.quit()

# 主程序
if __name__ == "__main__":
    # 训练模型
    print("开始训练DQN模型...")
    trained_agent = train_dqn()
    
    # 使用训练好的模型玩游戏
    print("训练完成，开始演示游戏...")
    play_game(trained_agent)