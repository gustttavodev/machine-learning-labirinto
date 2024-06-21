import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

# Definir o ambiente do labirinto
labirinto = np.array([
    [0, -1, 0, 0, 1],
    [0, -1, 0, -1, 0],
    [0, 0, 0, -1, 0],
    [-1, -1, 0, 0, 0]
])

# Definir o Q-Learning
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, n_actions, state_shape):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(state_shape + (n_actions,))
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.q_table.shape[2]))
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# Parâmetros e configuração
n_actions = 4  # Cima, baixo, esquerda, direita
state_shape = labirinto.shape
agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=n_actions, state_shape=state_shape)

# Função para verificar se o estado é terminal (saída do labirinto)
def is_terminal_state(state):
    return labirinto[state] == 1

# Função para obter a recompensa para um estado
def get_reward(state):
    return labirinto[state]

# Função para obter o próximo estado com base na ação
def get_next_state(state, action):
    if action == 0:  # Cima
        return (max(state[0] - 1, 0), state[1])
    elif action == 1:  # Baixo
        return (min(state[0] + 1, labirinto.shape[0] - 1), state[1])
    elif action == 2:  # Esquerda
        return (state[0], max(state[1] - 1, 0))
    elif action == 3:  # Direita
        return (state[0], min(state[1] + 1, labirinto.shape[1] - 1))

# Configurações da animação
fig, ax = plt.subplots()
cmap = colors.ListedColormap(['white', 'black', 'red', 'green'])
bounds = [-1, 0, 0.5, 1, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Inicializar o gráfico do labirinto
labirinto_plot = ax.imshow(labirinto, cmap=cmap, norm=norm)

# Função de animação para atualizar o gráfico
def update(frame):
    state = (0, 0)  # Estado inicial
    steps = 0
    while not is_terminal_state(state):
        action = agent.choose_action(state)
        next_state = get_next_state(state, action)
        if labirinto[next_state] == -1:
            next_state = state
        reward = get_reward(next_state)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        steps += 1

        labirinto_copy = labirinto.copy()
        labirinto_copy[state] = 2  # Estado atual do agente
        labirinto_plot.set_array(labirinto_copy)
        ax.set_title(f'Tentativa: {frame}, Passos: {steps}')
        return [labirinto_plot]

# Configurar animação
ani = FuncAnimation(fig, update, frames=np.arange(1, 101), interval=100, blit=True)

# Mostrar a animação
plt.show()
