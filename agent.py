
import torch
import random
import numpy as np
import os

from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

CHECKPOINT_PATH = './model/checkpoint.pth'
BEST_MODEL_PATH = './model/best_model.pth'


class Agent:

    def __init__(self):

        self.n_games = 0
        self.epsilon = 80
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        # Neural Network
        self.model = Linear_QNet(11, 256, 3)

        # Q-Learning Trainer
        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

    # --------------------------------
    # GET GAME STATE
    # --------------------------------

    def get_state(self, game):

        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(state, dtype=int)

    # --------------------------------
    # STORE EXPERIENCE
    # --------------------------------

    def remember(self, state, action, reward, next_state, done):

        self.memory.append(
            (state, action, reward, next_state, done)
        )

    # --------------------------------
    # LONG MEMORY TRAINING
    # --------------------------------

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:

            mini_sample = random.sample(
                self.memory,
                BATCH_SIZE
            )

        else:

            mini_sample = list(self.memory)

        if not mini_sample:
            return

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    # --------------------------------
    # SHORT MEMORY TRAINING
    # --------------------------------

    def train_short_memory(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done
        )

    # --------------------------------
    # GET ACTION
    # --------------------------------

    def get_action(self, state):

        # Exploration vs exploitation
        self.epsilon = max(0, 80 - self.n_games)

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)

            final_move[move] = 1

        else:

            state0 = torch.tensor(
                state,
                dtype=torch.float
            )

            prediction = self.model(state0)

            move = torch.argmax(prediction).item()

            final_move[move] = 1

        return final_move

    # --------------------------------
    # SAVE TRAINING CHECKPOINT
    # --------------------------------

    def save_checkpoint(
        self,
        best_score,
        total_score,
        plot_score,
        plot_mean_score
    ):

        os.makedirs('./model', exist_ok=True)

        checkpoint = {

            # Neural network
            'model_state_dict': self.model.state_dict(),

            # Optimizer
            'optimizer_state_dict':
                self.trainer.optimizer.state_dict(),

            # Training progress
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'best_score': best_score,
            'total_score': total_score,

            # Experience replay memory
            'memory': list(self.memory),

            # Graph history
            'plot_score': plot_score,
            'plot_mean_score': plot_mean_score,

            # Random states
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

        print(
            f"[CHECKPOINT] Saved after game {self.n_games}"
        )

    # --------------------------------
    # LOAD TRAINING CHECKPOINT
    # --------------------------------

    def load_checkpoint(self):

        if not os.path.exists(CHECKPOINT_PATH):

            print("[CHECKPOINT] No checkpoint found.")
            print("[CHECKPOINT] Starting fresh training.")

            return None

        try:

            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location='cpu',
                weights_only=False
            )

            # Restore neural network
            self.model.load_state_dict(
                checkpoint['model_state_dict']
            )

            # Restore optimizer
            self.trainer.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

            # Restore training progress
            self.n_games = checkpoint['n_games']
            self.epsilon = checkpoint['epsilon']

            # Restore replay memory
            self.memory = deque(
                checkpoint['memory'],
                maxlen=MAX_MEMORY
            )

            # Restore random states
            random.setstate(
                checkpoint['python_rng_state']
            )

            np.random.set_state(
                checkpoint['numpy_rng_state']
            )

            torch.set_rng_state(
                checkpoint['torch_rng_state']
            )

            print("\n========== TRAINING RESUMED ==========")
            print("Games completed:", self.n_games)
            print("Best score:", checkpoint['best_score'])
            print("Replay memory:", len(self.memory))
            print("Epsilon:", self.epsilon)
            print("======================================\n")

            return checkpoint

        except Exception as e:

            print("[CHECKPOINT ERROR]", e)
            print("Could not load checkpoint.")

            raise


# ====================================
# MAIN TRAINING FUNCTION
# ====================================

def train():

    agent = Agent()

    game = SnakeGameAI()

    # Load previous training session
    checkpoint = agent.load_checkpoint()

    if checkpoint is not None:

        plot_score = checkpoint['plot_score']
        plot_mean_score = checkpoint['plot_mean_score']

        total_score = checkpoint['total_score']
        best_score = checkpoint['best_score']

    else:

        plot_score = []
        plot_mean_score = []

        total_score = 0
        best_score = 0

    print("\nStarting AI Snake Training...\n")

    while True:

        # Get old state
        state_old = agent.get_state(game)

        # Get action
        final_move = agent.get_action(state_old)

        # Perform move
        reward, done, score = game.play_step(final_move)

        # Get new state
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        # Store experience
        agent.remember(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        if done:

            # Reset game
            game.reset()

            agent.n_games += 1

            # Train long memory
            agent.train_long_memory()

            # Update best score
            if score > best_score:

                best_score = score

                # Save best-performing model
                os.makedirs('./model', exist_ok=True)

                torch.save(
                    agent.model.state_dict(),
                    BEST_MODEL_PATH
                )

                print("[BEST MODEL] New record saved!")

            # Print progress
            print(
                'Game:', agent.n_games,
                '| Score:', score,
                '| Record:', best_score,
                '| Epsilon:', agent.epsilon
            )

            # Update graph data
            plot_score.append(score)

            total_score += score

            mean_score = total_score / agent.n_games

            plot_mean_score.append(mean_score)

            # Update graph
            plot(
                plot_score,
                plot_mean_score
            )

            # Save complete training checkpoint
            agent.save_checkpoint(
                best_score,
                total_score,
                plot_score,
                plot_mean_score
            )


if __name__ == '__main__':
    train()
