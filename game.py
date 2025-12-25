import numpy as np
from environnement import Env
from agent import Agent
import matplotlib.pyplot as plt
import argparse
import sys


parser = argparse.ArgumentParser(description="SnAIke Game")

def set_args() :
    parser.add_argument("-r", "--render", action="store_true", help="Show the game window.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs for training (default: 100).")
    parser.add_argument("-s", "--save", type=str, default="session", help="Save the training session.")
    parser.add_argument("-l", "--load", type=str, help="Load a training session.")
    parser.add_argument("-g", "--graph", action="store_true", help="Show training graphs.")
    parser.add_argument("-sg", "--save_graph", type=str, help="Save training graphs.")
    parser.add_argument("-ns", "--no_state", action="store_true", help="Do not show the detailed information.")
    parser.add_argument("-nl", "--no_learn", action="store_true", help="Disable learning.")
    #TODO: Change speed of the game
    #TODO: Step-by-step mode


def main():
    set_args()
    args = parser.parse_args()
    if args.save_graph and not args.graph:
        parser.error("You must use the -g option to save the graph.")
        sys.exit(1)
    env = Env(args.render)
    state_size = 12
    eps = 1.0
    if args.load:
        eps = 0.01
    agent = Agent(state_size, 0.01, 0.9, eps)
    batch_size = 32
    epochs = args.epochs
    steps = []
    sizes = []
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    if args.graph:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    if args.load:
        agent.load(args.load)
        print(f"Load trained model from {args.load}")

    try:
        for epoch in range(epochs):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            reward_per_epoch = 0

            print(f"Epoch: {epoch+1}")
            while not done:
                if not args.no_state:
                    env.print_state()
                action = agent.choose_action(state)
                if not args.no_state:
                    print(actions[action])
                next_state, reward, done = env.step(action)
                reward_per_epoch += reward
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if len(agent.memory) > batch_size and not args.no_learn:
                    agent.replay(batch_size)
                if done:
                    if args.graph:
                        steps.append(env.steps)
                        sizes.append(len(env.snake))
                        ax1.plot(steps, color='blue')
                        ax1.set_xlabel('Episode')
                        ax1.set_ylabel('Steps')
                        ax2.plot(sizes, color='red')
                        ax2.set_xlabel('Episode')
                        ax2.set_ylabel('Snake Size')
                        plt.pause(0.01)
                        if args.save_graph:
                            name = args.save_graph + ".png"
                            plt.savefig(name)
                    print("episode: {}/{}, score: {}, size: {},"
                        " e: {:.2}".format(epoch+1, epochs, reward_per_epoch, len(env.snake), agent.epsilon))
        if args.save:
            filename = args.save + ".weights.h5"
            agent.save(filename)
            print(f"Save learning state in {filename}")
    except KeyboardInterrupt:
        if args.save:
            filename = args.save + ".weights.h5"
            agent.save(filename)
            print(f"\nSave learning state in {filename}")
        print("\nTraining interrupted by user.")


if __name__ == "__main__":
    main()
