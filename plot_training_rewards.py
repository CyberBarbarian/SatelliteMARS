import matplotlib.pyplot as plt
import pandas as pd

# Global variables for customization
SMOOTHING_WEIGHT = 0.98  # Adjust this value to change smoothing level
LINE_COLOR = 'firebrick'  # Adjust this value to change the line color


def read_data(csv_file):
    return pd.read_csv(csv_file)


def smooth_data(data, weight=SMOOTHING_WEIGHT):  # Exponential moving average
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_total_reward(df, lab_name):
    epochs = df['Epoch']
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['Total Reward'], color='darkgrey', label='Total Reward (Raw)')
    plt.plot(epochs, smooth_data(df['Total Reward']), color=LINE_COLOR, label='Total Reward (Smoothed)')
    plt.title('Total Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(f'pic/{lab_name}_total_reward_plot.png')  # Save the figure
    plt.show()


def plot_agent_rewards(df, lab_name):
    epochs = df['Epoch']
    agents = [col for col in df.columns if 'Agent' in col]
    for agent in agents:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, df[agent], color='darkgrey', label=f'{agent} (Raw)')
        plt.plot(epochs, smooth_data(df[agent]), color=LINE_COLOR, label=f'{agent} (Smoothed)')
        plt.title(f'{agent}')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(f'pic/{lab_name}_{agent}_reward_plot.png')  # Save the figure
        plt.show()


def main():
    lab_name = "lab4"
    data_name = f'data/reward/{lab_name}_rewards.csv'
    data = read_data(data_name)
    plot_total_reward(data, lab_name)
    plot_agent_rewards(data, lab_name)


if __name__ == '__main__':
    main()
