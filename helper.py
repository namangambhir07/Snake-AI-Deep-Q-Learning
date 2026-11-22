
import matplotlib.pyplot as plt

plt.ion()

# Create one persistent graph window
fig, ax = plt.subplots(figsize=(6, 4))

fig.canvas.manager.set_window_title(
    "Snake AI - Live Training Progress"
)

ax.set_title("Snake AI Training Performance")
ax.set_xlabel("Number of Games")
ax.set_ylabel("Food Collected")

ax.grid(True, alpha=0.3)

score_line, = ax.plot(
    [], [], 'o',
    alpha=0.4,
    label="Individual Score"
)

mean_line, = ax.plot(
    [], [],
    linewidth=2,
    label="Overall Mean Score"
)

rolling_line, = ax.plot(
    [], [],
    linewidth=2,
    label="10-Game Average"
)

ax.legend()

plt.show(block=False)


def plot(scores, mean_scores):

    games = list(range(1, len(scores) + 1))

    # Individual scores
    score_line.set_data(games, scores)

    # Overall average
    mean_line.set_data(games, mean_scores)

    # Rolling average
    window = min(10, len(scores))

    rolling = [
        sum(scores[max(0, i-window+1):i+1]) /
        len(scores[max(0, i-window+1):i+1])
        for i in range(len(scores))
    ]

    rolling_line.set_data(games, rolling)

    ax.relim()
    ax.autoscale_view()

    ax.set_ylim(bottom=0)

    ax.set_title(
        f"Snake AI Training | Games: {len(scores)} | "
        f"Best: {max(scores)} | "
        f"Latest: {scores[-1]}"
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    plt.pause(0.05)
