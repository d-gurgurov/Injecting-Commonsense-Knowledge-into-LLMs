from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(logdir):
    event_acc = event_accumulator.EventAccumulator(logdir)
    event_acc.Reload()
    return event_acc

def plot_metrics(ax, event_acc, metric_names, title):
    for metric_name in metric_names:
        steps = [event.step for event in event_acc.scalars.Items(metric_name)]
        values = [event.value for event in event_acc.scalars.Items(metric_name)]

        ax.plot(steps, values, label=metric_name)

    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.legend()

def visualize_tensorboard_logs(logdir, save=False, save_path=None):
    event_acc = load_tensorboard_data(logdir)

    print(event_acc.scalars.Keys())

    # specifying the metrics to plot
    train_metrics = ['train/loss', 'eval/loss']
    eval_metrics = ['eval/f1']

    # creating subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # plotting evaluation accuracy on the left subplot
    plot_metrics(axs[0], event_acc, eval_metrics, 'Evaluation Accuracy')

    # plotting training loss and val loss on the right subplot
    plot_metrics(axs[1], event_acc, train_metrics, 'Training Metrics')
    
    graph = plt.gcf()

    plt.tight_layout()
    plt.show()

    if save==True:
        graph.savefig(save_path, pad_inches=0.1, bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    tensorboard_logdir = "evaluation/fus_sa_e1-4"
    print('cn_sa_e1-4_20')
    visualize_tensorboard_logs(tensorboard_logdir, save=True, save_path="./evaluation/fus_ad_sa_logs.png")