import matplotlib.pyplot as plt
import pandas as pd


def plot_df(df, time_period, y_pred, y_true):
    SMALL_SIZE = 4
    BIG_SIZE = 6

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIG_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=250)

    concepts = df[['concept_id', 'concept_name']].drop_duplicates()

    i = 0
    for concept_id, concept_name in zip(concepts['concept_id'], concepts['concept_name']):
        concept_df = df[df['concept_id'] == concept_id]
        times = pd.concat([concept_df['time'], pd.Series([time_period])])
        amounts_true = pd.concat([concept_df['amount'], pd.Series([y_true[concept_id]])]) / 1e6
        amounts_pred = pd.concat([concept_df['amount'], pd.Series([y_pred[concept_id]])]) / 1e6

        ax = axs[i // ncols, i % ncols]
        ax.plot(times, amounts_pred, label=f'Pred', linewidth=0.5)
        ax.plot(times, amounts_true, label='True', linewidth=0.5, alpha=0.6)

        ax.set_title(concept_name)
        ax.set_xlabel('Time')
        ax.set_ylabel('Investment (M$)')

        ax.tick_params(axis='x', labelrotation=90)

        ax.legend()

        i += 1

        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #
        # # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(rotation=90)
    plt.subplots_adjust(left=None, right=None, top=None, bottom=None, wspace=0.6, hspace=0.8)
    plt.show()
