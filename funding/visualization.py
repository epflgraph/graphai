import matplotlib.pyplot as plt


def plot_df(df):
    fig, ax = plt.subplots(dpi=150)

    concepts = df[['concept_id', 'concept_name']].drop_duplicates()

    for concept_id, concept_name in zip(concepts['concept_id'], concepts['concept_name']):
        concept_df = df[df['concept_id'] == concept_id]
        ax.plot(concept_df['time'], concept_df['amount'] / 1e6, label=concept_name)

    ax.set_title('Investment per concept')
    ax.set_xlabel('Time')
    ax.set_ylabel('Investment (M$)')

    ax.legend()

    plt.show()
