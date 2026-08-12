import matplotlib.pyplot as plt
import pandas as pd
from hxmap.config import CONTINUOUS_FEATURES


def plot_feature_points(mapping_df, out_dir):
    for feature, df in mapping_df.groupby('Feature'):
        plot_df = df[['Feature value', 'Nomogram Points']].sort_values('Feature value').drop_duplicates(subset=['Feature value'])
        fig, ax = plt.subplots(figsize=(5, 1.8), dpi=300)

        if feature in CONTINUOUS_FEATURES:
            ax.plot(plot_df['Feature value'], plot_df['Nomogram Points'], marker='o', linewidth=1.5)
        else:
            x = plot_df['Feature value'].astype(float).tolist()
            y = plot_df['Nomogram Points'].astype(float).tolist()
            ax.plot(x, y, marker='o', linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) if float(v).is_integer() else f'{v:g}' for v in x])
            ax.set_xlim(min(x) - 0.3, max(x) + 0.3)

        ax.set_xlabel(feature)
        ax.set_ylabel('Points')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / f'HXMAP_points_{feature.replace(" ", "_")}.png', dpi=600, bbox_inches='tight')
        plt.close()


def plot_points_probability_curve(points_probability_df, out_file):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.plot(points_probability_df['Predicted Probability'], points_probability_df['Total Points'], linewidth=2)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Total Points')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    plt.close()
