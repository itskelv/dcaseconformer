import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')

# Define the subdirectories to analyze separately
subdirs = [
    # "../DCASE2025_SELD_dataset/metadata_dev/dev-train-sony",
    # "../DCASE2025_SELD_dataset/metadata_dev/dev-train-tau",
    # "../DCASE2025_SELD_dataset/metadata_dev/dev-test-sony",
    # "../DCASE2025_SELD_dataset/metadata_dev/dev-test-tau",
    "../synth_data_test_1/metadata_dev/dev-train-synth"
    # "../DCASE2025_SELD_dataset/metadata_dev/dev-test-synth"
]


# Function to calculate event durations - kept exactly the same as original
def calculate_event_durations(df):
    event_data = []
    # Group by source and class
    for (src, cls), group in df.groupby(['source', 'class']):
        # Sort by frame
        group = group.sort_values('frame')

        # Identify events (sequences of consecutive frames)
        group['frame_diff'] = group['frame'].diff()
        # New event starts when frame diff > 1 or is NaN (first frame)
        group['new_event'] = (group['frame_diff'] > 1) | (group['frame_diff'].isna())
        group['event_id'] = group['new_event'].cumsum()

        # Calculate event durations
        event_durations = group.groupby('event_id').agg(
            duration=('frame', lambda x: x.max() - x.min() + 1),
            avg_distance=('distance', 'mean'),
            avg_azimuth=('azimuth', 'mean'),
            onscreen_ratio=('onscreen', 'mean')
        )

        for _, row in event_durations.iterrows():
            event_data.append({
                'class': cls,
                'source': src,
                'duration': row['duration'],
                'avg_distance': row['avg_distance'],
                'avg_azimuth': row['avg_azimuth'],
                'onscreen_ratio': row['onscreen_ratio']
            })

    return pd.DataFrame(event_data)


# Also create a consolidated analysis for all data
all_paths = glob.glob("../DCASE2025_SELD_dataset/metadata_dev/**/*.csv", recursive=True)
print(f"Found {len(all_paths)} total CSV files")

# Process each subdirectory separately
for subdir in subdirs + ["../DCASE2025_SELD_dataset/metadata_dev"]:  # Add the consolidated analysis
    # Use a descriptive name for the output directory
    if subdir == "../DCASE2025_SELD_dataset/metadata_dev":
        output_dir = "eda_plots_all_data"
        print(f"\n\n=== Processing ALL DATA ===\n")
        paths = all_paths
    else:
        output_dir = f"eda_plots_{os.path.basename(subdir)}"
        print(f"\n\n=== Processing {subdir} ===\n")
        # Load CSVs from just this subdirectory
        paths = glob.glob(f"{subdir}/**/*.csv", recursive=True)

    print(f"Found {len(paths)} CSV files in {subdir}")

    # Create directory for saving plots
    os.makedirs(output_dir, exist_ok=True)

    # Filter out any empty or corrupted files
    df_list = []
    for p in paths:
        if os.path.getsize(p) > 0:
            try:
                df = pd.read_csv(p)
                # If it doesn't have headers, set appropriate column names
                if len(df.columns) == 6 and df.columns[0] == '0':
                    df.columns = ['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']
                df_list.append(df)
            except Exception as e:
                print(f"Skipping file {p} due to error: {e}")

    print(f"Successfully loaded {len(df_list)} CSV files")

    # Proceed only if we have valid data
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)

        # Check columns and data types
        print("DataFrame columns:", df_all.columns.tolist())
        print("Data types:")
        print(df_all.dtypes)
        print("Sample data:")
        print(df_all.head())

        # Calculate class statistics
        class_counts = df_all['class'].value_counts().sort_index()
        total_samples = len(df_all)
        class_proportions = class_counts / total_samples

        # 1. Class Distribution with imbalance metrics
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
        plt.title(f'Class Distribution - {os.path.basename(subdir)}')
        plt.xlabel('Class Index')
        plt.ylabel('Count')

        # Calculate imbalance metrics
        gini = 1 - np.sum(np.square(class_proportions))
        class_entropy = entropy(class_proportions)

        plt.annotate(f'Gini Coefficient: {gini:.4f}\nEntropy: {class_entropy:.4f}',
                     xy=(0.7, 0.9), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distribution.png")
        plt.close()

        # 2. Event Duration Analysis per Class
        # First, identify events by grouping consecutive frames
        df_all = df_all.sort_values(['source', 'class', 'frame'])

        event_df = calculate_event_durations(df_all)

        # Plot event durations
        plt.figure(figsize=(14, 8))
        # Make sure classes are ordered numerically
        class_order = sorted(event_df['class'].unique())

        # Create boxplot with numerical order
        ax = sns.boxplot(data=event_df, x='class', y='duration', order=class_order, palette='coolwarm')
        plt.title(f'Event Duration per Class - {os.path.basename(subdir)}')
        plt.xlabel('Class Index')
        plt.ylabel('Duration (in frames)')

        # Add a table with statistics below the plot
        duration_stats = event_df.groupby('class')['duration'].describe().round(1)
        plt.figtext(0.5, -0.05, f"Duration statistics:\n{duration_stats.to_string()}",
                    ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        plt.tight_layout()
        plt.savefig(f"{output_dir}/event_duration_boxplot.png")
        plt.close()

        # 3. Distance Distribution per Class
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(data=df_all, x='class', y='distance', order=sorted(df_all['class'].unique()))
        plt.title(f'Distance Distribution per Class - {os.path.basename(subdir)}')
        plt.xlabel('Class Index')
        plt.ylabel('Distance (cm)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distance_distribution_boxplot.png")
        plt.close()

        # 4. Onscreen/Offscreen Ratio per Class
        # Check unique onscreen values
        unique_onscreen = df_all['onscreen'].unique()
        print(f"Unique onscreen values: {unique_onscreen}")

        on_off_stats = (
            df_all.groupby('class')['onscreen']
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # Renaming with handling for various possible column configurations
        column_mapping = {0: 'Offscreen', 1: 'Onscreen'}
        on_off_stats = on_off_stats.rename(columns=column_mapping)

        plt.figure(figsize=(12, 6))
        on_off_stats.plot(kind='bar', stacked=True, colormap='tab10')
        plt.title(f'Onscreen vs Offscreen Ratio per Class - {os.path.basename(subdir)}')
        plt.xlabel('Class Index')
        plt.ylabel('Proportion')
        plt.legend(title='Visibility')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/onscreen_offscreen_ratio.png")
        plt.close()

        # 5. Azimuth Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df_all['azimuth'], bins=50, kde=True, color='teal')
        plt.title(f"Distribution of Azimuth Angles - {os.path.basename(subdir)}")
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/azimuth_distribution.png")
        plt.close()

        # Distribution of Azimuth per Class
        plt.figure(figsize=(14, 10))
        for i, cls in enumerate(sorted(df_all['class'].unique())):
            plt.subplot(4, 4, i + 1)
            class_data = df_all[df_all['class'] == cls]
            sns.kdeplot(data=class_data['azimuth'], fill=True)
            plt.title(f"Class {cls}")
            plt.xlabel("Azimuth")
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(f"{output_dir}/azimuth_by_class.png")
        plt.close()

        # 6. Class Co-occurrence Matrix
        file_classes = {}
        for path in paths:
            if os.path.getsize(path) > 0:
                try:
                    df = pd.read_csv(path)
                    if len(df.columns) == 6 and df.columns[0] == '0':
                        df.columns = ['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']
                    unique_classes = df['class'].unique()
                    file_name = os.path.basename(path)
                    file_classes[file_name] = unique_classes
                except Exception as e:
                    print(f"Error processing {path} for co-occurrence matrix: {e}")

        # Determine the maximum class index
        all_classes = set()
        for classes in file_classes.values():
            all_classes.update(classes)

        max_class_index = int(max(all_classes)) + 1
        print(f"Maximum class index: {max_class_index - 1}")

        co_matrix = np.zeros((max_class_index, max_class_index))
        for cls_list in file_classes.values():
            for i in cls_list:
                i = int(i)
                for j in cls_list:
                    j = int(j)
                    co_matrix[i][j] += 1

        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(co_matrix, dtype=bool)
        mask[co_matrix == 0] = True  # Mask entries with zero co-occurrences
        sns.heatmap(co_matrix, annot=True, fmt=".0f", cmap="YlGnBu", mask=mask)
        plt.title(f"Class Co-occurrence Matrix - {os.path.basename(subdir)}")
        plt.xlabel("Class Index")
        plt.ylabel("Class Index")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_cooccurrence_matrix.png")
        plt.close()

        # 7. Create features for dimensionality reduction
        # Use numerical features we already have
        features_df = event_df[['class', 'duration', 'avg_distance', 'avg_azimuth', 'onscreen_ratio']]

        # Normalize the features
        from sklearn.preprocessing import StandardScaler

        X = features_df.drop('class', axis=1)
        y = features_df['class']
        X_scaled = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Class')
        plt.title(f'PCA of Event Features - {os.path.basename(subdir)}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_visualization.png")
        plt.close()

        # t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Class')
            plt.title(f't-SNE of Event Features - {os.path.basename(subdir)}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/tsne_visualization.png")
            plt.close()
        except Exception as e:
            print(f"t-SNE failed: {e}")

        # UMAP
        try:
            reducer = umap.UMAP(random_state=42)
            umap_result = reducer.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Class')
            plt.title(f'UMAP of Event Features - {os.path.basename(subdir)}')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/umap_visualization.png")
            plt.close()
        except Exception as e:
            print(f"UMAP failed: {e}")

        # Create a summary report
        with open(f"{output_dir}/eda_summary.txt", "w") as f:
            f.write(f"EDA Summary for {os.path.basename(subdir)} Dataset\n")
            f.write("===============================\n\n")

            f.write("1. Class Distribution\n")
            f.write("-----------------\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Number of classes: {len(class_counts)}\n")
            f.write(f"Class counts:\n{class_counts.to_string()}\n\n")
            f.write(f"Gini Coefficient (measure of inequality): {gini:.4f}\n")
            f.write(f"Entropy of class distribution: {class_entropy:.4f}\n\n")

            f.write("2. Event Duration Statistics\n")
            f.write("-----------------------\n")
            f.write(f"{event_df.groupby('class')['duration'].describe().to_string()}\n\n")

            f.write("3. Distance Statistics\n")
            f.write("------------------\n")
            f.write(f"{df_all.groupby('class')['distance'].describe().to_string()}\n\n")

            f.write("4. Azimuth Statistics\n")
            f.write("------------------\n")
            f.write(f"{df_all.groupby('class')['azimuth'].describe().to_string()}\n\n")

            f.write("5. Onscreen/Offscreen Statistics\n")
            f.write("----------------------------\n")
            f.write(f"{on_off_stats.to_string()}\n\n")

            f.write("6. Feature Correlation\n")
            f.write("------------------\n")
            correlation = event_df[['duration', 'avg_distance', 'avg_azimuth', 'onscreen_ratio']].corr()
            f.write(f"{correlation.to_string()}\n\n")

        print(f"EDA completed for {subdir}! Results saved to {output_dir}/ directory")

    else:
        print(f"No valid metadata CSV files found to analyze in {subdir}")

# Create overall summary comparing train and test datasets
# Focus on key statistics
print("\nGenerating comparative summary...")

try:
    # Get the class counts from each dataset
    datasets = {}
    for subdir in subdirs:
        output_dir = f"eda_plots_{os.path.basename(subdir)}"

        # Check if this directory exists
        if not os.path.exists(output_dir):
            continue

        # Load the CSV files for this dataset
        paths = glob.glob(f"{subdir}/**/*.csv", recursive=True)
        df_list = []
        for p in paths:
            if os.path.getsize(p) > 0:
                try:
                    df = pd.read_csv(p)
                    if len(df.columns) == 6 and df.columns[0] == '0':
                        df.columns = ['frame', 'class', 'source', 'azimuth', 'distance', 'onscreen']
                    df_list.append(df)
                except Exception:
                    pass

        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            datasets[os.path.basename(subdir)] = df

    # Create a comparison table
    if datasets:
        with open("eda_plots_comparison.txt", "w") as f:
            f.write("Comparison of Train and Test Datasets\n")
            f.write("=================================\n\n")

            # Compare class distributions
            f.write("1. Class Distribution Comparison\n")
            f.write("--------------------------\n")
            class_counts = {}
            for name, df in datasets.items():
                counts = df['class'].value_counts().sort_index()
                class_counts[name] = counts
                f.write(f"{name} class counts:\n{counts.to_string()}\n\n")

            # Compare event durations
            f.write("2. Event Duration Comparison\n")
            f.write("------------------------\n")
            for name, df in datasets.items():
                df = df.sort_values(['source', 'class', 'frame'])
                event_df = calculate_event_durations(df)
                duration_stats = event_df.groupby('class')['duration'].mean().sort_index()
                f.write(f"{name} mean event durations:\n{duration_stats.to_string()}\n\n")

            # Compare mean distances
            f.write("3. Mean Distance Comparison\n")
            f.write("------------------------\n")
            for name, df in datasets.items():
                distance_stats = df.groupby('class')['distance'].mean().sort_index()
                f.write(f"{name} mean distances:\n{distance_stats.to_string()}\n\n")

            # Compare onscreen ratios
            f.write("4. Onscreen Ratio Comparison\n")
            f.write("-------------------------\n")
            for name, df in datasets.items():
                onscreen_ratio = df.groupby('class')['onscreen'].mean().sort_index()
                f.write(f"{name} onscreen ratios:\n{onscreen_ratio.to_string()}\n\n")

            # If there are both train and test datasets, analyze class representation differences
            train_dfs = [df for name, df in datasets.items() if 'train' in name]
            test_dfs = [df for name, df in datasets.items() if 'test' in name]

            if train_dfs and test_dfs:
                f.write("5. Train vs Test Class Representation\n")
                f.write("--------------------------------\n")

                train_df = pd.concat(train_dfs, ignore_index=True)
                test_df = pd.concat(test_dfs, ignore_index=True)

                train_counts = train_df['class'].value_counts(normalize=True).sort_index()
                test_counts = test_df['class'].value_counts(normalize=True).sort_index()

                # Create a DataFrame with the comparison
                comparison = pd.DataFrame({
                    'Train %': train_counts * 100,
                    'Test %': test_counts * 100,
                })
                comparison['Difference %'] = comparison['Train %'] - comparison['Test %']

                f.write(f"Class representation comparison:\n{comparison.to_string()}\n\n")

                # Identify classes with significant differences
                threshold = 5.0  # 5% difference threshold
                significant_diff = comparison[abs(comparison['Difference %']) > threshold]

                if not significant_diff.empty:
                    f.write(f"Classes with significant representation differences (>{threshold}%):\n")
                    f.write(f"{significant_diff.to_string()}\n\n")
                    f.write("These classes may require special attention for data augmentation.\n\n")

        print(f"Comparison summary saved to eda_plots_comparison.txt")
    else:
        print("No datasets available for comparison")

except Exception as e:
    print(f"Error generating comparison summary: {e}")

print("All analyses completed!")