import pandas as pd
import os
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


class ModelFusion:
    """
    Flexible model fusion system that allows selecting which models to combine
    using frame-level winner-takes-all strategy.
    """

    def __init__(self):
        # Define all available models and their metrics
        self.available_models = {
            'model_conformer_original': {
                'description': 'The Localization Specialist',
                'dir_name': 'model_conformer_original',
                'metrics': {
                    0: [0.49, 18.84], 1: [0.53, 16.96], 2: [0.26, 22.87],
                    3: [0.18, 25.65], 4: [0.19, 22.92], 5: [0.40, 23.03],
                    6: [0.17, 21.72], 7: [0.18, 17.64], 8: [0.36, 23.91],
                    9: [0.31, 13.39], 10: [0.14, 19.50],
                    11: [0.00, 999], 12: [0.00, 999],
                }
            },
            'sed_doa': {
                'description': 'The SED Specialist',
                'dir_name': 'sed_doa',
                'metrics': {
                    0: [0.52, 18.68], 1: [0.56, 15.44], 2: [0.47, 16.43],
                    3: [0.20, 22.23], 4: [0.22, 22.00], 5: [0.41, 22.05],
                    6: [0.17, 21.09], 7: [0.21, 13.25], 8: [0.43, 22.27],
                    9: [0.29, 11.78], 10: [0.16, 16.76], 11: [0.03, 53.00],
                    12: [0.03, 9.00],
                }
            },
            'model_conformer_aug': {
                'description': 'The Rare Class Specialist',
                'dir_name': 'model_conformer_aug',
                'metrics': {
                    0: [0.48, 19.60], 1: [0.51, 16.67], 2: [0.16, 40.58],
                    3: [0.17, 21.65], 4: [0.19, 22.44], 5: [0.32, 24.34],
                    6: [0.13, 25.88], 7: [0.18, 13.98], 8: [0.34, 25.22],
                    9: [0.30, 14.24], 10: [0.16, 28.70], 11: [0.09, 37.66],
                    12: [0.11, 15.02],
                }
            }
        }

        self.base_dir = 'final_eval_outputs'

    def list_available_models(self) -> None:
        """Print all available models and their descriptions."""
        print("Available models:")
        for i, (model_name, info) in enumerate(self.available_models.items(), 1):
            print(f"{i}. {model_name} - {info['description']}")

    def validate_models(self, selected_models: List[str]) -> bool:
        """Validate that all selected models exist."""
        invalid_models = [m for m in selected_models if m not in self.available_models]
        if invalid_models:
            print(f"Error: Invalid model(s): {invalid_models}")
            print("Available models:", list(self.available_models.keys()))
            return False
        return True

    def load_model_data(self, model_name: str, file_name: str) -> Optional[pd.DataFrame]:
        """Load CSV data for a specific model and file."""
        model_dir = os.path.join(self.base_dir, self.available_models[model_name]['dir_name'], 'dev-test')
        file_path = os.path.join(model_dir, file_name)

        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            return None

    def calculate_frame_score(self, predictions: pd.DataFrame, model_name: str) -> float:
        """Calculate average F-score for predictions in a frame."""
        if predictions.empty:
            return -1

        metrics = self.available_models[model_name]['metrics']
        f_scores = [metrics.get(class_id, [0])[0] for class_id in predictions['class']]
        return sum(f_scores) / len(f_scores)

    def fuse_models(self, selected_models: List[str], output_suffix: str = None) -> bool:
        """
        Fuse selected models using frame-level winner-takes-all strategy.

        Args:
            selected_models: List of model names to fuse
            output_suffix: Optional suffix for output directory name

        Returns:
            bool: True if successful, False otherwise
        """
        # Validate inputs
        if not self.validate_models(selected_models):
            return False

        if len(selected_models) < 2:
            print("Error: Need at least 2 models to fuse.")
            return False

        if output_suffix is None:
            output_suffix = "_".join(selected_models)

        output_dir = os.path.join(self.base_dir, f'fusion_{output_suffix}', 'dev-test')
        os.makedirs(output_dir, exist_ok=True)

        first_model_dir = os.path.join(
            self.base_dir,
            self.available_models[selected_models[0]]['dir_name'],
            'dev-test'
        )

        try:
            csv_files = [f for f in os.listdir(first_model_dir) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} files.")
            print(f"Fusing models: {', '.join(selected_models)}")
        except FileNotFoundError:
            print(f"Error: Directory not found '{first_model_dir}'.")
            return False

        for file_name in tqdm(csv_files, desc=f"Fusing {len(selected_models)} Models"):
            model_data = {}
            for model_name in selected_models:
                df = self.load_model_data(model_name, file_name)
                if df is not None:
                    model_data[model_name] = df

            if not model_data:
                print(f"Warning: No valid data found for {file_name}")
                continue

            # Get all unique frames across all models
            all_frames = set()
            for df in model_data.values():
                all_frames.update(df['frame'].unique())
            all_frames = sorted(all_frames)

            final_predictions_df = pd.DataFrame()

            # Process each frame
            for frame in all_frames:
                frame_predictions = {}
                frame_scores = {}

                # Get predictions and scores for this frame from each model
                for model_name, df in model_data.items():
                    preds = df[df['frame'] == frame]
                    frame_predictions[model_name] = preds
                    frame_scores[model_name] = self.calculate_frame_score(preds, model_name)

                # Find the winner (model with highest score)
                if frame_scores:
                    winner_model = max(frame_scores.keys(), key=lambda k: frame_scores[k])
                    winner_predictions = frame_predictions[winner_model]

                    if not winner_predictions.empty:
                        final_predictions_df = pd.concat([final_predictions_df, winner_predictions])

            if not final_predictions_df.empty:
                final_predictions_df.sort_values(by=['frame', 'source'], inplace=True)

            output_path = os.path.join(output_dir, file_name)
            final_predictions_df.to_csv(output_path, index=False)

        print(f"\nModel fusion complete! Outputs saved to: {output_dir}")
        return True

    def quick_fuse_presets(self, preset: str) -> bool:
        """
        Quick fusion presets for common combinations.

        Available presets:
        - 'all': Fuse all three models
        - 'specialists': Fuse SED and Rare Class specialists
        - 'conformers': Fuse both conformer models
        """
        presets = {
            'all': ['model_conformer', 'sed_doa', 'model_conformer_aug'],
            'specialists': ['sed_doa', 'model_conformer_aug'],
            'conformers': ['model_conformer', 'model_conformer_aug']
        }

        if preset not in presets:
            print(f"Invalid preset. Available presets: {list(presets.keys())}")
            return False

        return self.fuse_models(presets[preset], output_suffix=preset)


def main():
    """Example usage of the ModelFusion class."""
    fusion = ModelFusion()

    # Show available models
    fusion.list_available_models()
    print()

    print("=== Fusing All Models ===")
    fusion.quick_fuse_presets('all')
    print()

    print("=== Fusing Specialists Only ===")
    fusion.quick_fuse_presets('specialists')
    print()

    print("=== Custom Fusion ===")
    selected = ['model_conformer', 'sed_doa']  # Choose your models here
    fusion.fuse_models(selected, output_suffix='localization_and_sed')


if __name__ == '__main__':
    # Simple usage examples:

    fusion = ModelFusion()

    # Option 1: Use presets
    # fusion.quick_fuse_presets('all')          # Fuse all 3 models
    # fusion.quick_fuse_presets('specialists')  # Fuse SED + Rare Class
    # fusion.quick_fuse_presets('conformers')   # Fuse both conformers

    # Option 2: Custom selection
    # fusion.fuse_models(['model_conformer', 'sed_doa'])  # Any 2 models
    # fusion.fuse_models(['sed_doa', 'model_conformer_aug'], 'custom_combo')

    # Uncomment the line below to run with your preferred combination:
    # fusion.quick_fuse_presets('all')  # Change this to your preference
    fusion.fuse_models(['model_conformer_original', 'model_conformer_aug'])