import utils
# from model import SELDModel
from model_conformer import SELDConformerModel as SELDModel
# from model_conformer3 import  SELDConformerModel as SELDModel
import pickle
import os
import torch
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor


def run_inference():
    params_file = os.path.join(model_dir, 'config.pkl')
    f = open(params_file, "rb")
    params = pickle.load(f)

    reference = model_dir.split('/')[-1]
    output_dir = 'final_eval_outputs/model_conformer_original'
    os.makedirs(output_dir, exist_ok=True)

    params['root_dir'] = '../DCASE2025_SELD_dataset'

    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='eval')  # Changed to 'eval' for clarity

    test_dataset = DataGenerator(params=params, mode='eval')
    test_iterator = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'],
                               shuffle=False, drop_last=False)

    num_eval_files = len(test_dataset.audio_files)
    output_filenames = [f"sample{i + 1:04d}.csv" for i in range(num_eval_files)]

    seld_model = SELDModel(params).to(device)
    model_ckpt = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device, weights_only=False)
    # model_ckpt = torch.load(os.path.join(model_dir, 'best_model_sed_doa_model.pth'), map_location=device, weights_only=False)
    seld_model.load_state_dict(model_ckpt['seld_model'])

    seld_model.eval()
    with torch.no_grad():
        for j, (input_features, _) in enumerate(test_iterator):

            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            logits = seld_model(audio_features, video_features)
            # preds = seld_model(audio_features, None)
            #
            # output_list = [preds['doa'].reshape(preds['doa'].size(0), preds['doa'].size(1), -1), torch.zeros_like(preds['sed'])]
            # logits = torch.cat(output_list, dim=2)

            batch_filenames = output_filenames[j * params['batch_size']: (j + 1) * params['batch_size']]

            utils.write_logits_to_dcase_format(logits, params, output_dir, batch_filenames)

        print(f"Inference complete. CSV outputs are in the '{output_dir}' directory.")


if __name__ == '__main__':
    model_dir = "checkpoints/orig_mc_trained_ckpt_1"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_inference()