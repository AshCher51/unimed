# UniMed

PyTorch Implementation of UniMed: Multimodal Multitask Learning for Medical Predictions


Instructions:
1. Download Clinical Bert from [here](https://drive.google.com/file/d/1X3WrKLwwRAVOaAfKQ_tkTi46gsPfY5EB) and place the `ClinicalBERT_checkpoint` directory in the root directory of this repo. Only the `ClinicalBERT_pretraining_pytorch_checkpoint` folder was used.
2. Run the `src/prep.ipynb` in order to generate preprocessed datasets from FIDDLE preprocessed data (`data/X.npz`, `data/s.npz`) and labels (`data/Shock.csv`, `data/ARF.csv`, `data/mortality_48.0h.csv`). You should also download the NOTEEVENTS, DIAGNOSES_ICD, and ICUSTAYS csv files from MIMIC-III for the multimodal text incorporation that should be stored also in the `data/` directory (I used also the feather format to store NOTEEVENTS to allow for faster dataset read-in time)
3. Run `python unimed.py` in the `src` directory to reproduce the UniMed paper's results!

wandb link to my test run: https://wandb.ai/ashcher51/unimed
