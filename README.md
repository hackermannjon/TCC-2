# Dating Recommendation with Graph Neural Networks

This repository contains the code and thesis for a final year project at the University of Brasília (UnB). The project investigates how multimodal data can be used to recommend potential matches on dating platforms. A modified version of **GraphRec** is trained to predict user preferences using profile images, textual descriptions and synthetic social connections.

## Repository layout

- `src/` – Python modules and scripts used in the pipeline
- `data/` – expected location for datasets and model checkpoints (not tracked in Git)
- `outputs/` – figures and tables generated for the final report
- `TCC2.pdf` – final monograph (in Portuguese)

## Requirements

The code targets **Python 3.10** or newer. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data preparation

The repository does not ship the raw datasets. To reproduce the experiments, place the following files under `data/raw/`:

1. `okcupid_profiles.csv` – available on Kaggle
2. Profile images in `data/raw/images/ProfilesDataSet`
3. An interaction log `interactions.csv` with columns `profile_id`, `like` and `timestamp`

After collecting the interaction log, run `python -m src.scripts.split_interactions` to create the train/test splits used by the pipeline.

## Running the pipeline

All steps—from feature extraction to evaluation—can be executed with:

```bash
python -m src.scripts.run_all
```

This command will:
1. generate multimodal features (`src/models/features.py`)
2. build a synthetic social graph (`src/models/social_graph.py`)
3. concatenate features (`src/models/combine_features.py`)
4. create user personas for evaluation (`src/scripts/create_personas.py`)
5. train the GraphRec model (`src/models/train_graphrec.py`)
6. compare baseline vs. persona performance (`src/scripts.evaluate_comparison.py`)

Artifacts such as `combined_features.npy`, `graphrec.pth` and evaluation tables are stored under `data/` and `outputs/`.

## Reproducing figures for the report

Additional plots and CSVs used in the thesis can be generated with:

```bash
python -m src.scripts.generate_report_data
```

## License

The thesis template inside `TCC2/` follows the Creative Commons Attribution-ShareAlike 4.0 license. Code in `src/` is provided under the MIT License.
