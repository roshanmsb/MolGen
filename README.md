# Pretrain GPT-2 on Chemical SMILES Strings

This project aims to pretrain the GPT-2 model from Hugging Face's transformers library on chemical SMILES strings using PyTorch Lightning and Hydra configs.

## Project Structure

The project has the following file structure:

```
pretrain-gpt2-chem-smiles
├── data
│   └── chem_smiles.txt
├── models
│   ├── gpt2.py
│   └── __init__.py
├── trainers
│   ├── gpt2_trainer.py
│   └── __init__.py
├── configs
│   ├── data
│   │   └── chem_smiles.yaml
│   ├── models
│   │   └── gpt2.yaml
│   └── trainers
│       └── gpt2_trainer.yaml
├── main.py
├── requirements.txt
└── README.md
```

The project has the following files:

- `data/chem_smiles.txt`: This file contains the chemical SMILES strings that will be used for pretraining the GPT-2 model.
- `models/gpt2.py`: This file exports a class `GPT2LMHeadModel` which is a modified version of the GPT-2 model from Hugging Face's transformers library. It includes a language modeling head that predicts the next token in the sequence.
- `trainers/gpt2_trainer.py`: This file exports a class `GPT2Trainer` which is responsible for training the GPT-2 model on the chemical SMILES strings. It uses PyTorch Lightning to handle the training loop and Hydra to manage the configuration.
- `configs/data/chem_smiles.yaml`: This file contains the configuration options for the chemical SMILES data. It specifies the path to the data file and the batch size for training.
- `configs/models/gpt2.yaml`: This file contains the configuration options for the GPT-2 model. It specifies the model architecture, the number of layers, and the size of the hidden state.
- `configs/trainers/gpt2_trainer.yaml`: This file contains the configuration options for the GPT-2 trainer. It specifies the learning rate, the number of epochs, and the optimizer.
- `main.py`: This file is the entry point of the application. It sets up the PyTorch Lightning trainer and Hydra configuration manager, and starts the training process.
- `requirements.txt`: This file lists the Python dependencies required for the project.
- `README.md`: This file contains the documentation for the project.

## Usage

To use the project, follow these steps:

1. Clone the repository: `git clone https://github.com/username/pretrain-gpt2-chem-smiles.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Modify the configuration files in the `configs` directory to suit your needs.
4. Run the training script: `python main.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.