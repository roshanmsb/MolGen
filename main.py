import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from models.gpt2 import GPT2LMHeadModel
from trainers.gpt2_trainer import GPT2Trainer


@hydra.main(config_path="configs", config_name="trainers/gpt2_trainer")
def main(cfg: DictConfig):
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    # Load the data
    with open(hydra.utils.to_absolute_path(cfg.data.path)) as f:
        data = f.read().splitlines()

    # Initialize the model
    model = GPT2LMHeadModel(cfg.model)

    # Initialize the trainer
    trainer = pl.Trainer(**cfg.trainer)

    # Train the model
    trainer.fit(GPT2Trainer(model, data))


if __name__ == "__main__":
    main()