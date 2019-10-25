# from trainer.customtrainer import FlowTrainer
from trainer.pyramidalTrainer import FlowTrainer


if __name__ == "__main__":
    trainer = FlowTrainer()
    trainer.run()
