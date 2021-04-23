from train_mask_detector import ModelTrainer
from arduino_adapter import ArduinoAdapter
from loguru import logger
import argparse
import os

class Application:

    def __init__(self, model: str, retrain_model: bool = False, dataset: str = "") -> None:
        if retrain_model or not os.path.isfile(model):
            logger.info("Retraining model...")
            self.retrain_model(dataset, model)
        self.model = model
        self.arduino = ArduinoAdapter()

    def retrain_model(self, dataset: str, model: str) -> None:
        model_trainer = ModelTrainer(dataset, model)
        model_trainer.train()

    def run(self):
        pass



def run(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
    ap.add_argument("-r", "--retrain", action="store_true", help="path to input dataset")
    ap.add_argument("-d", "--dataset", default="dataset", help="path to input dataset")
    args = vars(ap.parse_args())
    app = Application(args["model"], args["retrain"], args["dataset"])
    app.run()

if __name__ == "__main__":
    run()