from mask_detector import MaskDetector
from train_mask_detector import ModelTrainer
from arduino_adapter import ArduinoAdapter
from loguru import logger
from threading import Thread
import argparse
import os

class Application:

    def __init__(self, model: str, retrain_model: bool = False, dataset: str = "", face_detector: str = "", confidence: int = 0.5, source: int = 0) -> None:
        if retrain_model or not os.path.isfile(model):
            logger.info("Retraining model...")
            self.retrain_model(dataset, model)
        self.model = model
        self.arduino = ArduinoAdapter()
        self.mask_detector = MaskDetector(face_detector, model, confidence, source)

    def retrain_model(self, dataset: str, model: str) -> None:
        model_trainer = ModelTrainer(dataset, model)
        model_trainer.train()

    def run(self):
        #TODO Thread run
        self.mask_detector.run()



def run(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
    ap.add_argument("-r", "--retrain", action="store_true", help="path to input dataset")
    ap.add_argument("-d", "--dataset", default="dataset", help="path to input dataset")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--source", type=int, default=0, help="integer for source camera")
    args = vars(ap.parse_args())
    app = Application(args["model"], args["retrain"], args["dataset"], args["face"], args["confidence"], args["source"])
    app.run()

if __name__ == "__main__":
    run()