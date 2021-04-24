from mask_detector import MaskDetector
from train_mask_detector import ModelTrainer
from arduino_adapter import ArduinoAdapter
from loguru import logger
from imutils.video import VideoStream
from time import sleep
import argparse
import imutils
import cv2
import os

class Application:

    def __init__(self, model: str, port: str, retrain_model: bool = False, dataset: str = "", face_detector: str = "", confidence: int = 0.8, source: int = 0) -> None:
        if retrain_model or not os.path.isfile(model):
            logger.info("Retraining model...")
            self.retrain_model(dataset, model)
        self.model = model
        self.arduino = ArduinoAdapter(port)
        self.mask_detector = MaskDetector(face_detector, model, confidence, source)

    def retrain_model(self, dataset: str, model: str) -> None:
        model_trainer = ModelTrainer(dataset, model)
        model_trainer.train()
    
    def set_arduino_led(self, signal: bool) -> None:
        if self.arduino.board:
            self.arduino.set_led(signal)

    def run(self) -> None:
        logger.info("starting video stream...")
        vs = VideoStream(src=self.mask_detector.source).start()
        sleep(2)
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            
            mask_array = []

            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            locs,preds = self.mask_detector.detect_and_predict_mask(frame)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            
            # loop over the detected face locations and their corresponding locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = f"{label}: {round(max(mask, withoutMask) * 100, 2)}%"

                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                if mask >= withoutMask:
                    mask_array.append(True)
                else:
                    mask_array.append(False)

            # show the output frame
            cv2.imshow("Mask Detector Prototype", frame)
            key = cv2.waitKey(1) & 0xFF

            led_signal = not any(mask_array)
            if led_signal:
                logger.warning(f"Someone is not wearing a mask.")
            else:
                logger.debug(f"Everyone is wearing a mask.")
            self.set_arduino_led(led_signal)
            
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        vs.stop()


def run() -> None: 
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
    ap.add_argument("-r", "--retrain", action="store_true", help="path to input dataset")
    ap.add_argument("-d", "--dataset", default="dataset", help="path to input dataset")
    ap.add_argument("-c", "--confidence", type=float, default=0.8, help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--source", type=int, default=0, help="integer for source camera")
    ap.add_argument("-p", "--port", type=str, default="/dev/cu.usbmodem12341", help="port for arduino")
    args = vars(ap.parse_args())
    app = Application(args["model"], args["port"], args["retrain"], args["dataset"], args["face"], args["confidence"], args["source"])
    app.run()

if __name__ == "__main__":
    run()