from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

class MaskDetector():

	def __init__(self, face_detector, model, confidence, source):
		self.face_detector = face_detector
		self.model = model
		self.confidence = confidence
		self.source = source
		self.prototxtPath = os.path.sep.join([self.face_detector, "deploy.prototxt"])
		self.weightsPath = os.path.sep.join([self.face_detector,
		"res10_300x300_ssd_iter_140000.caffemodel"])
		self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
		self.maskNet = load_model(self.model)

	def detect_and_predict_mask(self,frame):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		self.faceNet.setInput(blob)
		detections = self.faceNet.forward()

		faces = [] 
		locs = []
		preds = []
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > self.confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype="float32")
			preds = self.maskNet.predict(faces, batch_size=32)
		return locs,preds

	# loop over the frames from the video stream
	def run(self):
		print("[INFO] loading face detector model...")
		print("[INFO] loading face mask detector model...")
		print("[INFO] starting video stream...")
		# initialize the video stream and allow the camera sensor to warm up
		vs = VideoStream(src=self.source).start()
		time.sleep(2.0)
		
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			locs,preds = self.detect_and_predict_mask(frame)
			# detect faces in the frame and determine if they are wearing a
			# face mask or not
			

			# loop over the detected face locations and their corresponding
			# locations
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw
				# the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		cv2.destroyAllWindows()
		vs.stop()

def run():
		# construct the argument parser and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
		ap.add_argument("-m", "--model", type=str, default="mask_detector.model",help="path to trained face mask detector model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
		ap.add_argument("-s", "--source", type=int, default=0, help="integer for source camera")
		args = vars(ap.parse_args())
		program = MaskDetector(args["face"],args["model"],args["confidence"],args["source"])
		program.run()

if __name__ == "__main__":
	run()
	
	