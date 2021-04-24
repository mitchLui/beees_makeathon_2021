from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from typing import Tuple
from loguru import logger
import numpy as np
import argparse
import os

class ModelTrainer:


	def __init__(self, dataset_path: str, model_path: str) -> None:
		logger.info("__init__ trainer")
		self.dataset_path = dataset_path
		self.model_path = model_path
		self.INIT_LR = 1e-4
		self.EPOCHS = 20
		self.BS = 32

	def load_images(self) -> Tuple[list, list]:
		# grab the list of images in our dataset directory, then initialize
		# the list of data (i.e., images) and class images
		logger.info("loading images...")
		imagePaths = list(paths.list_images(self.dataset_path))
		data = []
		labels = []
		# loop over the image paths
		for imagePath in imagePaths:
			# extract the class label from the filename
			label = imagePath.split(os.path.sep)[-2]

			# load the input image (224x224) and preprocess it
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)
			image = preprocess_input(image)

			# update the data and labels lists, respectively
			data.append(image)
			labels.append(label)

		data = np.array(data, dtype="float32")
		labels = np.array(labels)
		return data, labels

	def preprocess_labels(self, labels) -> Tuple[LabelBinarizer, list]:
		logger.info("preprocessing data")
		# perform one-hot encoding on the labels
		lb = LabelBinarizer()
		labels = lb.fit_transform(labels)
		labels = to_categorical(labels)
		return lb, labels
	
	def get_ImageDataGenerator(self) -> ImageDataGenerator:
		logger.info("loading ImageDataGenerator...")
		return ImageDataGenerator(
			rotation_range=20,
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
			fill_mode="nearest")
	
	def load_baseModel(self) -> Model:
		logger.info("loading baseModel...")
		return MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

	def load_headModel(self, baseModel) -> Model: 
		logger.info("loading headModel...")
		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(128, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation="softmax")(headModel)
		return headModel

	def compile_model(self, model: Model, baseModel) -> Tuple[Model, Model]:
		logger.info("compiling model...")
		for layer in baseModel.layers:
			layer.trainable = False
		opt = Adam(learning_rate=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,
			metrics=["accuracy"])
		return model, baseModel

	def train_network_head(self, model: Model, aug: ImageDataGenerator, trainX, testX, trainY, testY) -> Model:
		logger.info("training head...")
		H = model.fit(
			aug.flow(trainX, trainY, batch_size=self.BS),
			steps_per_epoch=len(trainX) // self.BS,
			validation_data=(testX, testY),
			validation_steps=len(testX) // self.BS,
			epochs=self.EPOCHS
		)
		return H

	def evaluate_network(self, model: Model, lb: LabelBinarizer, testX, testY) -> None:
		# make predictions on the testing set
		logger.info("evaluating network...")
		# for each image in the testing set we need to find the index of the
		# label with corresponding largest predicted probability
		predIdxs = model.predict(testX, batch_size=self.BS)
		predIdxs = np.argmax(predIdxs, axis=1)
		# show a nicely formatted classification report
		print(classification_report(testY.argmax(axis=1), predIdxs,
			target_names=lb.classes_))

	def train(self) -> None:
		logger.info("beginning training...")
		data, labels = self.load_images()
		lb, labels = self.preprocess_labels(labels)
		# partition the data into training and testing splits using 75% of
		# the data for training and the remaining 25% for testing
		(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
		aug = self.get_ImageDataGenerator()
		baseModel = self.load_baseModel()
		headModel = self.load_headModel(baseModel)
		# place the head FC model on top of the base model (this will become the actual model we will train)
		model = Model(inputs=baseModel.input, outputs=headModel)
		model, baseModel = self.compile_model(model, baseModel)
		H = self.train_network_head(model, aug, trainX, testX, trainY, testY)
		self.evaluate_network(model, lb, testX, testY)
		logger.info("saving mask detector model...")
		model.save(self.model_path, save_format="h5")


def run() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
	ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
	args = vars(ap.parse_args())
	trainer = ModelTrainer(args["dataset"], args["model"])
	trainer.train()

if __name__ == "__main__":
	run()