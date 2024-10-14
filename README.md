
# Transfer Learning with ResNet50 for Flower Classification 

This project demonstrates how to implement **Transfer Learning** using the pre-trained **ResNet50** model to classify different types of flowers. By leveraging a pre-trained model on the ImageNet dataset, we can reduce training time and improve performance on our specific dataset of flower images.

## Project Overview

- **Objective**: Train a model to classify flowers into 5 different classes using transfer learning.
- **Model**: ResNet50 (pre-trained on ImageNet) with additional custom layers for fine-tuning.
- **Dataset**: Flower dataset from [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz).
- **Libraries Used**: TensorFlow, Keras, OpenCV, Matplotlib, Numpy.

## Key Steps

1. **Install Dependencies**:
   Install the required libraries:
   ```bash
   %pip install tensorflow opencv-python matplotlib
   ```

2. **Prepare the Data**:
   We load the flower dataset, resize images to 180x180 pixels, and split the data into training and validation sets.
   ```python
   data_dir = Path("D:/Coding Projects/transfer-learning-with-ResNet_50/data")
   train_ds = tf.keras.preprocessing.image_dataset_from_directory(
       data_dir,
       validation_split=0.2,
       subset="training",
       seed=123,
       image_size=(180, 180),
       batch_size=32
   )
   val_ds = tf.keras.preprocessing.image_dataset_from_directory(
       data_dir,
       validation_split=0.2,
       subset="validation",
       seed=123,
       image_size=(180, 180),
       batch_size=32
   )
   ```

3. **Building the Model**:
   We use the pre-trained **ResNet50** model (with frozen layers) and add custom layers to fine-tune it for our dataset of flowers.
   ```python
   resnet_model = Sequential()
   pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                         input_shape=(180,180,3),
                         pooling='avg', weights='imagenet')
   for layer in pretrained_model.layers:
       layer.trainable = False

   resnet_model.add(pretrained_model)
   resnet_model.add(Flatten())
   resnet_model.add(Dense(512, activation='relu'))
   resnet_model.add(Dense(5, activation='softmax'))
   resnet_model.compile(optimizer=Adam(learning_rate=0.001), 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
   ```

4. **Training the Model**:
   Train the model for 6 epochs using the flower dataset:
   ```python
   history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=6)
   ```

5. **Evaluating the Model**:
   After training, we evaluate the model by plotting the accuracy and loss metrics:
   ```python
   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.title('Model Accuracy')
   plt.ylabel('Accuracy')
   plt.xlabel('Epochs')
   plt.legend(['train', 'validation'])
   plt.show()

   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('Model Loss')
   plt.ylabel('Loss')
   plt.xlabel('Epochs')
   plt.legend(['train', 'validation'])
   plt.show()
   ```

6. **Making Predictions**:
   Use the trained model to make predictions on new images:
   ```python
   image = cv2.imread(str(roses[6]))
   image_resized = cv2.resize(image, (180,180))
   image = np.expand_dims(image_resized, axis=0)
   pred = resnet_model.predict(image)
   output_class = class_names[np.argmax(pred)]
   print("The predicted class is", output_class)
   ```

## Dataset

The flower dataset contains images of five types of flowers:
- Daisies
- Dandelions
- Roses
- Sunflowers
- Tulips

The dataset is split into training and validation sets with an 80-20 ratio.

## Project Structure

```
├── ResNet_50.ipynb          # Jupyter notebook with the entire code
├── data/                    # Folder containing flower dataset
└── README.md                # Project description
```
## Results

- **Accuracy**: The model achieves high accuracy in classifying the flowers after fine-tuning.
- **Loss**: The loss decreases steadily, showing good model performance.
