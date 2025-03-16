# yamnet
Let me explain the code I provided, which creates a pipeline for training YAMNet using CSV files derived from annotated JSON data.
1. JSON to CSV Conversion
The first part of the code converts your JSON annotation files to CSV format:

convert_json_to_csv(): This function takes a JSON file with audio annotations and converts it to a CSV file.

It extracts each audio file path, label(s), and time boundaries from the JSON
It handles both single-label and multi-label scenarios
It preserves any additional metadata from the JSON
It outputs a structured CSV with columns like file_path, label, start_time, end_time


process_all_json_files(): This function processes multiple JSON files in a directory:

It converts each JSON file to a separate CSV
It also combines all the individual CSVs into one master CSV file
This is useful if you have multiple JSON files for different datasets or splits



2. Audio Processing Functions
These functions handle audio data loading and augmentation:

load_and_preprocess_audio(): This loads and prepares audio files for YAMNet:

It uses librosa to load audio at the required 16kHz sample rate for YAMNet
It can trim audio using start_time and end_time values
It ensures audio has minimum required length (padding if necessary)
It normalizes audio volume for consistency
It includes error handling for missing files


augment_audio(): This function applies random augmentations to audio samples:

Time shifting: Moves the audio slightly forward or backward
Noise addition: Adds small random noise to improve robustness
Pitch shifting: Changes the pitch slightly without affecting speed
Each augmentation is applied with 50% probability
This helps prevent overfitting and improves model generalization



3. AudioDataGenerator Class
This class inherits from TensorFlow's Sequence class to efficiently generate batches of audio data:

__init__(): Initializes the generator:

Loads the CSV file
Encodes text labels to numeric using LabelEncoder
Sets up batching and shuffling parameters


__len__(): Returns the number of batches per epoch
__getitem__(): Generates one batch of data:

Gets the corresponding subset of data for the batch
Loads and processes each audio file
Applies augmentation if enabled
Returns arrays of audio samples and corresponding labels


on_epoch_end(): Shuffles data after each epoch
get_label_mapping(): Returns a dictionary mapping between numeric indices and original label names

4. YAMNet Model Creation

create_yamnet_model(): Builds a model that fine-tunes YAMNet:

Loads the pre-trained YAMNet model from TensorFlow Hub
Sets up the input layer for raw audio
Uses YAMNet to extract audio embeddings
Adds custom classification layers (Dense, BatchNormalization, Dropout)
Creates and returns the final model architecture



5. Main Training Function

train_yamnet_from_csv(): Orchestrates the entire training process:

Creates data generators for training and validation
Can either use a separate validation CSV or split the training data
Sets up the model with the correct number of classes
Saves the label mapping for later use in inference
Configures training with callbacks:

ModelCheckpoint: Saves the best model
EarlyStopping: Prevents overfitting by stopping when validation loss plateaus
ReduceLROnPlateau: Reduces learning rate when progress stalls
TensorBoard: For visualization of training metrics


Trains the model and saves training history
Saves the final trained model



6. Inference Function

predict_with_yamnet(): Uses the trained model to make predictions:

Loads the model and label mapping
Processes the audio file
For longer audio files, it processes in overlapping segments
Averages predictions across segments for more stable results
Returns top predictions with their probabilities



How to Use This Pipeline

First, convert your JSON annotations to CSV:

pythonCopyprocess_all_json_files("path/to/json_annotations", "path/to/output_csv_files")

Then train the model using the generated CSV:

pythonCopymodel, history = train_yamnet_from_csv(
    train_csv="path/to/output_csv_files/combined_annotations.csv",
    output_dir="my_yamnet_model",
    epochs=30,
    batch_size=32
)

For inference on new audio:

pythonCopypredictions = predict_with_yamnet(
    model_path="my_yamnet_model/final_model",
    audio_file="path/to/new_audio.wav",
    label_mapping_path="my_yamnet_model/label_mapping.json"
)
The key advantage of this approach is that CSV files are more universal and easier to work with than JSON for many machine learning workflows. They also make it simpler to split data, perform stratified sampling, and manage class distributions.
