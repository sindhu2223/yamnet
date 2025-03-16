import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. Data preparation functions
def load_and_preprocess_audio(file_path, start_time=None, end_time=None, target_sr=16000):
    """Load and preprocess audio for YAMNet"""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        # Return silence with small noise as fallback
        return np.random.normal(0, 0.001, target_sr)
    
    # Load audio with librosa
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Trim audio if timestamps are provided
        if start_time is not None and start_time >= 0 and end_time is not None and end_time > start_time:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr) if end_time > 0 else len(audio)
            audio = audio[start_sample:min(end_sample, len(audio))]
            
        # Ensure minimum length (1 second)
        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)), mode='constant')
            
        # Standardize audio volume
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        return audio
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.random.normal(0, 0.001, target_sr)  # Return fallback audio

# 2. Audio data augmentation
def augment_audio(audio, sr=16000):
    """Apply random augmentation to audio"""
    augmented = audio.copy()
    
    # Apply augmentation techniques with 50% probability each
    
    # Time shift
    if np.random.random() > 0.5:
        shift_factor = np.random.uniform(-0.1, 0.1)
        shift_amount = int(len(audio) * shift_factor)
        if shift_amount > 0:
            augmented = np.pad(augmented, (shift_amount, 0), mode='constant')[:len(audio)]
        else:
            augmented = np.pad(augmented, (0, -shift_amount), mode='constant')[(-shift_amount):]
    
    # Add noise
    if np.random.random() > 0.5:
        noise_factor = np.random.uniform(0.001, 0.01)
        noise = np.random.randn(len(augmented)) * noise_factor
        augmented = augmented + noise
        
    # Pitch shift (using librosa)
    if np.random.random() > 0.5:
        n_steps = np.random.uniform(-2, 2)
        try:
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
        except:
            pass  # Skip if pitch shift fails
    
    # Ensure proper length and scaling
    if len(augmented) > len(audio):
        augmented = augmented[:len(audio)]
    elif len(augmented) < len(audio):
        augmented = np.pad(augmented, (0, len(audio) - len(augmented)), mode='constant')
        
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-10)
    
    return augmented

# 3. Dataset generator for TensorFlow
class AudioDataGenerator(tf.keras.utils.Sequence):
    """Generator class for batches of audio data"""
    def __init__(self, csv_file, batch_size=32, do_augment=False, shuffle=True):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.do_augment = do_augment
        self.shuffle = shuffle
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.data['label_encoded'] = self.label_encoder.fit_transform(self.data['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        # Get indices for this batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data.iloc[batch_indices]
        
        # Initialize arrays
        batch_x = []
        batch_y = []
        
        # Process each item in the batch
        for _, row in batch_data.iterrows():
            # Load audio
            audio = load_and_preprocess_audio(
                row['file_path'], 
                row.get('start_time'), 
                row.get('end_time')
            )
            
            # Apply augmentation if needed
            if self.do_augment:
                audio = augment_audio(audio)
                
            batch_x.append(audio)
            batch_y.append(row['label_encoded'])
        
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_label_mapping(self):
        """Get mapping from encoded to original labels"""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}

# 4. YAMNet model creation
def create_yamnet_model(num_classes):
    # Load YAMNet base model
    base_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Create model architecture
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    
    # YAMNet processes the audio to extract embeddings
    _, embeddings_output, _ = base_model(input_layer)
    
    # Add classification layers
    x = tf.keras.layers.Dense(256, activation='relu')(embeddings_output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# 5. Main training function
def train_yamnet_from_csv(
    train_csv, 
    val_csv=None,
    output_dir='yamnet_model',
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    val_split=0.2,
    class_weights=None
):
    """Train YAMNet model using CSV annotations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generators
    train_generator = AudioDataGenerator(
        train_csv, 
        batch_size=batch_size, 
        do_augment=True, 
        shuffle=True
    )
    
    # Either use provided validation CSV or split from training
    if val_csv:
        val_generator = AudioDataGenerator(
            val_csv, 
            batch_size=batch_size, 
            do_augment=False, 
            shuffle=False
        )
    else:
        # Split train data
        train_df = pd.read_csv(train_csv)
        train_df, val_df = train_test_split(train_df, test_size=val_split, stratify=train_df['label'], random_state=42)
        
        # Save split datasets
        train_split_csv = os.path.join(output_dir, 'train_split.csv')
        val_split_csv = os.path.join(output_dir, 'val_split.csv')
        train_df.to_csv(train_split_csv, index=False)
        val_df.to_csv(val_split_csv, index=False)
        
        # Create generators from split
        train_generator = AudioDataGenerator(
            train_split_csv, 
            batch_size=batch_size, 
            do_augment=True, 
            shuffle=True
        )
        val_generator = AudioDataGenerator(
            val_split_csv, 
            batch_size=batch_size, 
            do_augment=False, 
            shuffle=False
        )
    
    # Get number of classes
    num_classes = train_generator.num_classes
    
    # Save label mapping
    label_mapping = train_generator.get_label_mapping()
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)
    
    # Create the model
    model = create_yamnet_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs')
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights  # Optional class weights
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model'))
    
    return model, history

# 6. Inference function
def predict_with_yamnet(model_path, audio_file, label_mapping_path, start_time=None, end_time=None):
    """Make predictions with trained YAMNet model"""
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load label mapping
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    # Load and preprocess audio
    audio = load_and_preprocess_audio(audio_file, start_time, end_time)
    
    # For longer audio, process in segments
    if len(audio) > 16000*5:  # If longer than 5 seconds
        segment_length = 16000  # 1 second
        segments = []
        
        # Create overlapping segments
        for i in range(0, len(audio) - segment_length, segment_length // 2):
            segment = audio[i:i + segment_length]
            segments.append(segment)
        
        # Get predictions for each segment
        predictions = []
        for segment in segments:
            pred = model.predict(np.array([segment]), verbose=0)
            predictions.append(pred[0])
        
        # Average predictions
        final_prediction = np.mean(np.array(predictions), axis=0)
    else:
        # For short audio, just get a single prediction
        final_prediction = model.predict(np.array([audio]), verbose=0)[0]
    
    # Get top predictions
