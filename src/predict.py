
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Dense, Lambda, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
import backbone
import model as mod

# Define amsoftmax_loss function
def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

# Define a class to hold model arguments
class ModelArgs:
    def __init__(self, net, loss, vlad_cluster, ghost_cluster, bottleneck_dim, aggregation_mode, optimizer):
        self.net = net
        self.loss = loss
        self.vlad_cluster = vlad_cluster
        self.ghost_cluster = ghost_cluster
        self.bottleneck_dim = bottleneck_dim
        self.aggregation_mode = aggregation_mode
        self.optimizer = optimizer

# Define the model architecture
def vggvox_resnet2d_icassp(input_dim=(257, 250, 1), num_class=8631, mode='eval', args=None):
    if args is None:
        raise ValueError("The 'args' parameter is required but not provided.")
    
    if not hasattr(args, 'loss'):
        raise AttributeError("The 'args' object must have an attribute 'loss'.")
    mode='eval'
    net = args.net
    loss = args.loss
    vlad_clusters = args.vlad_cluster
    ghost_clusters = args.ghost_cluster
    bottleneck_dim = args.bottleneck_dim
    aggregation = args.aggregation_mode
    mgpu = len(tf.config.list_physical_devices('GPU'))
    
    weight_decay = 1e-4  # Define weight decay for regularization

    # Initialize the ResNet architecture
    if net == 'resnet34s':
        inputs, x = backbone.resnet_2D_v1(input_dim=input_dim, mode=mode)
    else:
        inputs, x = backbone.resnet_2D_v2(input_dim=input_dim, mode=mode)
    
    # Fully Connected Block 1
    x_fc = Conv2D(bottleneck_dim, (7, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer='orthogonal',
                   use_bias=True, trainable=True,
                   kernel_regularizer=l2(weight_decay),
                   bias_regularizer=l2(weight_decay),
                   name='x_fc')(x)

    # Feature Aggregation
    if aggregation == 'avg':
        if mode == 'train':
            x = AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
            x = tf.reshape(x, (-1, bottleneck_dim))
        else:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.reshape(x, (1, bottleneck_dim))

    elif aggregation == 'vlad':
        x_k_center = Conv2D(vlad_clusters, (7, 1),
                            strides=(1, 1),
                            kernel_initializer='orthogonal',
                            use_bias=True, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='vlad_center_assignment')(x)
        x = mod.VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = Conv2D(vlad_clusters + ghost_clusters, (7, 1),
                            strides=(1, 1),
                            kernel_initializer='orthogonal',
                            use_bias=True, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='gvlad_center_assignment')(x)
        x = mod.VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([x_fc, x_k_center])

    else:
        raise ValueError('==> unknown aggregation mode')

    # Fully Connected Block 2
    x = Dense(bottleneck_dim, activation='relu',
              kernel_initializer='orthogonal',
              use_bias=True, trainable=True,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay),
              name='fc6')(x)

    # Softmax Vs AMSoftmax
    if loss == 'softmax':
        y = Dense(num_class, activation='softmax',
                  kernel_initializer='orthogonal',
                  use_bias=False, trainable=True,
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay),
                  name='prediction')(x)
        trnloss = 'categorical_crossentropy'

    elif loss == 'amsoftmax':
        x_l2 = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = Dense(num_class,
                  kernel_initializer='orthogonal',
                  use_bias=False, trainable=True,
                  kernel_constraint=tf.keras.constraints.UnitNorm(),
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay),
                  name='prediction')(x_l2)
        trnloss = amsoftmax_loss

    else:
        raise ValueError('==> unknown loss.')

    if mode == 'eval':
        y = Lambda(lambda x: K.l2_normalize(x, 1))(x)

    model = Model(inputs, y, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))

    if mode == 'train':
        if mgpu > 1:
            model = ModelMGPU(model, gpus=mgpu)
        # Set up optimizer
        if args.optimizer == 'adam':
            opt = Adam(learning_rate=1e-3)
        elif args.optimizer == 'sgd':
            opt = SGD(learning_rate=0.1, momentum=0.9, decay=0.0, nesterov=True)
        else:
            raise ValueError('==> unknown optimizer type')
        model.compile(optimizer=opt, loss=trnloss, metrics=['acc'])
    return model

def load_pretrained_model(model_path):
    """
    Load the pretrained VGGVox model with weights.
    """
    # Define model arguments or configuration here
    args = ModelArgs(
        net='resnet34s',
        loss='softmax',
        vlad_cluster=8,
        ghost_cluster=2,
        bottleneck_dim=512,
        aggregation_mode='vlad',
        optimizer='adam'
    )
    
    # Initialize the model
    model = vggvox_resnet2d_icassp(args=args)
    
    # Print model summary to check layer structure
    # model.summary()
    
    try:
        # Load weights
        model.load_weights(model_path)
    except ValueError as e:
        print("Error loading weights:", e)
        # Handle error or re-load model with updated structure
    
    return model


def preprocess_wav_file(wav_path, target_sr=16000, duration=5):
    """
    Load and preprocess a WAV file for model input.
    """
    y, sr = librosa.load(wav_path, sr=target_sr, duration=duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)
    
    if log_mel_spectrogram.shape[1] < 250:
        padding = 250 - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :250]

    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)
    return np.expand_dims(log_mel_spectrogram, axis=0)
def extract_embeddings(model, wav_path):
    # Preprocess the audio file
    preprocessed_audio = preprocess_wav_file(wav_path)
    
    # Print shape for debugging
    print("Shape before padding:", preprocessed_audio.shape)
    
    # Apply padding
    if preprocessed_audio.shape[1] < 257:
        # Calculate padding width
        pad_width = ((0, 0), (0, 257 - preprocessed_audio.shape[1]), (0, 0), (0, 0))
        # Apply padding
        preprocessed_audio = np.pad(preprocessed_audio, pad_width, mode='constant')
    
    # Print shape after padding
    print("Shape after padding:", preprocessed_audio.shape)

    # Predict embeddings
    embeddings = model.predict(preprocessed_audio)
    print("Embeddings shape:", embeddings.shape)
    return embeddings

# Path to the pretrained model and WAV file
model_path = 'weights.h5'
wav_path_0 = '00001.wav'
wav_path_1 = '00002.wav'
wav_path_2 = '00003.wav'



# Load the model
model = load_pretrained_model(model_path)

# Extract embeddings

embeddings0 = extract_embeddings(model, wav_path_0)
embeddings1 = extract_embeddings(model, wav_path_1)
embeddings2 = extract_embeddings(model, wav_path_2)

# Calculate cosine similarity
print('Same Person Similarity')
similarity = np.dot(embeddings0, embeddings1.T)
print(similarity)

print('Different Person Similarity')
similarity = np.dot(embeddings0, embeddings2.T)
print(similarity)

print('High EER expected!!!')