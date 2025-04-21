# test_gpu.py
import os
import tensorflow as tf

# Imposta le variabili d'ambiente prima di importare TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Riduce i messaggi di log
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Permette la crescita graduale della memoria

# Stampa informazioni
print("TensorFlow versione:", tf.__version__)
print("GPU disponibili:", tf.config.list_physical_devices('GPU'))

# Esegui un piccolo test su GPU se disponibile
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Crea un tensore sulla GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
    print("Operazione su GPU completata:")
    print(c)
else:
    print("Nessuna GPU disponibile.")