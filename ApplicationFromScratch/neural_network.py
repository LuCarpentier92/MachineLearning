import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.losses import BinaryCrossentropy
import warnings

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def relu(tensor):
    return tf.maximum(0.0, tensor)

def sigmoid(output):
    return 1 / (1 + tf.exp(-output))

def accuracy(y_pred, y_real):
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_real[i]:
            count+=1
    return 100 * (count / len(y_pred))

class Layer:
    def __init__(self, input_size, output_size):
        initializer = GlorotUniform(seed=42)
        self.weights = tf.Variable(initializer([input_size, output_size]), trainable=True)
        self.biases = tf.Variable(initializer([1, output_size]), trainable=True)


    def forward(self, inputs, apply_activation):
        self.inputs = inputs
        self.z = tf.matmul(inputs, self.weights) + self.biases
        if apply_activation == 'relu':
            self.output = relu(self.z)
        elif apply_activation == 'sigmoid':
            self.output = sigmoid(self.z)
        else:
            self.output = self.z
        return self.output

class NeuralNetwork:
    def __init__(self, verbose):
        self.layers = []
        self.verbose = verbose
        self.loss_fn = BinaryCrossentropy()
        self.loss = float
        self.trainable_variables = None
        self.moments = []
        self.variances = []
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999

    def add_layer(self, input_size, output_size):
        self.layers.append(Layer(input_size, output_size))

    def predict(self, inputs):
        tensor = inputs
        for i, layer in enumerate(self.layers):
            apply_activation = 'relu' if i < len(self.layers) - 1 else 'sigmoid'
            tensor = layer.forward(tensor, apply_activation)
        return tf.round(tensor)

    def compute_gradient(self, learning_rate,optimizer, t=1 ):
        with tf.GradientTape() as tape:
            tensor = X
            for i, layer in enumerate(self.layers):
                apply_activation = 'relu' if i < len(self.layers) - 1 else 'sigmoid'
                tensor = layer.forward(tensor, apply_activation)

            predictions = tensor
            self.loss = self.loss_fn(y, predictions)

            self.trainable_variables = []
            for layer in self.layers:
                self.trainable_variables.extend([layer.weights, layer.biases])

            grads = tape.gradient(self.loss, self.trainable_variables)


            if optimizer == 'adam':
                if not self.moments:
                    self.moments = [tf.zeros_like(var) for var in grads]
                    self.variances = [tf.zeros_like(var) for var in grads]

                updated_var = []

                for i,grad in enumerate(grads):
                    self.moments[i]= self.beta1*self.moments[i] + (1-self.beta1)*grad
                    self.variances[i] = self.beta2*self.variances[i] + (1-self.beta2)*tf.square(grad)


                for i, layer in enumerate(self.layers):
                    # Mise à jour des poids
                    m_hat_weights = self.moments[2 * i] / (1 - self.beta1 ** t)
                    v_hat_weights = self.variances[2 * i] / (1 - self.beta2 ** t)
                    update_weights = learning_rate * m_hat_weights / (tf.sqrt(v_hat_weights) + self.epsilon)
                    layer.weights.assign_sub(update_weights)

                    # Mise à jour des biais
                    m_hat_biases = self.moments[2 * i + 1] / (1 - self.beta1 ** t)
                    v_hat_biases = self.variances[2 * i + 1] / (1 - self.beta2 ** t)
                    update_biases = learning_rate * m_hat_biases / (tf.sqrt(v_hat_biases) + self.epsilon)
                    layer.biases.assign_sub(update_biases)
            else:
                for i, layer in enumerate(self.layers):
                    layer.weights.assign_sub(learning_rate * grads[2 * i])
                    layer.biases.assign_sub(learning_rate * grads[2 * i + 1])

    def train(self, X ,y , epochs, learning_rate, optimizer):
        for epoch in range(epochs):
            self.compute_gradient(learning_rate, optimizer)
            if self.verbose == 1 and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {np.array(self.loss)}")

# Running the neural network
if __name__ == "__main__":
    # XOR dataset
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

    # Define and train the neural network
    nn = NeuralNetwork(verbose=True)
    nn.add_layer(2, 4)
    nn.add_layer(4, 1)

    # Train the model with reduced learning rate
    nn.train(X, y, epochs=10000, learning_rate=0.01, optimizer = 'adam')

    # Make predictions
    print("Prédictions:")
    y_pred = nn.predict(X)
    print(y_pred.numpy())
    print(f"Score de prediction {accuracy(y_pred, y)}%")
    print(accuracy(y_pred, y))




