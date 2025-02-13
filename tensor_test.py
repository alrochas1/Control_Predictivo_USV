from tensorflow.keras.datasets import mnist

# Preparar los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


# Diseñar el modelo
from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])


# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")


# Realizar predicciones
predictions = model.predict(x_test[:5])
print(predictions)


# Guardar y cargar el modelo
model.save('model.keras')

from tensorflow.keras.models import load_model
loaded_model = load_model('model.h5')


# Recursos avanzados

#     Experimenta con callbacks como EarlyStopping o TensorBoard.
#     Usa la API funcional para arquitecturas más complejas.
#     Trabaja con datos grandes utilizando tf.data.

