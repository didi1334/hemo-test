import keras as k
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_frame = pd.read_csv("hemog1.csv")
input_names = ["age", "sex", "red","green","blue","rgb","hemoglobin"]
output_names = ["anemia"]

max_age = 100
max_color_red = 255
max_color_green = 255
max_color_blue = 255
max_color_rgb = 255
max_hemoglobin = 100

encoders = {"age": lambda age: [age / max_age],
            "sex": lambda gen: [gen],
            "red": lambda red: [red / max_color_red],
            "green": lambda green: [green / max_color_green],
            "blue": lambda blue: [blue / max_color_blue],
            "rgb": lambda rgb: [rgb / max_color_rgb],
            "hemoglobin": lambda hem: [hem / max_hemoglobin],
            "anemia": lambda a_value: [a_value]}
#"sex": lambda gen: {"male": [0], "female": [1]}.get(gen),

def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result


def make_supervised(df):
    raw_input_data = data_frame[input_names]
    raw_output_data = data_frame[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}


def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


supervised = make_supervised(data_frame)
encoded_input = np.array(encode(supervised["inputs"]))
encoded_output = np.array(encode(supervised["outputs"]))
print(encoded_input)
print(encoded_output)

train_x = encoded_input[:30]
train_y = encoded_output[:30]

test_x = encoded_input[30:]
test_y = encoded_output[30:]


model = k.Sequential()
model.add(k.layers.Dense(units=7, activation="relu"))
model.add(k.layers.Dense(units=7, activation="relu"))
model.add(k.layers.Dense(units=7, activation="relu"))
# model.add(k.layers.Dense(units=7, activation="relu"))
model.add(k.layers.Dense(units=1, activation="sigmoid"))

model.compile(loss="mse",optimizer="sgd",metrics=["accuracy"])
fit_result = model.fit(x=train_x,y=train_y,epochs=100,validation_split=0.2)

plt.plot(fit_result.history["loss"], label="Train")
plt.plot(fit_result.history["val_loss"],label="Validation")
plt.legend()
plt.show()

plt.plot(fit_result.history["accuracy"],label="Train")
plt.plot(fit_result.history["val_accuracy"],label="Validation")
plt.legend()
plt.show()

predicted_test = model.predict(test_x)
real_data = data_frame.iloc[30:][input_names+output_names]
real_data["PAnemia"] = predicted_test
print(real_data)