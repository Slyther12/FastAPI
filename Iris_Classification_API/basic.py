from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Train model
iris = load_iris()
x, y = iris.data, iris.target

model = GaussianNB()
model.fit(x, y)

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Iris API is running 🚀"}


@app.post("/predict")
def predict(data: IrisFeatures):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)[0]

    return {
        "prediction": iris.target_names[prediction]
    }
