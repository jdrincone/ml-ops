from .models import PredictionRequest
from .utils import get_model, transform_to_dataframe

model = get_model()


def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    prediction = correccion_origen(data_to_predict)
    return max(0, prediction)


def correccion_origen(data_to_predict):
    if data_to_predict.sum().sum() == 0:
        prediction = 0
    else:
        prediction = model.predict(data_to_predict)[0]
    return prediction
