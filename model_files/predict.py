from catboost import CatBoostClassifier
def predict(model, data):
	return model.predict(data)