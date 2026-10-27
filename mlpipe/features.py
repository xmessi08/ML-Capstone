from  sklearn.compose import ColumnTransformer
from sklearn.imoute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandartScaler

def preprocessing(X):
	num = X.select_dtypes("number").columns
	cat = X.columns.defference(num)
	return ColumnTransformer([
		("num", make_pipeline(SimpleImputer(strategy="median"),
					StandartScaler()), num),
		("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
					OneHowEncoder(handle_unknown="ignore")), cat),
