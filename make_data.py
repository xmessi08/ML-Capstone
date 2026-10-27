from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True).frame
df["malignant"] = 1 - df.pop("target")  # 1 = злокачественная
df.to_csv("data/raw.csv", index=False)
new = df.sample(5, random_state=1).drop(columns="malignant")
new.to_csv("data/new_patients.csv", index=False)
print("Сохранено:", df.shape, "-> data/raw.csv")
