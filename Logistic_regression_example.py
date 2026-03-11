from sklearn.linear_model import LogisticRegression
X=[30],[200],[90],[500],[180],[250]
model=LogisticRegression()
model.fit(X,Y)
print(model.predict([[30]]))