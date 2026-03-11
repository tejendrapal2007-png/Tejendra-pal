from sklearn.preprocessing import OneHotEncoder

  color = [["laal"],["hara"],["neela"]]

  encoder=OneHotEncoder()

  result=encoder.fit_transform(color)

  print(result.toarray())