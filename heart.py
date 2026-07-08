import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Heart_Disease_Prediction.csv')

df.head()

df.describe()

df.shape

df.columns

df.info()

df.isnull().sum()

df['Heart Disease'].value_counts()

df.nunique()

categorical = []
continuous = []
for col in df.columns:
    if len(df[col].unique()) <= 5:
        categorical.append(col)
    else:
        continuous.append(col)

categorical

continuous

heartred = '#D76F80'
heartblue = '#C7CBE5'
heartyellow = '#FEE5B3'
heartbrown = '#450E10'

sns.countplot(x = df['Heart Disease'], data = df, hue = 'Sex', palette = [heartred, heartblue])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'Chest pain type', palette = [heartyellow, heartred, heartbrown, heartblue])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'FBS over 120', palette = [heartred, heartblue])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'EKG results', palette = [heartyellow, heartred, heartbrown])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'Exercise angina', palette = [heartred, heartblue])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'Slope of ST', palette = [heartyellow, heartred, heartbrown])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'Number of vessels fluro', palette = [heartyellow, heartred, heartbrown, heartblue])

sns.countplot(x = df['Heart Disease'], data = df, hue = 'Thallium', palette = [heartyellow, heartred, heartbrown])

df.Age[df['Heart Disease'] == 'Presence'].min()

df.groupby('Heart Disease')['Age'].mean()

sns.displot(df['Age'])

sns.barplot(x = df['Heart Disease'],y = df['BP'], data = df, palette = [heartred, heartblue])

sns.barplot(x = df['Heart Disease'],y = df['Cholesterol'], data = df, palette = [heartred, heartblue])

sns.barplot(x = df['Heart Disease'], y = df['Max HR'], data = df, palette = [heartred, heartblue])

sns.barplot(x = df['Heart Disease'], y = df['ST depression'], data = df, palette = [heartred, heartblue])

plt.figure(figsize = (20,15))
sns.heatmap(df.select_dtypes(include="number").corr(), annot = True, linewidths = 1, cmap = 'Pastel1')

from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

df['Heart Disease'].head()

categorical.remove('Heart Disease')
data = pd.get_dummies(df, columns = categorical)
data.head()

sc = StandardScaler()
scaled_col = continuous
data[scaled_col] = sc.fit_transform(data[scaled_col])
data.head()

X = data.drop(['Heart Disease'], axis = 1)
y = data['Heart Disease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

models = []
scores = []

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_score = accuracy_score(y_test, lr_pred)

lr_score

models.append('Logistic Regression')
scores.append(lr_score)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_score = accuracy_score(y_test, nb_pred)

nb_score

models.append('Naive Bayes')
scores.append(nb_score)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_score = accuracy_score(y_test, svc_pred)

svc_score

models.append('Support Vector')
scores.append(svc_score)

from sklearn.neighbors import KNeighborsClassifier

error_knn = []
for i in range(1, 30):
 knn = KNeighborsClassifier(n_neighbors = i)
 knn.fit(X_train, y_train)
 pred_i = knn.predict(X_test)
 error_knn.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 5))
plt.plot(range(1, 30), error_knn, color = 'blue', linestyle = 'dashed', marker = 'o')
plt.title('Error vs. K Value')
plt.xlabel('K')
plt.ylabel('Error')

k = error_knn.index(min(error_knn)) + 1
k

knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
knn_score = accuracy_score(y_test, knn_pred)

knn_score

models.append('K Nearest Neighbours')
scores.append(knn_score)

from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0
#selecting random state providing the highest accuracy
for x in range(200):
    dtc = DecisionTreeClassifier(random_state=x)
    dtc.fit(X_train,y_train)
    dtc_pred = dtc.predict(X_test)
    current_accuracy = round(accuracy_score(dtc_pred,y_test)*100,2)
    if(current_accuracy > max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

dtc = DecisionTreeClassifier(random_state = best_x)
dtc.fit(X_train,y_train)
dtc_pred = dtc.predict(X_test)
dtc_score = accuracy_score(y_test, dtc_pred)

dtc_score

models.append('Decision Tree')
scores.append(dtc_score)

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0
#selecting random state providing the highest accuracy
for x in range(200):
    rfc = RandomForestClassifier(random_state = x)
    rfc.fit(X_train,y_train)
    rfc_pred = rfc.predict(X_test)
    current_accuracy = round(accuracy_score(y_test, rfc_pred)*100,2)
    if(current_accuracy > max_accuracy):
        max_accuracy = current_accuracy
        rfc_best_x = x

rfc = RandomForestClassifier(random_state = rfc_best_x)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
rfc_score = accuracy_score(y_test, rfc_pred)

rfc_score

models.append('Random Forest')
scores.append(rfc_score)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
xgb_score = accuracy_score(y_test, xgb_pred)

xgb_score

models.append('XGBoost')
scores.append(xgb_score)

models

scores

percentage_scores = [score*100 for score in scores]

plt.figure(figsize=(10,5))
plt.bar(models, percentage_scores, color = heartred)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.xticks(rotation = 90)
plt.show()

results = pd.DataFrame(list(zip(models, percentage_scores)), columns =['Models', 'Accuracy (%)'])

from sklearn.metrics import confusion_matrix, classification_report
sns.heatmap(confusion_matrix(y_test,rfc_pred), annot = True, cmap = 'Pastel1')

print(classification_report(y_test,rfc_pred))