import numpy as np
import pandas as pd 
import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB


@st.cache(ttl=86400, allow_output_mutation=True)
def load_data():
    df = pd.read_csv('dataset.csv')
    df = df.dropna(axis=1)
    df.drop("id",axis=1,inplace=True)
    labelencoder_Y = LabelEncoder()
    df.iloc[:,0]= labelencoder_Y.fit_transform(df.iloc[:,0].values)
    
    return df


@st.cache(ttl=86400, allow_output_mutation=True)
def get_model(model):  
    if model == "Logistic Regression":
        return LogisticRegression(random_state = 25)

    if model == "Random Forest Classifier":
        return RandomForestClassifier(random_state=25) 

    if model == "K Nearest Neighbours":
        return KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) 

    if model == "Support Vector Machines":
        return SVC(kernel = 'linear', random_state = 25)

    if model == "Decision Tree Classifier":
        return DecisionTreeClassifier(criterion = 'entropy', random_state = 25)

    if model == "Naive Bayes":
        return GaussianNB() 


@st.cache(ttl=86400, allow_output_mutation=True)
def get_metrics(model, X_train, y_train, X_test, y_test):
    model = model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test,prediction)
    recall = recall_score(y_test,prediction)
    f1 = f1_score(y_test,prediction)

    return accuracy, recall, f1, prediction


def get_visualization(df, X, Y, y_test, prediction):
    cm = confusion_matrix(y_test, prediction)

    # <--Visualization-->
    st.title('Confusion Matrix:')
    fig_cm = plt.figure(figsize=(10,5)) 
    sns.heatmap(cm,annot=True,fmt="d")
    st.pyplot(fig_cm)
    plt.close()
    
    st.title('Data Visualization:')
    
    st.subheader("Heat Map")
    fig_hm = plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=True, fmt='.0%')
    st.pyplot(fig_hm)
    plt.close()

    st.subheader("Swarm Plot")
    sns.set(style="whitegrid", palette="muted")
    data = (X - X.mean()) / (X.std())
    data = pd.concat([Y,data.iloc[:,0:10]],axis=1)
    data = pd.melt(data,id_vars="diagnosis",
                        var_name="features",
                        value_name='value')
    fig_sp = plt.figure(figsize=(20,10))
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
    
    plt.xticks(rotation=90)
    st.pyplot(fig_sp)
    plt.close()

    st.subheader("Strip Plot")
    sns.set(style="whitegrid", palette="muted")
    data = (X - X.mean()) / (X.std())
    data = pd.concat([Y,data.iloc[:,0:10]],axis=1)
    data = pd.melt(data,id_vars="diagnosis",
                        var_name="features",
                        value_name='value')
    fig_spp = plt.figure(figsize=(20,10))

    sns.stripplot(x="features", y="value", hue="diagnosis", data=data)
    plt.xticks(rotation=90)
    st.pyplot(fig_spp)
    plt.close()


st.markdown(f"""<style>.reportview-container .main .block-container{{max-width: {1500}px; }}</style>""", unsafe_allow_html=True)


def main():
    df = load_data()

    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.selectbox("Select the Model", ["Random Forest Classifier", "K Nearest Neighbours", "Decision Tree Classifier", 
    "Logistic Regression", "Support Vector Machines", "Naive Bayes"])

    key_features = ['radius_mean', 'perimeter_mean', 'concavity_mean', 'texture_worst', 'radius_worst']
    X = df[key_features]
    Y = df.diagnosis

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=25)
    model = get_model(selected_model)
    accuracy, recall, f1_score, prediction = get_metrics(model, X_train, y_train, X_test, y_test)
    
    # <---User Inputs--->
    st.sidebar.header("Select Parameters")
    radius_mean = st.sidebar.slider('Radius Mean', 6.0, 30.0, 15.0)
    perimeter_mean = st.sidebar.slider('Perimeter Mean', 40.0, 200.0, 100.0)
    concavity = st.sidebar.slider('Concavity Mean', 0.000, 0.500, 0.200)
    texture = st.sidebar.slider('Texture Worst', 10.0, 50.0, 20.0)
    radius_worst = st.sidebar.slider('Radius Worst', 5.0, 40.0, 10.0)
    st.write('')
    st.sidebar.info(" This App is maintained by Manoj Aryal mailto:me@manoj-aryal.com")
    
    # <---main-page-->
    st.success('**::: Breast Cancer Predictor :::**')
    st.write('')
    st.write('**The Model has a Accuracy of:** ', float("{:.2f}".format(accuracy*100)))
    st.write('**The Model has a Recall of:** ', float("{:.2f}".format(recall*100)))
    st.write('**The Model has a F1 score of:** ', float("{:.2f}".format(f1_score*100)))
    
    new_prediction = model.predict([[radius_mean, perimeter_mean, concavity, texture, radius_worst]])
    st.title('Prediction:')
    st.write('')
    if new_prediction[0] == 0:
        st.write('**The Model has Predicted:** ', new_prediction[0], '**It is Benign.**')
    else:
        st.write('**The Model has Predicted:** ', new_prediction[0], '**It is Malignant.**')
    st.write('')
    st.write('')

    get_visualization(df, X, Y, y_test, prediction)


if __name__ == '__main__':
    main()
