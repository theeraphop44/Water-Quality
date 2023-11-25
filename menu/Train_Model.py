import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# model 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
# Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# output 
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.metrics import classification_report, confusion_matrix
#Over Under : Sampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import  RandomOverSampler

# call df and filename

def run_train_page():
    if "df_replaced" in st.session_state:
        df = st.session_state.df_replaced
    elif "df" in st.session_state :
        df = st.session_state["df"]
    
    # Name Title 
    st.markdown('# Train Model Water Queality')

    # Def of select Sampler
    def  Sampler():
        x = df.drop(['Potability'],axis=1)
        y =df.Potability

        if random_sampler is "null" :
            x_sampler = x
            y_sampler = y
        elif random_sampler is "Under Sampler":
            under = RandomUnderSampler(sampling_strategy=1)
            x_sampler, y_sampler = under.fit_resample(x, y)
        elif random_sampler is "OverSampler":
            oversample = RandomOverSampler(sampling_strategy=1)
            x_sampler, y_sampler = oversample.fit_resample(x, y)
        return x_sampler, y_sampler
        
    # Def 
    def predict():
        col1, col2 = st.columns(2)
        #X_train,X_test,y_train,y_test = train_test_split(x_sampler,y_sampler,test_size=0.3,random_state=0)
        X = np.array(x_sampler)
        Y = np.array(y_sampler)

        if Select_Algorithm is "Logistic Regression":
            model = LogisticRegression(penalty='l2', C=10000,solver='sag')
        elif Select_Algorithm is "SVM" :
            model = SVC(kernel = 'poly')
        elif Select_Algorithm is "Random Forest":
            model = RandomForestClassifier(n_jobs=-1,random_state=123)
        elif Select_Algorithm is "DecisionTree":
            model = DecisionTreeClassifier()

        scoring = ['precision', 'recall','f1','roc_auc','accuracy']
        scores = cross_validate(model,X, Y, scoring=scoring,cv=Select_K4,return_train_score=True,return_estimator=True)

        # model.fit(x_train, Y_train)
        # Y_train_hat = model.predict(x_train)
        # Y_test_hat = model.predict(x_test)

        
        # Output Test
        with col1:
            st.write(f'test_accuracy max',max(scores['test_accuracy']))
        
            st.write("\--------------------------------------------------------------")
                        
        #     st.write('-------------------------------------------------------')
        #     st.write('Confusion matrix')
        #     cm1 = confusion_matrix(Y_test, Y_test_hat)
        #     cm1_str = "\n".join(["\t".join(map(str, row)) for row in cm1])
        #     st.text(cm1_str)

        #     st.write('-------------------------------------------------------')
        #     st.write('accuracy score')
            
        #     st.success(f"test data accuracy score : {accuracy_score(Y_test, Y_test_hat)*100:.2f}")

        # # Output Train
        with col2:
            st.write('train_accuracy mean',scores['train_accuracy'].mean())
            st.write('train_accuracy ',scores['train_accuracy'])
            st.write('train_accuracy max',max(scores['train_accuracy']))
        #     st.write('-------------------------------------------------------')
        #     st.write('Train performance')
        #     # st.image(classification_report_train_file_path)
                    
        #     st.write('-------------------------------------------------------')
        #     st.write('Confusion matrix')
        #     cm2 = confusion_matrix(Y_train, Y_train_hat)
        #     cm2_str = "\n".join(["\t".join(map(str, row)) for row in cm2])
        #     st.text(cm2_str)

        #     st.write('-------------------------------------------------------')
        #     st.write('accuracy score')
        #     st.success(f"train data accuracy score : {accuracy_score(Y_train, Y_train_hat)*100:.2f}")    

    def savemodel():
        return 

    # Select Algorithm and Kfold
    col1 , col2 = st.columns(2)
    with col1:
        Select_Algorithm = st.selectbox(
        "Select Algorithm ",
        ("Logistic Regression", "SVM", "Random Forest","DecisionTree","KNeighbors","MLP"),
        index=0,
        placeholder="Select Algorithm...",
        )

        st.write('You selected:', Select_Algorithm)

    with col2:
        Select_K4 = st.number_input(label="K-Fold", min_value=2, max_value=30, step=1, format="%d", value=5,key="k-4")

    # Select Sampler
    random_sampler = st.radio(
        "Select Sampler",
        ["null", "Under Sampler", "OverSampler"],
        index=0,
        help="Select the sampling method"
    )
    st.write("You selected:", random_sampler)

    # Save file png and show countplot
    if random_sampler :
        target_folder = "Picture"

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Bar Sampler
        x_sampler, y_sampler = Sampler()
        pyplot_sampler_bar = sns.countplot(data=df, x=y_sampler)
        fig = pyplot_sampler_bar.get_figure()  

        # Save the countplot image to the target folder
        countplot_file_path = os.path.join(target_folder, "countplot.png")
        fig.savefig(countplot_file_path)

        # Display the image from the target location
        st.write("Count Plot:")
        st.image(countplot_file_path)
        st.markdown('---')

    # Button Train
    col1_btn, col2_btn, col3_btn = st.columns(3)
    
    with col1_btn:
        btn_predict_data = st.button("Button Train Model")
    with col3_btn:
        btn_save_model = st.button("Button Save Model")

    if btn_predict_data:
        predict()
    if btn_save_model:
        savemodel()