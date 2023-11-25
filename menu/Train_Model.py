import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
from joblib import dump, load
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
        if "save_model" in st.session_state:
            st.session_state.save_model = None

        if "save_scores" in st.session_state:
            st.session_state.save_scores = []
        col1, col2 = st.columns(2)
        #X_train,X_test,y_train,y_test = train_test_split(x_sampler,y_sampler,test_size=0.3,random_state=0)
        X = np.array(x_sampler)
        Y = np.array(y_sampler)

        if Select_Algorithm is "Logistic Regression":
            model = LogisticRegression()
        elif Select_Algorithm is "SVM" :
            model = SVC(kernel = 'linear')
        elif Select_Algorithm is "Random Forest":
            model = RandomForestClassifier(n_jobs=-1,random_state=150)
        elif Select_Algorithm is "DecisionTree":
            model = DecisionTreeClassifier(max_depth=3)
        elif Select_Algorithm is "KNeighbors":
            model = KNeighborsClassifier(n_neighbors=5)
        elif Select_Algorithm is "MLP":
            model = MLPClassifier(solver='adam',alpha=0.01,learning_rate='constant')

        scoring = ['precision', 'recall','f1','roc_auc','accuracy']
        scores = cross_validate(model,X, Y, scoring=scoring,cv=Select_K4,return_train_score=True,return_estimator=True)
        
        # Output Test
        with col1:
            st.success(f'Test Accuracy  : {max(scores["test_accuracy"])*100:.2f}')
            st.markdown('---')
            st.success(f'Test Precision : {max(scores["test_precision"])*100:.2f}')
            st.success(f'Test Recall : {max(scores["test_recall"])*100:.2f}')
            st.success(f'Test F1  : {max(scores["test_f1"])*100:.2f}')
            st.success(f'Test ROC  : {max(scores["test_roc_auc"])*100:.2f}')
            

        # # Output Train
        with col2:
            st.success(f'Train Accuracy  : {max(scores["train_accuracy"])*100:.2f}')
            st.markdown('---')
            st.success(f'Train Precision : {max(scores["train_precision"])*100:.2f}')
            st.success(f'Train Recall : {max(scores["train_recall"])*100:.2f}')
            st.success(f'Train F1  : {max(scores["train_f1"])*100:.2f}')
            st.success(f'Train ROC  : {max(scores["train_roc_auc"])*100:.2f}')

        train_acc = f'Train Accuracy  : {max(scores["train_accuracy"])*100:.2f}'
        test_acc = f'Test Accuracy  : {max(scores["test_accuracy"])*100:.2f}'

        m=max(scores['test_accuracy'])
        a=scores['test_accuracy']
        max_indices = np.where(a == m)[0]
        allmodels = scores['estimator']
        modelsave = allmodels[int(max_indices)]

        if "save_model" not in st.session_state:
            st.session_state.save_model = None

        if "save_scores" not in st.session_state:
            st.session_state.save_scores = []
        
        st.session_state.save_model = modelsave
        st.session_state.save_scores.append((test_acc,train_acc))
        

    def savemodel():
        if "save_model" not in st.session_state:
            st.write("Pls Train Model")
        else:
            st.write(st.session_state.save_model)
            st.write(st.session_state.save_scores[0][0])
            st.write(st.session_state.save_scores[0][0])
            n=user_name
            if n == '':
                st.write("Enter name model")
            else:
                conn = sqlite3.connect("./DB/data.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM pathmodels WHERE name=?", (n,))
                count = cursor.fetchone()[0]

                if count > 0:
                    st.write(f"Model name '{n}' is already in use. Please choose another name.")
                else:
                    partname=f"./Models/"+n+".pkl"
                    cursor.execute('''
                        INSERT INTO pathmodels (name,path,testaccuracy,trainaccuracy) VALUES (?,?,?,?)
                    ''',(n,partname,st.session_state.save_scores[0][0],st.session_state.save_scores[0][1]))
                    conn.commit()
                    conn.close()
                    dump(st.session_state.save_model,partname )
                    st.write(f"Model '{n}' has been saved successfully.")


    # Select Algorithm and Kfold
    col1 , col2 = st.columns(2)
    user_name = col1.text_input("Name model:", "")
    with col1:
        Select_Algorithm = st.selectbox(
        "Select Algorithm ",
        ("Logistic Regression", "SVM", "Random Forest","DecisionTree","KNeighbors","MLP"),
        index=0,
        placeholder="Select Algorithm...",
        )

    with col2:
        Select_K4 = st.number_input(label="K-Fold", min_value=2, max_value=30, step=1, format="%d", value=5,key="k-4")

    # Select Sampler
    random_sampler = st.radio(
        "Select Sampler",
        ["null", "Under Sampler", "OverSampler"],
        index=0,
        help="Select the sampling method"
    )

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
    st.markdown("---")

    if btn_predict_data:
        predict()
    if btn_save_model:
        savemodel()