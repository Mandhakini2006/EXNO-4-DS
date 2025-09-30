# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler

      import pandas as pd
      df1 = pd.read_csv("C:\\Users\\admin\\Downloads\\bmi.csv")
      df1 
      
<img width="396" height="493" alt="image" src="https://github.com/user-attachments/assets/21dbaade-215c-4cfa-bbec-d1eb057be51c" />


  
      df2 = df1.copy()


      enc = StandardScaler()


      df2[['new_height', 'new_weight']] = enc.fit_transform(df2[['Height', 'Weight']])


       df2
       
<img width="497" height="417" alt="image" src="https://github.com/user-attachments/assets/5926f10e-192d-439f-8ded-6d5aac801854" />

     df3 = df1.copy()
     enc = MinMaxScaler()
     df3[['new_height', 'new_weight']] = enc.fit_transform(df3[['Height', 'Weight']])
      df3
   
<img width="552" height="452" alt="image" src="https://github.com/user-attachments/assets/23ef1107-69c0-4059-87cd-602b11190b36" />

    df4 = df1.copy()
    enc = MaxAbsScaler()
    df4[['new_height', 'new_weight']] = enc.fit_transform(df4[['Height', 'Weight']])
    df4
    
<img width="531" height="458" alt="image" src="https://github.com/user-attachments/assets/a00c75f0-98c9-46e2-88c8-8bbbde157bbc" />

    df5 = df1.copy()
    enc = Normalizer()
    df5[['new_height', 'new_weight']] = enc.fit_transform(df5[['Height', 'Weight']])
    df5

<img width="580" height="503" alt="image" src="https://github.com/user-attachments/assets/68cb09c4-0507-4670-acba-73c3b2df6271" />


     df6 = df1.copy()
     enc = RobustScaler()
     df6[['new_height', 'new_weight']] = enc.fit_transform(df6[['Height', 'Weight']])
     df6
     
<img width="588" height="502" alt="image" src="https://github.com/user-attachments/assets/85cda9ce-49de-4198-80a9-c42f82bddbfb" />

    df=pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")
    df


<img width="927" height="515" alt="image" src="https://github.com/user-attachments/assets/6e94e018-5832-4069-b142-80805c96253d" />

       from sklearn.preprocessing import LabelEncoder
       df_encoded=df.copy()
       le=LabelEncoder()

       for col in df_encoded.select_dtypes(include="object").columns:
       df_encoded[col]=le.fit_transform(df_encoded[col])
    
       x=df_encoded.drop("SalStat",axis=1)
       y=df_encoded["SalStat"]


       x


<img width="947" height="380" alt="image" src="https://github.com/user-attachments/assets/21b663bd-be3d-4def-a6cb-be6e860a6198" />


       from sklearn.feature_selection import SelectKBest, chi2

       chi2_selector=SelectKBest(chi2,k=5)
       chi2_selector.fit(x,y)

       selected_features_chi2=x.columns[chi2_selector.get_support()]
       print("Selected features (Chi-Square):",list(selected_features_chi2))

       mi_scores=pd.Series(chi2_selector.scores_,index=x.columns)
       print(mi_scores.sort_values(ascending=False))
       

<img width="892" height="277" alt="image" src="https://github.com/user-attachments/assets/3177322d-3a66-4fdc-b60f-52db189a8280" />

          from sklearn.feature_selection import f_classif

          anova_selector=SelectKBest(f_classif,k=5)
          anova_selector.fit(x,y)

          selected_features_anova=x.columns[anova_selector.get_support()]
          print("Selected features (ANOVA F-test):",list(selected_features_anova))

          mi_scores=pd.Series(anova_selector.scores_,index=x.columns)
          print(mi_scores.sort_values(ascending=False))

<img width="913" height="302" alt="image" src="https://github.com/user-attachments/assets/3428f8ea-4206-4331-8f97-9357265ba256" />

       from sklearn.feature_selection import mutual_info_classif
       mi_selector=SelectKBest(mutual_info_classif,k=5)
       mi_selector.fit(x,y)

      selected_features_mi=x.columns[mi_selector.get_support()]
      print("Selected features (Mutual Info):",list(selected_features_mi))
      mi_scores=pd.Series(mi_selector.scores_,index=x.columns)
      print("\nMutual Information Scores:\n",mi_scores.sort_values(ascending=False))


<img width="1006" height="316" alt="image" src="https://github.com/user-attachments/assets/56aaea9c-330f-435e-bc58-78d8dac22e0c" />


      from sklearn.linear_model import LogisticRegression
      from sklearn.feature_selection import RFE 

      model = LogisticRegression(max_iter=100)
      rfe = RFE(model, n_features_to_select=5)
      rfe.fit(x,y)

      selected_features_rfe=x.columns[rfe.support_]
      print("Selected features (RFE):", list(selected_features_rfe))
      

<img width="901" height="1013" alt="494699632-4103bee4-88b5-4006-8fb9-b80ddc69802e" src="https://github.com/user-attachments/assets/315f99e6-d825-4f05-a30a-e4c353cf482f" />



     from sklearn.linear_model import LogisticRegression
     from sklearn.feature_selection import SequentialFeatureSelector

     model = LogisticRegression(max_iter=100)
     rfe = SequentialFeatureSelector(model, n_features_to_select=5)
     rfe.fit(x,y)

     selected_features_rfe=x.columns[rfe.support_]
     print("Selected features (SF):", list(selected_features_rfe))
     

<img width="1093" height="534" alt="494699649-65c6ce51-e4dc-4908-953e-fb71e3612eb0" src="https://github.com/user-attachments/assets/3718763c-3f8d-4a91-97ac-10a6aad471dd" />


    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier()
    rf.fit(x,y)
    importances=pd.Series(rf.feature_importances_,index=x.columns)
    selected_features_rf=importances.sort_values(ascending=False)
    print(importances)
    print("Selected features (RandomForestClassifier):",list(selected_features_rf


<img width="1132" height="477" alt="494699664-ec9bc29c-cea7-4a44-960e-7bef54d872a5" src="https://github.com/user-attachments/assets/5c97349d-925f-4899-9f93-4fc201c8eb52" />

    from sklearn.linear_model import LassoCV
    import numpy as np
    lasso=LassoCV(cv=5).fit(x,y)
    importance=np.abs(lasso.coef_)
    selected_features_lasso=x.columns[importance>0]
    print("Selected features (lasso):",list(selected_features_lasso))


<img width="1182" height="294" alt="494699674-57bad28c-78c4-4d7f-9e39-a4ee3c820b05" src="https://github.com/user-attachments/assets/77cad048-bbed-49cf-a64d-662bc75978e6" />


       import pandas as pd 
       from sklearn.model_selection import train_test_split
       from sklearn.preprocessing import LabelEncoder,StandardScaler 
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
       df=pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")
       le=LabelEncoder()
       df_encoded=df.copy()

       for col in df_encoded.select_dtypes(include="object").columns:
       df_encoded[col]=le.fit_transform(df_encoded[col])

       x=df_encoded.drop("SalStat",axis=1)
       y=df_encoded["SalStat"]

       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
       scaler=StandardScaler()
       x_train=scaler.fit_transform(x_train)
       x_test=scaler.transform(x_test)

       knn=KNeighborsClassifier(n_neighbors=3)
       knn.fit(x_train,y_train)
       y_pred=knn.predict(x_test)

       print("Accuracy:",accuracy_score(y_test,y_pred))
       print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
       print("\nClassification Report:\n",classification_report(y_test,y_pred))

<img width="790" height="355" alt="495496750-f0e607b6-4fe5-4725-bd30-1a90948a814d" src="https://github.com/user-attachments/assets/99daeb23-d830-448b-8c94-ff83ac9e9ed0" />









# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset
