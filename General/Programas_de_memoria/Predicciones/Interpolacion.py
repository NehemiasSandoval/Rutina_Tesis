import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import  matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
import randomimport 
import tkinter as tk
from tkinter import filedialog

def abrir_ventana():
    ventana = tk.Tk()
    ventana.withdraw()
    path = filedialog.askopenfilename()
    if path:
        print("El path seleccionado es:", path)
    else:
        print("No se seleccionó ningún archivo.")
        path = r"Sharepoint/General/"
    return(path)
path = abrir_ventana()
path =  path + r"Programas de memoria/Data_correlacion"
import os
directoriro = os.listdir(path)
random.shuffle(directoriro)
for h in directoriro:
 os.chdir(path + r"Programas de memoria//Data_correlacion")
 import os
 if str(h) != "Evaluacion_modelos" and str(h) != "Graficos":
  hh = str(h)
  name_1 = (hh.split(sep='_'))[0]
  dens_1 = (hh.split(sep='_'))[1]
  df_1 = pd.read_csv(hh)
  for p in random.shuffle(os.listdir(path)):
     if str(p) != "Evaluacion_modelos" and str(p) != "Graficos":
      pp = str(p)
      name_2 = (pp.split(sep='_'))[0]
      dens_2 = (pp.split(sep='_'))[1]
      lista_interp = []
      path_final = path + r"Programas de memoria/Data_correlacion/Evaluacion_modelos/interpolacion_misma_malla"
      for Nombres_archivos_interpolacion in os.listdir(path_final):
        Malla_gruesa , Malla_fina = (Nombres_archivos_interpolacion.split(sep='-'))[1] , (Nombres_archivos_interpolacion.split(sep='-'))[2]
        dens_malla_fina , dens_malla_gruesa , Nombre_mol = (Malla_fina.split(sep='_')[1]), (Malla_gruesa.split(sep='_')[1]) , (Malla_gruesa.split(sep='_')[0])
        a = [Nombre_mol , dens_malla_gruesa, dens_malla_fina]
        lista_interp.append(a)
      Lista_a_comprobar = [name_2,dens_1 , dens_2]
      print(name_1, Lista_a_comprobar)
      if name_2 == name_1 and dens_1 < dens_2 and Lista_a_comprobar not in lista_interp:
        Checkeo = False
        for u in random.shuffle(os.listdir(path)):
         if str(u) != "Evaluacion_modelos" and str(u) != "Graficos":
          uu = str(u)
          name_3 = (uu.split(sep='_'))[0]
          dens_3 = (uu.split(sep='_'))[1]
          if name_3 == name_2 and dens_3< dens_2 and dens_3 > dens_1:
            Checkeo = True
        if Checkeo:
            df_1 = pd.read_csv(hh)
            df_2 = pd.read_csv(pp)
            frames = [df_1, df_2]
            df = pd.concat(frames , ignore_index=True,axis=0)
            df=(df-df.min())/(df.max()-df.min())
            Puntuación_modelos = np.zeros([15,1])
            depth = 40
            Percent = 90
            TR_size = 0.95
            percentiles = stats.scoreatpercentile(df['E_u'] , Percent)
            X = df.iloc[:,0:11]
            Y = df['E_u']
            Refinamiento_test =np.zeros([len(df['E_u']),1])
            for j in range(len(df['E_u'])):
              if (df['E_u'])[j] >= percentiles:
                Refinamiento_test[j] = True
              else:
                Refinamiento_test[j] = False
            df['Si/No'] = Refinamiento_test
            Y_df = df['Si/No']
            X_train , X_test, Y_train , Y_test = train_test_split(X,Y_df,train_size = TR_size , random_state = 0)
            Tree = DecisionTreeClassifier(max_depth = depth)
            Arbol_Error = Tree.fit(X_train , Y_train)
            Y_pred = Arbol_Error.predict(X)
            adr = DecisionTreeRegressor(max_depth = 5)
            adr.fit(X_train ,Y_train)
            y = adr.predict(X_test)
            Refinamiento_pred_a= np.zeros([len(df['E_u']),1])
            for j in range(len(df['E_u'])):
              X_eval = np.array(df.iloc[j,0:11])
              pred =adr.predict([X_eval])
              if pred >= percentiles:
               Refinamiento_pred_a[j] = True
              else:
               Refinamiento_pred_a[j] = False
            X_train , X_test, Y_train , Y_test = train_test_split(X,df['E_u'],train_size = TR_size , random_state = 0)
            multiple_linear = LinearRegression()
            multiple_linear.fit(X_train, Y_train)
            multiple_linear.score(X_test, Y_test)
            Refinamiento_pred= np.zeros([len(df['E_u']),1])
            for j in range(len(df['E_u'])):
              X_eval = np.array(df.iloc[j,0:11])
              pred =multiple_linear.predict([X_eval])
              if pred >= percentiles:
               Refinamiento_pred[j] = True
              else:
               Refinamiento_pred[j] = False
            degreed = 4
            poly_reg = PolynomialFeatures(degree = degreed)
            X_train_poli = poly_reg.fit_transform(X_train)
            X_test_poli = poly_reg.fit_transform(X_test)
            pr = linear_model.LinearRegression()
            pr.fit(X_train_poli , Y_train)
            Y_pred_pr = pr.predict(X_test_poli)
            Refinamiento_pred= np.zeros([len(df['E_u']),1])
            for j in range(len(df['E_u'])):
              X_eval = (np.array(df.iloc[j,0:11])).reshape(1,-1)
              X_eval = poly_reg.fit_transform(X_eval)
              pred =pr.predict(X_eval)
            RFR = RandomForestRegressor(n_estimators=200 , max_depth = 8)
            RFR.fit(X_train , Y_train)
            Refinamiento_pred= np.zeros([len(df['E_u']),1])
            for j in range(len(df['E_u'])):
              X_eval = np.array(df.iloc[j,0:11]).reshape(1,-1)
              pred = RFR.predict(X_eval)
            df = pd.DataFrame(Puntuación_modelos , columns = [pp])
            df.rename(index={0:'Glob Class Tree',1:'Ref Class Tree',2:'No Ref Class Tree', 3:'Glob Reg Tree',4:'Ref Reg Tree',5:'No Ref Reg Tree', 6:'Glob Lin Reg',7:'Ref Lin Reg',8:'No Ref Lin Reg', 9:'Glob Pol Reg',10:'Ref Pol Reg',11:'No Ref Pol Reg', 12:'Glob RFR',13:'Ref RFR',14:'No Ref RFR'}, inplace=True)
            for u in os.listdir(path):
             if str(u) != "Evaluacion_modelos" and str(u) != "Graficos":
              uu = str(u)
              name_3 = (uu.split(sep='_'))[0]
              dens_3 = (uu.split(sep='_'))[1]
              if name_3 == name_2 and dens_3< dens_2 and dens_3 > dens_1:
                df_3 = pd.read_csv(uu)
                df_3=(df_3-df_3.min())/(df_3.max()-df_3.min())
                percentiles_2 = stats.scoreatpercentile(df_3['E_u'] , Percent)
                Puntuación_modelos_2 = np.zeros([15,1])
                percentiles_2 = stats.scoreatpercentile(df_3['E_u'] , Percent)
                Puntuación_modelos_2 = np.zeros([15,1])
                X_2 = df_3.iloc[:,0:11]
                Y_2 = df_3['E_u']
                Refinamiento_pred_2= np.zeros([len(df_3['E_u']),1])
                for j in range(len(df_3['E_u'])):
                  X_eval = np.array(df_3.iloc[j,0:11])
                  pred =multiple_linear.predict([X_eval])
                  if pred >= percentiles_2:
                    Refinamiento_pred_2[j] = True
                  else:
                    Refinamiento_pred_2[j] = False
                Refinamiento_test_2 =np.zeros([len(df_3['E_u']),1])
                for j in range(len(df_3['E_u'])):
                  if (df_3['E_u'])[j] >= percentiles_2:
                    Refinamiento_test_2[j] = True
                  else:
                    Refinamiento_test_2[j] = False
                Matriz_confusion_lin_2 = confusion_matrix(Refinamiento_test_2, Refinamiento_pred_2)
                Presicion_global_lin_2 = np.sum(Matriz_confusion_lin_2.diagonal())/np.sum(Matriz_confusion_lin_2)
                Presicion_SI_lin_2 = (Matriz_confusion_lin_2[1,1])/sum(Matriz_confusion_lin_2[1,])
                Presicion_NO_lin_2 = (Matriz_confusion_lin_2[0,0])/sum(Matriz_confusion_lin_2[0,])
                Puntuación_modelos_2[6,0] = Presicion_global_lin_2
                Puntuación_modelos_2[7,0] = Presicion_SI_lin_2
                Puntuación_modelos_2[8,0] = Presicion_NO_lin_2
                for j in range(len(df_3['E_u'])):
                  X_eval = (np.array(df_3.iloc[j,0:11])).reshape(1,-1)
                  X_eval = poly_reg.fit_transform(X_eval)
                  pred =pr.predict(X_eval)
                  if pred >= percentiles_2:
                    Refinamiento_pred_2[j] = True
                  else:
                    Refinamiento_pred_2[j] = False
                Matriz_confusion_tree_reg = confusion_matrix(Refinamiento_test_2, Refinamiento_pred_2)
                Presicion_global_tree_reg = np.sum(Matriz_confusion_tree_reg.diagonal())/np.sum(Matriz_confusion_tree_reg)
                Presicion_SI_tree_reg = (Matriz_confusion_tree_reg[1,1])/sum(Matriz_confusion_tree_reg[1,])
                Presicion_NO_tree_reg = (Matriz_confusion_tree_reg[0,0])/sum(Matriz_confusion_tree_reg[0,])
                Puntuación_modelos_2[9,0] = Presicion_global_tree_reg
                Puntuación_modelos_2[10,0] = Presicion_SI_tree_reg
                Puntuación_modelos_2[11,0] = Presicion_NO_tree_reg
                for j in range(len(df_3['E_u'])):
                  X_eval = np.array(df_3.iloc[j,0:11])
                  pred =RFR.predict([X_eval])
                  if pred >= percentiles_2:
                   Refinamiento_pred_2[j] = True
                  else:
                   Refinamiento_pred_2[j] = False
                Matriz_confusion_tree_reg = confusion_matrix(Refinamiento_test_2, Refinamiento_pred_2)
                Presicion_global_tree_reg = np.sum(Matriz_confusion_tree_reg.diagonal())/np.sum(Matriz_confusion_tree_reg)
                Presicion_SI_tree_reg = (Matriz_confusion_tree_reg[1,1])/sum(Matriz_confusion_tree_reg[1,])
                Presicion_NO_tree_reg = (Matriz_confusion_tree_reg[0,0])/sum(Matriz_confusion_tree_reg[0,])
                Puntuación_modelos_2[3,0] = Presicion_global_tree_reg
                Puntuación_modelos_2[4,0] = Presicion_SI_tree_reg
                Puntuación_modelos_2[5,0] = Presicion_NO_tree_reg
                for j in range(len(df_3['E_u'])):
                  X_eval = np.array(df_3.iloc[j,0:11])
                  pred =RFR.predict([X_eval])
                  if pred >= percentiles_2:
                   Refinamiento_pred_2[j] = True
                  else:
                   Refinamiento_pred_2[j] = False
                Matriz_confusion_tree_reg = confusion_matrix(Refinamiento_test_2, Refinamiento_pred_2)
                Presicion_global_tree_reg = np.sum(Matriz_confusion_tree_reg.diagonal())/np.sum(Matriz_confusion_tree_reg)
                Presicion_SI_tree_reg = (Matriz_confusion_tree_reg[1,1])/sum(Matriz_confusion_tree_reg[1,])
                Presicion_NO_tree_reg = (Matriz_confusion_tree_reg[0,0])/sum(Matriz_confusion_tree_reg[0,])
                Puntuación_modelos_2[12,0] = Presicion_global_tree_reg
                Puntuación_modelos_2[13,0] = Presicion_SI_tree_reg
                Puntuación_modelos_2[14,0] = Presicion_NO_tree_reg
                for j in range(len(df_3['E_u'])):
                  X_eval = np.array(df_3.iloc[j,0:11])
                  pred =adr.predict([X_eval])
                  if pred >= percentiles_2:
                    Refinamiento_pred_2[j] = True
                  else:
                    Refinamiento_pred_2[j] = False
                Matriz_confusion_tree_reg = confusion_matrix(Refinamiento_test_2, Refinamiento_pred_2)
                Presicion_global_tree_reg = np.sum(Matriz_confusion_tree_reg.diagonal())/np.sum(Matriz_confusion_tree_reg)
                Presicion_SI_tree_reg = (Matriz_confusion_tree_reg[1,1])/sum(Matriz_confusion_tree_reg[1,])
                Presicion_NO_tree_reg = (Matriz_confusion_tree_reg[0,0])/sum(Matriz_confusion_tree_reg[0,])
                Puntuación_modelos_2[0,0] = Presicion_global_tree_reg
                Puntuación_modelos_2[1,0] = Presicion_SI_tree_reg
                Puntuación_modelos_2[2,0] = Presicion_NO_tree_reg
                df[hh +"-"+ pp+ "---" + uu] = Puntuación_modelos_2
            os.chdir(path + r"Programas de memoria/Data_correlacion/Evaluacion_modelos/interpolacion_misma_malla")
            df.to_csv("Interpolacion"+"-" + hh +"-"+ pp +"-"+ str(TR_size*100) +"%_Ref_percent_" + str(Percent) + "%")
            os.chdir(path + r"Programas de memoria//Data_correlacion")