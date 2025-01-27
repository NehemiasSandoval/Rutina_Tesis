import os
import sys
import time
import argparse
import platform
import multiprocessing
import subprocess
import numpy as np
import trimesh
import pyperclip
import threevis
import queue
import numba
import asyncio
import bempp.api
from pathlib import Path
import os.path
from rcsbsearchapi.search import StructSimilarityQuery
from bempp.api.operators.potential import laplace as lp
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from quadrature import *
from constants import *
from Grid_Maker import *
from Mesh_refine import *
import requests
from bs4 import BeautifulSoup
import resource
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import  matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
import getpass
import random
from pathos.multiprocessing import ProcessingPool as Pool
from numba import jit
import random
import glob
import re
from rcsbsearchapi.search import SequenceQuery
import psutil
from contextlib import redirect_stdout, redirect_stderr

# Configuraciones iniciales
script_path = os.path.abspath(__file__)
General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
path = General_path
Org_path  =os.path.join(General_path , "Programas_de_memoria" , "Refinamiento_adaptativo")
sys.path.append(os.path.join(path , "Programas_de_memoria" , "Predicciones"))
os.environ['PATH'] += os.pathsep + os.path.expanduser('~/.local/bin')
Numero_similudes = 10
def encontrar_mayor_decimal(directorio):
    decimal_mas_alto = float('-inf')  # Inicializar con el menor número posible
    archivos = os.listdir(directorio)  # Listar todos los archivos en el directorio
    patron = r'\d+\.\d+'  # Expresión regular para números con decimales

    for archivo in archivos:
        # Buscar todos los números con decimales en el nombre del archivo
        numeros_encontrados = re.findall(patron, archivo)
        # Convertirlos a float y comparar con el más alto actual
        for numero in numeros_encontrados:
            numero_decimal = float(numero)
            if numero_decimal > decimal_mas_alto:
                decimal_mas_alto = numero_decimal
    print("Buscando el decimal más alto")

    return decimal_mas_alto if decimal_mas_alto != float('-inf') else 0.025

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def Calculo_parametros_geometricos(grid, PQR_File):
  '''
  Librerias dependientes = Bempp.api ; trimesh
  Input
  Malla = Formato .msh
  Output = [Id_Elemento , Radio_de_curvatura , normal_punto_Radio_curvatura , Radio_curv/R_max , Area , Lmax_Lmin ]
  '''
  q , x_q = np.array([]),np.empty((0,3))
  ep_in = 4.
  ep_ex = 80.
  k = 0.125
  molecule_file = open (PQR_File,"r").read().split("\n")
  for line in molecule_file :
    line = line.split()
    if len(line)==0 or line[0]!= "ATOM" : continue
    q = np.append(q,float(line[8] ) )
    x_q = np . vstack((x_q, np.array(line[5:8]).astype(float)))

  #Analisis sobre un elemnto i:
  N_vertex = grid.number_of_vertices
  N_Elements = grid.number_of_elements
  Matriz_vertices_ID = np.ones([N_vertex , 10])*-1
  for i in range(N_vertex):
    Matriz_vertices_ID[i , 0] = i
  for j in range(N_Elements):
    Vertices_ID_elem_j = grid.elements[:, j]
    Vert_1_ID , Vert_2_ID , Vert_3_ID = Vertices_ID_elem_j
    Vertices_vecinos_1 , Vertices_vecinos_2 , Vertices_vecinos_3 = Matriz_vertices_ID[Vert_1_ID , :] , Matriz_vertices_ID[Vert_2_ID , :] , Matriz_vertices_ID[Vert_3_ID , :]
    if Vert_2_ID not in Vertices_vecinos_1:
      for n in range(len(Vertices_vecinos_1)):
        if Vertices_vecinos_1[n] == -1:
            Matriz_vertices_ID[Vert_1_ID , n] = Vert_2_ID
            Vertices_vecinos_1 = Matriz_vertices_ID[Vert_1_ID , :]
            break
    #Update vertex 3 to 1
    if Vert_3_ID not in Vertices_vecinos_1:
      for n in range(len(Vertices_vecinos_1)):
        if Vertices_vecinos_1[n] ==-1:
            Matriz_vertices_ID[Vert_1_ID , n] = Vert_3_ID
            Vertices_vecinos_1 = Matriz_vertices_ID[Vert_1_ID , :]
            break
    #Chekeo vecinos del vertice 2
    #Update vertex 1 to 2
    if Vert_1_ID not in Vertices_vecinos_2:
      for n in range(len(Vertices_vecinos_2)):
        if Vertices_vecinos_2[n] == -1:
            Matriz_vertices_ID[Vert_2_ID , n] = Vert_1_ID
            Vertices_vecinos_2 = Matriz_vertices_ID[Vert_2_ID , :]
            break
    #Update vertex 3 to 2
    if Vert_3_ID not in Vertices_vecinos_2:
      for n in range(len(Vertices_vecinos_2)):
        if Vertices_vecinos_2[n] == -1:
            Matriz_vertices_ID[Vert_2_ID , n] = Vert_3_ID
            Vertices_vecinos_2 = Matriz_vertices_ID[Vert_2_ID , :]
            break
    #Chekeo vecinos del vertice 3
    #Update vertex 1 to 3
    if Vert_1_ID not in Vertices_vecinos_3:
      for n in range(len(Vertices_vecinos_3)):
        if Vertices_vecinos_3[n] == -1:
            Matriz_vertices_ID[Vert_3_ID , n] = Vert_1_ID
            Vertices_vecinos_3 = Matriz_vertices_ID[Vert_3_ID , :]
            break
    #Update vertex 2 to 3
    if Vert_2_ID not in Vertices_vecinos_3:
      for n in range(len(Vertices_vecinos_3)):
        if Vertices_vecinos_2[n] == -1:
            Matriz_vertices_ID[Vert_3_ID , n] = Vert_2_ID
            Vertices_vecinos_3 = Matriz_vertices_ID[Vert_3_ID , :]
            break
  Matriz_radio_curvatura = np.zeros([N_vertex, 5])
  Matriz_Radio_elemento = np.zeros([N_Elements , 11])
  for n in range(N_vertex):
    Matriz_radio_curvatura[n , 0] = n
    Points_ID = Matriz_vertices_ID[n , :]
    Points_ID = Points_ID[Points_ID != 0.000001]
    Points_Coordinates = []
    for i in range(len(Points_ID)):
      Coordinates_point_i = grid.vertices[:,int(Points_ID[i])]
      Points_Coordinates.append(Coordinates_point_i)
    (Centro , Radio , Error)=trimesh.nsphere.fit_nsphere(Points_Coordinates, prior=None)
    Vector_al_centro =  Points_Coordinates[0] - Centro
    Matriz_radio_curvatura[n,2:5] = Vector_al_centro
    Vector_al_centro_unitario = Vector_al_centro/((Vector_al_centro[0]**2 + Vector_al_centro[1]**2 + Vector_al_centro[2]**2)**(1/2))
    Matriz_radio_curvatura[n,1] = Radio
    '''
    Ahora generare una lista que enmarque los elementos con sus respectivos radios de curvatura
    '''
  for i in range(N_Elements):
    #Calculo el area del elemento
    Matriz_Radio_elemento[i,0] = grid.volumes[i]
    #Calculo del radio de curvatura promedio
    Vert_1_ID , Vert_2_ID, Vert_3_ID = grid.elements[:, i]
    R_mean = (Matriz_radio_curvatura[Vert_1_ID,1]+Matriz_radio_curvatura[Vert_2_ID,1]+Matriz_radio_curvatura[Vert_3_ID,1])/3
    Matriz_Radio_elemento[i,1] = R_mean
    #Calculo de concavidad / convexidad
    Dir_R_mean = (Matriz_radio_curvatura[Vert_1_ID,2:5] + Matriz_radio_curvatura[Vert_2_ID,2:5] + Matriz_radio_curvatura[Vert_3_ID,2:5])/3
    Matriz_Radio_elemento[i,2] = np.dot(grid.normals[i,:],Dir_R_mean)
    #Calculo Potencial de Coulomb
    x_cent = grid.centroids[i,:]
    nrm_cent = np.sqrt((x_cent[0]-x_q [:,0 ])**2 + (x_cent[1]-x_q [:,1])**2 + ( x_cent[2]-x_q[:,2])**2)
    nrm_cent_2 = (x_cent[0]-x_q [:,0 ])**2 + (x_cent[1]-x_q [:,1])**2 + ( x_cent[2]-x_q[:,2])**2
    aux_cent = np.sum(q/nrm_cent)
    aux_cent_2 = np.sum(q/nrm_cent_2)
    D_coul_cent = abs((aux_cent_2)/(4*np.pi))
    Matriz_Radio_elemento[i,3] = abs(aux_cent/(4*np.pi))
    Vert_1_Cord , Vert_2_Cord , Vert_3_Cord =  grid.vertices[:, grid.elements[:, i]]
    nrm_1 = np.sqrt(( Vert_1_Cord[0]-x_q [:,0 ])**2 + ( Vert_1_Cord[1]-x_q [:,1])**2 + ( Vert_1_Cord[2]-x_q[:,2])**2)
    aux_1 = np.sum(q/nrm_1)
    Matriz_Radio_elemento[i,4] = abs(aux_1/(4*np.pi))
    nrm_2 = np.sqrt((Vert_2_Cord[0]-x_q [:,0 ])**2 + (Vert_2_Cord[1]-x_q [:,1])**2 + ( Vert_2_Cord[2]-x_q[:,2])**2)
    aux_2 = np.sum(q/nrm_2)
    Matriz_Radio_elemento[i,5] = abs(aux_2/(4*np.pi))
    nrm_3 = np.sqrt((Vert_3_Cord[0]-x_q [:,0 ])**2 + (Vert_3_Cord[1]-x_q [:,1])**2 + ( Vert_3_Cord[2]-x_q[:,2])**2)
    aux_3 = np.sum(q/nrm_3)
    Matriz_Radio_elemento[i,6] = abs(aux_3/(4*np.pi))
    #Calculo L_Max L_Min
    L = [((Vert_1_Cord[0]-Vert_2_Cord[0])**2 + (Vert_1_Cord[1]-Vert_2_Cord[1])**2 + (Vert_1_Cord[2]-Vert_2_Cord[2])**2)**(1/2) , ((Vert_1_Cord[0]-Vert_3_Cord[0])**2 + (Vert_1_Cord[1]-Vert_3_Cord[1])**2 + (Vert_1_Cord[2]-Vert_3_Cord[2])**2)**(1/2) , ((Vert_2_Cord[0]-Vert_3_Cord[0])**2 + (Vert_2_Cord[1]-Vert_3_Cord[1])**2 + (Vert_2_Cord[2]-Vert_3_Cord[2])**2)**(1/2)]
    Matriz_Radio_elemento[i,7] = max(L)/min(L)
    Matriz_Radio_elemento[i,10] =  D_coul_cent


#Calculo de R/R_vecino_Max
  neighbors_indices = grid.element_neighbors[0]
  neighbors_indptr =  grid.element_neighbors[1]
  for i in range(N_Elements):
    Vecinos = neighbors_indices[neighbors_indptr[i] : neighbors_indptr[i +1]]
    Vecinos = Vecinos[Vecinos != i]
    Radios_vecinos = []
    Radios_vecinos_1 = []
    for j in range(np.size(Vecinos)):
      Radios_vecinos.append(Matriz_Radio_elemento[j,1])
      Radios_vecinos_1.append(Matriz_Radio_elemento[j,4])
      Radios_vecinos_1.append(Matriz_Radio_elemento[j,5])
      Radios_vecinos_1.append(Matriz_Radio_elemento[j,6])
    Matriz_Radio_elemento[i,8] = Matriz_Radio_elemento[i,1]/(max(Radios_vecinos))
    Matriz_Radio_elemento[i,9] =  Matriz_Radio_elemento[i,4]/(max(Radios_vecinos_1))


  return(Matriz_Radio_elemento)
os.chdir(Org_path)

def set_memory_limit(memory_limit_percent):
    """
    Establece el límite de memoria para el proceso actual como un porcentaje de la memoria total.
    """
    # Obtener la memoria total del sistema en bytes
    total_memory = psutil.virtual_memory().total
    
    # Calcular el límite en bytes según el porcentaje especificado
    memory_limit_bytes = total_memory * (memory_limit_percent / 100)
    
    # Convertir los límites a enteros
    soft, hard = int(memory_limit_bytes), int(memory_limit_bytes)
    
    # Establecer el límite de memoria
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    print(f"Límite de memoria establecido en {memory_limit_percent}% de la memoria total ({memory_limit_bytes / (1024 * 1024):.2f} MB)")


def log_error(mol_name, dens, log_file_path, type):
    if os.path.exists(log_file_path):
        if type == "TimeoutError":
            texto_buscar = f"{mol_name}_{dens}"
            modificado = False
            with open(log_file_path, 'r') as archivo:
                lineas = archivo.readlines()

            with open(log_file_path, 'w') as archivo:
                for linea in lineas:
                    if texto_buscar in linea:
                        nueva_linea = linea.replace("Time_Exception::False", "Time_Exception:True")
                        archivo.write(nueva_linea)
                        modificado = True
                    else:
                        archivo.write(linea)

            # Si no se encontró la cadena y no se modificó nada, escribir las líneas originales
            if not modificado:
                archivo.writelines(lineas)
        if type == "MemoryError":
            texto_buscar = f"{mol_name}_{dens}"
            modificado = False
            with open(log_file_path, 'r') as archivo:
                lineas = archivo.readlines()

            with open(log_file_path, 'w') as archivo:
                for linea in lineas:
                    if texto_buscar in linea:
                        nueva_linea = linea.replace("Memory_Exception:False", "Memory_Exception:True")
                        archivo.write(nueva_linea)
                        modificado = True
                    else:
                        archivo.write(linea)

            # Si no se encontró la cadena y no se modificó nada, escribir las líneas originales
            if not modificado:
                archivo.writelines(lineas)

def run_with_limits(timeout=10, memory_limit_percent=95):
    """
    Ejecuta una función con límites de tiempo y memoria, y registra errores en un archivo de texto especificado.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Obtenemos los valores de mol_name y dens
            if args:
                mol_name = args[0]
                dens = str(args[1])
                General_path = os.path.abspath(os.path.join(script_path, "../../.."))
                log_file_path = os.path.join(General_path, "Programas_de_memoria", "Refinamiento_adaptativo", "Molecule", mol_name, mol_name + "_History")
            else:
                log_file_path = "errores_log.txt"  # Log por defecto si no hay argumentos
            
            # Obtener el valor de memory_limit_percent
            memory_limit_percent = kwargs.get('memory_limit_percent', 99)

            # Crear una cola para capturar el resultado o errores
            result = multiprocessing.Queue()

            def target_function(*args, **kwargs):
                try:
                    # Establecer el límite de memoria como un porcentaje de la memoria total
                    set_memory_limit(memory_limit_percent)
                    # Ejecutar la función y colocar el resultado en la cola
                    result.put(func(*args, **kwargs))
                except MemoryError:
                    result.put("MemoryError")
                    print("Memory error")
                    log_error(mol_name, dens, log_file_path, "MemoryError")
                except TimeoutError:
                    result.put("TimeoutError")
                    print("Timeout")
                    log_error(mol_name, dens, log_file_path, "TimeoutError")

            # Crear un proceso separado
            process = multiprocessing.Process(target=target_function, args=args, kwargs=kwargs)
            process.start()
            process.join(timeout)  # Esperar hasta el límite de tiempo

            if process.is_alive():
                process.terminate()  # Forzar la terminación del proceso
                process.join()
                log_error(mol_name, dens, log_file_path, "TimeoutError")
                raise TimeoutError(f"La función {func.__name__} excedió el tiempo límite de {timeout} segundos.")

            if not result.empty():
                output = result.get()
                if output == "MemoryError":
                    print("Memory error")
                    raise MemoryError(f"La función {func.__name__} excedió el límite de memoria del {memory_limit_percent}% de la memoria total.")
                elif output == "TimeoutError":
                    print("Timeout")
                    raise TimeoutError(f"La función {func.__name__} excedió el tiempo límite de {timeout} segundos.")
                return output

            return None

        return wrapper
    return decorator
def redirect_output(stdout_path, stderr_path, func, *args, **kwargs):
    """Redirige stdout y stderr a archivos mientras ejecuta una función."""
    with open(stdout_path, 'a') as stdout_file, open(stderr_path, 'a') as stderr_file:  # 'a' para no borrar contenido
        original_stdout_fd = os.dup(1)  # Duplicar descriptor de stdout
        original_stderr_fd = os.dup(2)  # Duplicar descriptor de stderr
        os.dup2(stdout_file.fileno(), 1)  # Redirigir stdout
        os.dup2(stderr_file.fileno(), 2)  # Redirigir stderr
        try:
            return func(*args, **kwargs)  # Llamar la función con argumentos
        finally:
            # Restaurar stdout y stderr originales
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)

# Uso del decorador con límite porcentual de memoria
#@run_with_limits(timeout=10, memory_limit_percent=95)  # 25% de la memoria total
def base ( name, dens, input_suffix, output_suffix, percentaje=0.1, N=25, N_ref = 4, N_it_smoothing=2 , smooth=True, Mallador='Nanoshaper', refine=False,  Use_GAMer = False,
          sphere=False, Estimator = 'E_u', x_q = None, q = None, r = np.nan, fine_vert_array=None):
    '''
    Calculates the solvation energy and refines the mesh.
    Params:
    name  : Name of the molecule. Must have a Molecule/{name} folder with the .pqr if not running sphere cases
    dens  : Mesh density. Set this to 0 if using sphere_cases
    input_suffix  : Mesh to be refined. If this is equal to "-0" the mesh will be build using MSMS
    output_suffix : suffix of the refined mesh.
    percentaje    : Percentaje of elements to be refined, which absolute error contribution is
                    less than percentaje*sum of the total absoulute error
    N             : Number of points used in the Gauss cuadrature. Can be {1,7,13,17,19,25,37,48,52,61,79}
    N_ref         : Number of UNIFORM refinements used to calculate phi.
    smooth        : Smooths the mesh using a 40.0 [el/A2] grid, must be created before with MSMS
    refine        : True if the mesh is going to be refined, False if not
    Mallador      : Can be MSMS or NanoShaper
    Use_GAMer     : True if using GAMer or False if not. Read the README to use this.
    sphere        : True if the mesh is a spherical grid, and False if not
    Estimator     : E_phi or E_u
    x_q           : For the SPHERE case, can be defined in np.array([N_q , 3]) format
    q             : For the SPHERE case, can be defined in np.array([N_q , 1]) format
    r             : For the SPHERE case, this is the radius of the sphere
    N_it_smoothing: Number of iterations used to smoothing from Pygamer.

    This function gives the following output
    S_trad : Solvation energy using bempp potential operators
    S_Ap   : I
    S_Ex   : II
    N_elements : Number of elements of the input_suffix grid
    N_El_adj   : Number of elements of the adjoint grid
    total_solving_time : Time needed to create the operators and solving.
    S_trad_time        : Time needed to calculate S_trad, not counting total_solving_time
    S_Ap_time          : Time needed to calculate I, not counting total_solving_time
    S_Ex_time          : Time needed to calculate II
    operators_time_U   : Time needed to build the associated U operators [Not used]
    assembly_time_U    : Time needed to assembly the blocked operator    [Not used]
    solving_time_U     : Time needed to solve the U system
    it_count_U         : Number of iterations to solve the U system
    '''

    if sphere:
      print(a)
    else:
        if Estimator ==   'E_u':
            General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
            path = General_path
            Org_path  =os.path.join(General_path , "Programas_de_memoria" , "Refinamiento_adaptativo")
            path_for_bempp = os.path.join(Org_path,'Molecule' ,name,name)
            mesh_info.mol_name     = name
            mesh_info.mesh_density = dens
            mesh_info.suffix       = input_suffix
            mesh_info.path         = os.path.join(Org_path,'Molecule' , mesh_info.mol_name)
            mesh_info.q , mesh_info.x_q = run_pqr(mesh_info.mol_name)
            PQR_File = os.path.join(mesh_info.path , name + ".pqr")

            init_time_0 = time.time()
            mesh_info.u_space , mesh_info.u_order     = 'DP' , 0
            mesh_info.phi_space , mesh_info.phi_order = 'P' , 1
            mesh_info.u_s_space , mesh_info.u_s_order = 'P' , 1

            if input_suffix == '-0' and not sphere:
                grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density , mesh_info.suffix , Mallador, GAMer = False)
                if grid == "Abort":
                   return("Error linea 273",0,0,0,0,0,0,0,0)
            else:
                print("Loading the built grid.")

                grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density ,  mesh_info.suffix , 'Self')
                if grid == "Abort":
                   return("Error linea 280",0,0,0,0,0,0,0,0)
            t_0 = time.time() - init_time_0
            init_time_solving_out = time.time()
            # Archivos para redirigir stdout y stderr
            Bempp_stdout_out = os.path.join(
                General_path, 
                "Programas_de_memoria", 
                "Refinamiento_adaptativo", 
                "Molecule", 
                name, 
                name + '_{0}{1}_N_ref_{2:d}'.format(dens, input_suffix, N_ref) + "_BemppST-out_.txt"
            )

            Bempp_stderr_out = os.path.join(
                General_path, 
                "Programas_de_memoria", 
                "Refinamiento_adaptativo", 
                "Molecule", 
                name, 
                name + '_{0}{1}_N_ref_{2:d}'.format(dens, input_suffix, N_ref) + "_BemppERR-out_.txt"
            )

            # Crear los directorios si no existen
            os.makedirs(os.path.dirname(Bempp_stdout_out), exist_ok=True)
            os.makedirs(os.path.dirname(Bempp_stderr_out), exist_ok=True)

            print("Starting BEMPP U_tot_boundary")
            print("This may take much time ....")
            print("More details at: " , Bempp_stdout_out )
            print("And :" ,Bempp_stderr_out )
            U, dU , spaces_time_U , operators_time_U , assembly_time_U , GMRES_time_U , UdU_time, it_count_U = redirect_output(
    Bempp_stdout_out,
    Bempp_stderr_out,
    U_tot_boundary,
    grid,
    mesh_info.q,
    mesh_info.x_q,
    return_time=True,
    save_plot=False,
    tolerance=1e-5
)
            print("Grid for loaded succesfully for " + str(mesh_info.mol_name)+" and dens " + str(mesh_info.mesh_density) )
            total_solving_time = time.time() - init_time_solving_out
            print('Total surface is: {0:4f}'.format(np.sum(grid.volumes)))
            init_time_S_trad = time.time()
            S_trad = redirect_output(
    Bempp_stdout_out,
    Bempp_stderr_out,
    S_trad_calc_R,
    potential.dirichl_space_u,
    potential.neumann_space_u,
    U,
    dU,
    mesh_info.x_q
)
            t_S_trad = time.time()-init_time_S_trad
            print('Measured time to obtain S_trad : {0:.4f}'.format(t_S_trad))
            init_time_S_Ap = time.time()
            S_Ap, S_Ap_i = redirect_output(
    Bempp_stdout_out,
    Bempp_stderr_out,
    delta_G_tent_Pool,
    grid,
    U.coefficients.real,
    dU.coefficients.real,
    mesh_info.u_space,
    mesh_info.u_order,
    N,
    mesh_info.q,
    mesh_info.x_q
)
            S_Ap_time = time.time() - init_time_S_Ap
            print('Time to calculate S_ap: {0:.2f}'.format(S_Ap_time))
            print("Starting S_Exact_in_Adjoint_Mesh_with_N_Ref_Pool")  # Este print no será redirigido
            (S_Ex, S_Ex_i, it_count_phi, N_El_adj, flat_ref_time_adj,
            spaces_time_adj, operators_time_adj, matrix_time_adj,
            GMRES_time_adj, phidphi_time, S_Ex_time) = redirect_output(
                Bempp_stdout_out,
                Bempp_stderr_out,
                S_Exact_in_Adjoint_Mesh_with_N_Ref_Pool,
                name,
                grid,
                dens,
                input_suffix,
                N,
                N_ref,
                mesh_info.q,
                mesh_info.x_q,
                Mallador,
                save_energy_plot=True,
                test_mode=True,
                return_times=True
            )
            print(str(N_El_adj) + " Elements present in the adjoint mesh")
            Parametros_geometricos = Calculo_parametros_geometricos(grid, PQR_File)
            total_phi_time = matrix_time_adj + GMRES_time_adj + phidphi_time
            print('Time to solve the adjoint and dependencies: {0:.2f}'.format(total_phi_time))
            init_time_E = time.time()
            print("Exporting meshes")
            print("Starting the export")
            const_space = bempp.api.function_space(grid,  "DP", 0)
            S_Ap_bempp = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ap_i[:,0])
            S_Ex_bempp    = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ex_i)
            dif =S_Ap_i[:,0]-S_Ex_i
            error_max, error_max_area, ratio_max, pot_max, pot = error_test(dif, grid, mesh_info.q, mesh_info.x_q)
            dif_F = bempp.api.GridFunction(const_space, fun=None, coefficients=np.abs(dif) )
            bempp.api.export(( path_for_bempp+ '_S_Ap_bempp_{0}{1}_N_ref_{2:d}.msh'.format( dens,
                                                                        input_suffix , N_ref )
                    ), grid_function = S_Ap_bempp , data_type = 'element')

            bempp.api.export(( path_for_bempp+ '_S_Ex_bempp_{0}{1}_N_ref_{2:d}.msh'.format( dens,
                                                                        input_suffix , N_ref )
                    ), grid_function = S_Ex_bempp , data_type = 'element')

            bempp.api.export(( path_for_bempp+ '_DIF_F_{0}{1}_N_ref_{2:d}.msh'.format( dens,
                                                                        input_suffix , N_ref )
                    ), grid_function = dif_F , data_type = 'element')
            E_time = time.time()-init_time_E
            init_time_ref = time.time()
            if True: #Marked elements
                init_time_status = time.time()
                face_array = np.transpose(grid.elements) + 1
                status = value_assignor_starter(face_array , np.abs(dif) , percentaje)
                const_space = bempp.api.function_space(grid,  "DP", 0)
                Status    = bempp.api.GridFunction(const_space, fun=None, coefficients=status)
                status_time = time.time()-init_time_status
                bempp.api.export(path_for_bempp + '_{0}{1}_Marked_elements_{2}.msh'.format(
                                                dens, input_suffix , N_ref )
                             , grid_function = Status , data_type = 'element')
            face_array = np.transpose(grid.elements)+1
            vert_array = np.transpose(grid.vertices)
            N_elements = len(face_array)
            print("Total of elements:" +  str(N_elements))
            smoothing_time = 0
            mesh_refiner_time = 0
            GAMer_time = 0
            if refine:
                init_time_mesh_refiner = time.time()
                new_face_array , new_vert_array = mesh_refiner(face_array , vert_array , np.abs(dif) , percentaje )
                vert_and_face_arrays_to_text_and_mesh( name , new_vert_array , new_face_array.astype(int)[:] ,output_suffix, dens , Self_build=True)
                grid = Grid_loader( name , dens ,output_suffix,'Self')
                if grid == "Abort":
                   return("Error linea 349",0,0,0,0,0,0,0,0)
                mesh_refiner_time  = time.time()-init_time_mesh_refiner
                if smooth:
                    print("Smoothing mesh")
                    init_time_smoothing = time.time()
                    molecula = (path_for_bempp + "_"  +str(dens) +str(output_suffix) + '.vert')
                    fine_vert_array = np.loadtxt(molecula.format(mesh_info.mol_name))[:,:3]
                    #fine_vert_array = text_to_list(name , '_40.0-0' , '.vert' , info_type=float)
                    aux_vert_array  = smoothing_vertex(new_vert_array , fine_vert_array, len(vert_array))
                    smoothing_time      = time.time()-init_time_smoothing
                    #vert_and_face_arrays_to_text_and_mesh( name , aux_vert_array , new_face_array.astype(int)[:] ,
                    #                                    output_suffix, dens , Self_build=True)
                elif not smooth:
                    aux_vert_array = new_vert_array.copy()


            if Use_GAMer:
                print("Using pyGAMer for adaptative refinement")
                init_time_GAMer = time.time()
                new_face_array, new_vert_array = use_pygamer (new_face_array , aux_vert_array , mesh_info.path ,
                                                                  mesh_info.mol_name+ '_' + str(dens) + output_suffix, N_it_smoothing)

                vert_and_face_arrays_to_text_and_mesh( name , new_vert_array , new_face_array.astype(int)[:] ,
                                                        output_suffix, dens , Self_build=True)

                grid = Grid_loader( name , dens , output_suffix ,'Self')
                if grid == "Abort":
                   print("Error al crear la malla, cancelando rutina")
                   return("Error linea 377",0,0,0,0,0,0,0,0)
                GAMer_time = time.time()-init_time_GAMer

            t_ref =time.time()- init_time_ref
            times = np.array([ t_0 , spaces_time_U , operators_time_U , assembly_time_U ,
                              GMRES_time_U , UdU_time, t_S_trad, S_Ap_time, flat_ref_time_adj ,spaces_time_adj ,
                              operators_time_adj , matrix_time_adj , GMRES_time_adj , phidphi_time ,
                              S_Ex_time , E_time , status_time , mesh_refiner_time , smoothing_time  ,
                              GAMer_time ])

            return ( S_trad , S_Ap , S_Ex , N_elements , N_El_adj , it_count_U  , times, dif , Parametros_geometricos )

def U_tot_boundary(grid , q, x_q, return_time =False , save_plot=False , tolerance = 1e-5):
    '''
    Returns the total electrostatic potential in the boundary.
    Parameters:
    grid   : Bempp object
    '''
    init_time_spaces = time.time()
    potential.dirichl_space_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)
    potential.neumann_space_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)
    potential.dual_to_dir_s_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)
    spaces_time      = time.time()-init_time_spaces
    U, dU , operators_time , assembly_time , GMRES_time , UdU_time , it_count= U_tot(potential.dirichl_space_u ,
                        potential.neumann_space_u , potential.dual_to_dir_s_u , tolerance, q, x_q)
    if save_plot:

        U_func  = bempp.api.GridFunction(potential.dirichl_space_u , fun=None, coefficients= U.coefficients.real)
        dU_func = bempp.api.GridFunction(potential.neumann_space_u , fun=None, coefficients=dU.coefficients.real)

        #bempp.api.export('Molecule/' + mesh_info.mol_name +'/' + mesh_info.mol_name + '_{0}{1}_U.vtk'.format(
                                      #mesh_info.mesh_density,   mesh_info.suffix )
                     #, grid_function = U_func , data_type = 'element')

        #bempp.api.export('Molecule/' + mesh_info.mol_name +'/' + mesh_info.mol_name + '_{0}{1}_dU.vtk'.format(
                                      #mesh_info.mesh_density,   mesh_info.suffix )
                     #, grid_function = dU_func , data_type = 'element')

    if return_time:
        return U, dU , spaces_time , operators_time , assembly_time , GMRES_time , UdU_time , it_count

    return U , dU

def U_tot(dirichl_space , neumann_space , dual_to_dir_s , tolerance, q, x_q):
    '''
    Computes the Total electrostatic mean potential on the boundary.
    Params:
    dirichl_space
    neumann_space
    dual_to_dir_s
    tolerance     : Tolerance of the solver
    '''

    init_time_operators = time.time()

    identity = sparse.identity(     dirichl_space, dirichl_space, dual_to_dir_s)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s, assembler="fmm")
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, assembler="fmm")

    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dual_to_dir_s, k, assembler="fmm")
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, k, assembler="fmm")
    #@bempp.api.real_callable
    #def q_times_G_L(x, n, domain_index, result):
    #    nrm = np.sqrt((x[0] - x_q[:, 0]) ** 2 + (x[1] - x_q[:, 1]) ** 2 + (x[2] - x_q[:, 2]) ** 2)
    #    aux = np.sum(q / nrm)
    #    result[0] = aux / (4 * np.pi * ep_m)

    @bempp.api.real_callable
    def q_times_G_L(x, n, domain_index, result):
        valor = np.empty (len(x_q))
        for i in range (len(x_q)):
            valor[i] = np.linalg.norm((x-x_q)[i])
        global ep_m
        result[:] = 1. / (4.*np.pi*ep_m)  * np.sum( q  / valor)
    #c = np.loadtxt ('coeficientesvicente')

    @bempp.api.real_callable
    def zero_i(x, n, domain_index, result):
        result[:] = 0

    charged_grid_fun = bempp.api.GridFunction(dirichl_space, fun=q_times_G_L) #coefficients=c
    zero_grid_fun    = bempp.api.GridFunction(neumann_space, fun=zero_i     )
    #print (charged_grid_fun.coefficients.real)
    operators_time = time.time() - init_time_operators

    init_time_matrix = time.time()
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_m/ep_s)*slp_out
    rhs = [charged_grid_fun, zero_grid_fun]

    assembly_time = time.time() - init_time_matrix



    print('GMRES Tolerance = {0}'.format(str(tolerance)))
    init_time_GMRES = time.time()
    sol, info,it_count = bempp.api.linalg.gmres( blocked, rhs, return_iteration_count=True , tol=tolerance ,
                                               restart = 300, use_strong_form=True)
    GMRES_time = time.time()-init_time_GMRES
    print("The linear system for U_tot was solved in {0} iterations".format(it_count))
    init_time_UdU = time.time()
    U , dU = sol
    UdU_time      = time.time()-init_time_UdU


    return U, dU , operators_time , assembly_time , GMRES_time , UdU_time , it_count


def S_trad_calc_R(dirichl_space, neumann_space, U, dU, x_q):

    # Se definen los operadores
    slp_in_O = lp.single_layer(neumann_space, x_q.transpose())
    dlp_in_O = lp.double_layer(dirichl_space, x_q.transpose())

    # Y con la solucion de las fronteras se fabrica el potencial evaluada en la posicion de cada carga
    U_R_O = slp_in_O * dU  -  dlp_in_O * U

    # Donde agregando algunas constantes podemos calcular la energia de solvatacion S

    S     = K * np.sum(mesh_info.q * U_R_O).real
    print("Solvation Energy : {:7.8f} [kcal/mol] ".format(S) )

    return S

def S_Exact_in_Adjoint_Mesh_with_N_Ref_Pool(mol_name , grid  , dens , input_suffix , N , N_ref , q, x_q,Mallador,save_energy_plot=True  , test_mode = False , return_times = False):
    face_array = np.transpose(grid.elements)
    vert_array = np.transpose(grid.vertices)
    aux_face = face_array.copy()
    aux_vert = vert_array.copy()
    flat_ref_time_init = time.time()
    path_for_bempp = os.path.join(Org_path,'Molecule' ,mol_name,mol_name)
    if N_ref == 0:
        adj_grid = grid
        flat_ref_time = time.time()-flat_ref_time_init
        i=1
    elif N_ref>=1:
        flat_ref_time_init = time.time()

        aux_grid = grid
        i=1
        while i <= N_ref:
            mesh = trimesh.Trimesh(vertices=aux_vert , faces=aux_face)
            aux_mesh = mesh.subdivide()
            aux_vert , aux_face = aux_mesh.vertices , aux_mesh.faces

            i+=1

        flat_ref_time = time.time()-flat_ref_time_init

    if N_ref>=1:
        print('The flat refinement was done in {0:.2f} seconds'.format(flat_ref_time))
    saving_refined_mesh_init = time.time()
    vert_and_face_arrays_to_text_and_mesh(mol_name , aux_vert , aux_face+1 ,input_suffix + '_adj_'+ str(i-1), dens, Self_build=True)
    saving_refined_mesh_time = time.time() - saving_refined_mesh_init
    #adj_grid = aux_grid
    adj_grid  = Grid_loader( mol_name , dens , input_suffix + '_adj_' + str(N_ref) , 'Self')
    if grid == "Abort":
                   print("Error al crear la malla adjunta")
    print('The grid was uniformly refined in {0:.2f} seconds'.format(flat_ref_time))
    print("Usando la adj_grid " + mol_name + '_'+str(dens)+input_suffix + '_adj_' + str(N_ref)+'.msh')
    adj_face_array = np.transpose(adj_grid.elements)
    adj_vert_array = np.transpose(adj_grid.vertices)

    #adj_el_pos = check_contained_triangles__(grid , adj_grid)

    #print(len(adj_face_array))
    #print(adj_el_pos)
    init_time_spaces_adj  = time.time()
    dirichl_space_phi = bempp.api.function_space(adj_grid,  mesh_info.phi_space , mesh_info.phi_order)
    neumann_space_phi = bempp.api.function_space(adj_grid,  mesh_info.phi_space , mesh_info.phi_order)
    dual_to_dir_s_phi = bempp.api.function_space(adj_grid,  mesh_info.phi_space , mesh_info.phi_order)
    spaces_time_adj   = time.time()-init_time_spaces_adj

    phi , dphi , operators_time_adj , matrix_time_adj , GMRES_time_adj , phidphi_time , it_count = adjoint_equation(dirichl_space_phi , neumann_space_phi , dual_to_dir_s_phi ,
                                            q, x_q, save_plot = True , suffix = '_'+str(N_ref))
    init_time_S_Ex = time.time()
    print("Iniciando rutina Exact_aproach_with_u_s_Teo_Pool puede tomar un tiempo")
    S_Ex , S_Ex_j = Exact_aproach_with_u_s_Teo_Pool( adj_face_array , adj_vert_array ,
                                                    phi.coefficients.real , dphi.coefficients.real , N, q, x_q )
    #S_Ex    , rearange_S_Ex_i  , S_Ex_i , _= Exact_aproach_with_u_s_Teo( adj_face_array , adj_vert_array , phi , dphi , N ,
    #                                             grid_relation = adj_el_pos , return_values_on_Adj_mesh = True)

    N_el_adjoint = len(adj_face_array)


    if test_mode:
        print("Test mode")

        adj_el_pos = (np.arange(len(adj_face_array))/4).astype(int)

        const_space = bempp.api.function_space(adj_grid,  "DP", 0)
        counter     = bempp.api.GridFunction(const_space, fun=None, coefficients=adj_el_pos )
        #bempp.api.export('Molecule/' + mol_name +'/' + mol_name + '_{0}{1}_Counter_{2}.msh'.format(
                                        #dens, input_suffix , N_ref )
                     #, grid_function = counter , data_type = 'element')

        const_space_u = bempp.api.function_space(grid,  "DP", 0)
        counter_u     = bempp.api.GridFunction(const_space_u, fun=None, coefficients=np.arange(len(face_array)) )
        #bempp.api.export('Molecule/' + mol_name +'/' + mol_name + '_{0}{1}_Counter_original_{2}.msh'.format(
                                        #dens, input_suffix , N_ref )
                     #, grid_function = counter_u , data_type = 'element')

    if save_energy_plot:
        print("Save_energy_plot")
        const_space = bempp.api.function_space(adj_grid,  "DP", 0)
        S_Ex_BEMPP  = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ex_j)
        bempp.api.export(( path_for_bempp+ 'S_Ex_bempp_in_adjoint_{0}{1}_N_ref_{2:d}.msh'.format( dens,
                                                                input_suffix , N_ref )
            ), grid_function = S_Ex_BEMPP , data_type = 'element')
        #bempp.api.export('Molecule/' + mol_name +'/' + mol_name + '_{0}{1}_S_Exact_{2}.msh'.format(
                                        #dens, input_suffix , N_ref )
                     #, grid_function = S_Ex_BEMPP , data_type = 'element')


    print('Exact solvation energy {0:.5f} [kcal/kmol]'.format(S_Ex))
    rearange_S_Ex_i = np.sum( np.reshape(S_Ex_j , (-1,4**N_ref) )  , axis = 1)
    S_Ex_time      = time.time()-init_time_S_Ex
    print("Function done")


    return (S_Ex , rearange_S_Ex_i , it_count , N_el_adjoint , flat_ref_time , spaces_time_adj , operators_time_adj ,
                matrix_time_adj , GMRES_time_adj , phidphi_time , S_Ex_time )

def U_Reac(U, dU , dirichl_space , neumann_space, q, x_q):

    @bempp.api.real_callable
    def u_s_G(x,n,domain_index,result):
        global ep_m
        valor = np.empty (len(x_q))
        for i in range (len(x_q)):
            valor[i] = np.linalg.norm((x-x_q)[i])
        result[:] = 1. / (4.*np.pi*ep_m)  * np.sum(q / valor)

    @bempp.api.real_callable
    def du_s_G(x,n,domain_index,result):
        global ep_m
        valor = np.empty (len(x_q))
        for i in range (len(x_q)):
            valor[i] = np.linalg.norm((x-x_q)[i])
        result[:] = -1./(4.*np.pi*ep_m)  * np.sum( np.dot( x-x_q , n  )  * q / valor**3  )

    U_s  = bempp.api.GridFunction(dirichl_space , fun =  u_s_G)
    dU_s = bempp.api.GridFunction(neumann_space , fun = du_s_G)

    U_R  =  U -  U_s
    dU_R = dU - dU_s

    return U_R , dU_R

def phi_with_N_ref(mol_name , coarse_grid , face_array , vert_array , dens ,
                     input_suffix , N_ref , q, x_q, return_grid = False , calculate_phi = True ):
    '''
    Finds and creates the adjoint mesh using N_ref cycles of UNIFORM refinement.
    mol_name : Molecule/Ion name, only for files saving.
    face_array : array of faces
    vert_array : array of vertices
    dens       : used mesh density
    input_suffix : Normaly related to a number of iterations. If doing for a mesh obtained via MSMS/NanoShaper
                   use "-0".
    return_grid : Boolean
    '''

    aux_face = face_array.copy()
    aux_vert = vert_array.copy().astype(float)


    if N_ref == 0:
        adj_grid = coarse_grid


    elif N_ref>=1:

        for i in range(1,N_ref+1):

            new_face , new_vertex = mesh_refiner(aux_face +1 , aux_vert , np.ones((len(aux_face[0:,]))) , 1.5 )

            vert_and_face_arrays_to_text_and_mesh( mol_name , new_vertex , new_face.astype(int), input_suffix +
                                                  '_adj_'+ str(i), Mallador, dens=dens, Self_build=True)

            aux_face , aux_vert = new_face.copy()- 1 , new_vertex.copy()
        print("Making adjoint mesh")
        adj_grid = Grid_loader( mol_name , dens , input_suffix + '_adj_' + str(N_ref) )
        if grid == "Abort":
                   print("Error al crear la malla adjunta")

    if not calculate_phi:
        phi , dphi = 0. , 0.

        return phi , dphi , adj_grid

    adj_face_array = np.transpose(adj_grid.elements)
    adj_vert_array = np.transpose(adj_grid.vertices)

    #adj_el_pos = check_contained_triangles_alternative_2(coarse_grid , adj_grid , N_ref )

    potential.dirichl_space_phi = bempp.api.function_space(adj_grid,  mesh_info.phi_space ,
                                                           mesh_info.phi_order)
    potential.neumann_space_phi = bempp.api.function_space(adj_grid,  mesh_info.phi_space ,
                                                           mesh_info.phi_order)
    potential.dual_to_dir_s_phi = bempp.api.function_space(adj_grid,  mesh_info.phi_space ,
                                                           mesh_info.phi_order)

    phi , dphi , it_count = adjoint_equation( potential.dirichl_space_phi ,
                        potential.neumann_space_phi , potential.dual_to_dir_s_phi, q, x_q)

    potential.phi , potential.dphi = phi , dphi

    if return_grid:
        return phi , dphi , adj_grid


    return phi , dphi

def adjoint_equation( dirichl_space , neumann_space , dual_to_dir_s , q, x_q, save_plot = True ,suffix = ''):

    global ep_m , ep_s , k

    init_time_operators = time.time()
    identity = sparse.identity(     dirichl_space, dirichl_space, dual_to_dir_s)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s, assembler="fmm")
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, assembler="fmm")
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dual_to_dir_s, k, assembler="fmm")
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, k, assembler="fmm")
    operators_time_adj = time.time() - init_time_operators

    init_time_operators = time.time()
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_m/ep_s)*slp_out

    @bempp.api.real_callable
    def q_times_G_L(x, n, domain_index, result):
        valor = np.empty (len(x_q))
        for i in range (len(x_q)):
            valor[i] = np.linalg.norm((x-x_q)[i])
        global ep_m
        result[:] = 1. / (4.*np.pi*ep_m)  * np.sum( q  / valor)

    @bempp.api.real_callable
    def zero_i(x, n, domain_index, result):
        result[:] = 0

    zero = bempp.api.GridFunction(dirichl_space , fun=zero_i)
    P_GL = bempp.api.GridFunction(dirichl_space, fun=q_times_G_L)
    #print (P_GL.coefficients.real)
    rs_r = [P_GL , zero]
    matrix_time_adj = time.time() - init_time_operators

    init_time_GMRES_adj = time.time()
    sol_r, info,it_count = bempp.api.linalg.gmres( blocked, rs_r , return_iteration_count=True, tol=1e-5,
                                                  restart = 300, use_strong_form=True )
    GMRES_time_adj = time.time()-init_time_GMRES_adj
    print("The linear system for phi was solved in {0} iterations".format(it_count))
    phidphi_time = time.time()
    phi_r , dphi_r = sol_r
    phidphi_time = time.time()-phidphi_time

    if save_plot:

        phi_func  = bempp.api.GridFunction( dirichl_space , fun=None, coefficients= phi_r.coefficients.real)
        dphi_func = bempp.api.GridFunction( neumann_space , fun=None, coefficients=dphi_r.coefficients.real)

        #bempp.api.export('Molecule/' + mesh_info.mol_name +'/' + mesh_info.mol_name + '_{0}{1}_phi_{2}.msh'.format(
                                      #mesh_info.mesh_density,   mesh_info.suffix , suffix )
                     #, grid_function = phi_func )

        #bempp.api.export('Molecule/' + mesh_info.mol_name +'/' + mesh_info.mol_name + '_{0}{1}_dphi_{2}.vtk'.format(
                                      #mesh_info.mesh_density,   mesh_info.suffix , suffix )
                     #, grid_function = dphi_func , data_type = 'node')

    return phi_r , dphi_r  , operators_time_adj , matrix_time_adj , GMRES_time_adj , phidphi_time , it_count

def Exact_aproach_with_u_s_Teo_Pool( face_array , vert_array , phi , dphi , N, q, x_q ):
    print("Exact_aproach_with_u_s_Teo_Pool Starting")
    mesh    = trimesh.Trimesh(vertices=vert_array , faces=face_array)
    normals = mesh.face_normals
    Areas   = mesh.area_faces

    quadrule = quadratureRule_fine(N)
    X_K , W  = quadrule[0].reshape(-1,3) , quadrule[1]

    def integrate_i(c):
        return S_ex_integrate_face(c , face_array , vert_array  , normals, phi ,
                                   dphi, X_K , W , N, q, x_q)
    print("Startign integration")
    Integrates = np.array(list(map( integrate_i , np.arange(len(face_array)) )))
    #Integrates = np.array(Pool().map( integrate_i , np.arange(len(face_array)) ))

    Solv_Exact_i = K * Integrates * ep_m * Areas
    print("Exact_aproach_with_u_s_Teo_Pool Done")
    return np.sum(Solv_Exact_i) , Solv_Exact_i

def Aproximated_Sol_Adj_UDP0(U_R , dU_R , phi , dphi , face_array , vert_array ,
                             face_array_adj , vert_array_adj , N , coarse_grid , adj_grid , N_ref ,
                            return_relation=False):
    '''
    Returns the integral over Gamma for phi and dphi per element.
    U MUST BE IN DP0 IN ORDER TO WORK.
    Params: U_R , dU_R , phi , dphi , face_array , vert_array , face_array_adj , vert_array_adj , N
    face_array , vert_array         : Array of faces and vertices from the coarse mesh
    face_array_adj , vert_array_adj : Array of faces and vertices from the adjoint mesh
    N: number of quadrature points used (Only for phi), as U is in DP0 means is constant per element.
    '''

    phi_array , dphi_array = integral_per_element(phi , dphi , face_array_adj , vert_array_adj ,
                                                  mesh_info.phi_space , mesh_info.phi_order ,
                                                  mesh_info.phi_space , mesh_info.phi_order , N , adj_grid)


    relationship = check_contained_triangles__( coarse_grid , adj_grid )

    rearange_S_Aprox_i = np.zeros((len(face_array),1))

    c_adj=0
    for c in relationship:
        rearange_S_Aprox_i[c] += ep_m *(  dU_R.coefficients.real[c]*phi_array[c_adj] -
                                           U_R.coefficients.real[c]*dphi_array[c_adj]   )
        c_adj+=1

    rearange_S_Aprox_i = K*rearange_S_Aprox_i.copy()

    S_Aprox = np.sum(rearange_S_Aprox_i )#[0]

    print('Aproximated Solvation Energy : {0:10f}'.format(S_Aprox))

    if return_relation==True:
        return S_Aprox , rearange_S_Aprox_i , relationship



    return S_Aprox , rearange_S_Aprox_i

def delta_G_tent_Pool(grid , U , dU , U_space , U_order , N, q, x_q):

    face_array = np.transpose(grid.elements)
    vert_array = np.transpose(grid.vertices)

    if U_space == 'DP' and U_order == 0:

        mesh = trimesh.Trimesh(vertices=vert_array , faces=face_array)
        normals = mesh.face_normals
        Areas = mesh.area_faces

        quadrule = quadratureRule_fine(N)
        X_K , W  = quadrule[0].reshape(-1,3) , quadrule[1]

        #Integral_func = lambda c : Delta_G_tent_int(c , face_array , vert_array , normals , Areas
        #                                            ,U , dU , N , X_K , W)
        def func(c):
            return Delta_G_tent_int(c , face_array ,
                        vert_array , normals , Areas ,U , dU , N , X_K , W, q, x_q)

        Integral      = np.array(Pool().map( func , np.arange(len(face_array)) ))
        #Pool().terminate()
        #Pool().join()


    Solv_Ex_i = K * ep_m * Integral
    #Pool().clear()
    print('Aproximated Solvation energy {0:.6f}'.format(np.sum(Solv_Ex_i)))

    return np.sum(Solv_Ex_i) , np.reshape(Solv_Ex_i , (-1,1))

def Delta_G_tent_int(c , face_array , vert_array , normals , Areas , U , dU , N , X_K , W, q, x_q):

    face = face_array[c]

    v1 , v2 , v3 = vert_array[face[0]] , vert_array[face[1]] , vert_array[face[2]]

    X = evaluation_points_and_weights_new(v1,v2,v3 , N , X_K , W)

    def u_s_Teo( x ):
        return (1. / (4.*np.pi*ep_m) ) * np.sum( q / np.linalg.norm( x - x_q, axis=1 ) )

    def du_s_Teo(x, n):
        return -1./(4.*np.pi*ep_m)  * np.sum( np.dot( x-
                                    x_q , n)  * q / np.linalg.norm( x - x_q, axis=1 )**3 )

    u_c_face_2  = np.array(list(map( lambda x_i : u_s_Teo(x_i)              , X )))
    du_c_face_2 = np.array(list(map( lambda x_i : du_s_Teo(x_i, normals[c]) , X )))

    U_local , dU_local = U[c] , dU[c]

    I2 =  np.sum( U_local  * du_c_face_2 * W )
    I1 =  np.sum( dU_local * u_c_face_2  * W )

    return (I1-I2)* Areas[c]
def S_ex_integrate_face(c , face_array , vert_array , normals, phi , dphi , X_K , W  , N, q, x_q):
    f1 , f2 , f3 = face_array[c]
    v1 , v2 , v3 = vert_array[f1] , vert_array[f2] , vert_array[f3]
    A = matrix_lineal_transform( v1 , v2 , v3 )
    x_i = evaluation_points_and_weights_new(v1,v2,v3 , N , X_K , W)
    phi_a  = np.array([  phi[f1] ,  phi[f2] ,  phi[f3] ])
    dphi_a = np.array([ dphi[f1] , dphi[f2] , dphi[f3] ])
    phi_local  = np.array(list(map( lambda x_g : local_f( x_g , A ,  phi_a  , 1), x_i)))
    dphi_local = np.array(list(map( lambda x_g : local_f( x_g , A , dphi_a  , 1), x_i)))
    def u_s_Teo( x ):
        return (1. / (4.*np.pi*ep_m) ) * np.sum( q / np.linalg.norm( x - x_q, axis=1 ) )

    def du_s_Teo(x, n):
        return -1./(4.*np.pi*ep_m)  * np.sum( np.dot( x-
                                    x_q , n)  * q / np.linalg.norm( x - x_q, axis=1 )**3 )

    u_s_local  = np.array(list(map( lambda x_g : u_s_Teo( x_g )                 , x_i)))
    du_s_local = np.array(list(map( lambda x_g : du_s_Teo( x_g , normals[c] )   , x_i)))

    Integrate = np.sum( (dphi_local * u_s_local  -  phi_local * du_s_local) * W )

    return Integrate

def saved_sphere_distributions(name , r, q):
    '''
    Useful when running spherical grids.
    Inputs
    name  : Name of the distribution. Can be 'sphere_cent', 'sphere_offcent' or 'charge-dipole'.
    r     : Sphere radius
    Returns x_q , q
    '''

    if name == 'sphere_cent':
        x_q = np.array( [[  1.E-12 ,  1.E-12 ,  1.E-12 ]]  )
        q = np.array( [q] )

    if name == 'sphere_offcent':
        x_q = np.array( [[  1.E-12 ,  1.E-12 ,   r/2. ]]  )
        q = np.array( [q] )

    if name == 'charge-dipole':
        x_q = np.array( [[  1.E-12 ,  1.E-12 ,  0.62 ],
                 [  1.E-12 ,  0.62*np.cos(np.pi*1.5 + 5.*np.pi/180.) ,
                                                      0.62*np.sin(np.pi*1.5 + 5.*np.pi/180. ) ] ,
                 [  1.E-12 ,  0.62*np.cos(np.pi*1.5 - 5.*np.pi/180.) ,
                                                      0.62*np.sin(np.pi*1.5 - 5.*np.pi/180. )  ]
                       ] )
        q = np.array( [1. , 1. , -1.])

    return x_q , q

def use_pygamer (face_array, vert_array, path, file_name, N_it_smoothing):
    '''
    Suaviza la malla generando ángulos de mínimo 15 y máximo 165.
    Evita triangulos alargados

    Parametros:
    face_array: archivo con las caras de la malla
    vert_array: archivo con los vértices de la malla
    path: lugar donde se encuentra
    file_name: nombre molecula
    N_it_smooothing: número de iteraciones para el smoothing de pygamer, cuidado, variar esto causa variaciones
    significativas. Generalmente uso 6, la idea es realizar los suficientes para no tener ningún ángulo fuera
    de los límites.

    Retorna:
    face_array y vert_array ya suavizados.

    '''
    # Se crea el archivo .off
    script_path = Path(__file__).resolve()
    two_levels_up = script_path.parents[1]
    Use_gamer_path = two_levels_up / "Sub_rutinas" / "Use_gamer"
    face_and_vert_to_off(face_array , vert_array , path , file_name)
    #mesh = pygamer.readOFF('Molecule/sphere_cent/sphere_cent_0-0.off')
    Comando_Use_Gamer = ["./" + Use_gamer_path ,  os.path.join( path , file_name + '.off' ) ,os.path.join( path , file_name +'_GAMER_' +'.off' ) , N_it_smoothing  ]
    try:
        result = subprocess.run(Comando_Use_Gamer, check=True, text=True, capture_output=True)
        print("Ejecución exitosa.")
    except subprocess.CalledProcessError as e:
        print("Error al ejecutar el comando.")
        print("Código de retorno:", e.returncode)
    new_off_file = open(os.path.join( path , file_name +'_GAMER_' +'.off' ))
    algo = new_off_file.readlines()
    new_off_file.close()

    num_vert = int(algo[1].split()[0])
    num_face = int(algo[1].split()[1])

    vert_array = np.empty((0,3))
    for line in algo[2:num_vert+2]:
        vert_array = np.vstack((vert_array, line.split()))

    face_array = np.empty((0,3))
    for line in algo[num_vert+2:]:
        face_array = np.vstack((face_array, line.split()[1:]))

    vert_array = vert_array.astype(float)
    face_array = face_array.astype(int)+1

    return face_array, vert_array
def main(path , mol_name , mallador , N_ref , dens ,result_queue  , Use_rclone = False):
    script_path = os.path.abspath(__file__)
    General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
    config_path = os.path.join(General_path , "Software" , "config" , "rclone" , "rclone.conf")
    Org_path  = os.path.join(General_path , "Programas_de_memoria" , "Refinamiento_adaptativo")
    os.chdir(Org_path)
    print("Calculating Solvation Energy for " + mol_name)
    Mol_directory  = os.path.join(General_path , "Programas_de_memoria" , "Refinamiento_adaptativo" , "Molecule")
    ruta_historial = os.path.join(Mol_directory , mol_name ,mol_name + "_History")
    try:
      (S_trad , S_Ap , S_Ex , N_elements , N_El_adj , it_count_U  , times, dif,Parametros_geometricos) = base(mol_name, dens, '-0', '-1', 0.1,N = 25, N_ref = N_ref, N_it_smoothing = 2, smooth=True, Mallador=mallador, refine =False, Use_GAMer = False, sphere=False , Estimator = 'E_u', x_q = None , q= None, r = np.nan , fine_vert_array = None)
      if isinstance(S_trad, str):
        print("ERROR AL EJECUTAR RUTINA DE CALCULO DE ENERGÍA")
        Resultado = False
        result_queue.put(Resultado)
        return(None)
      else:
        df = pd.DataFrame(Parametros_geometricos, columns = ['Area', 'R_c', 'Concavidad', 'E_coul_cent' , "E_coul_vert_1" ,"E_coul_vert_2" , "E_coul_vert_3" , 'L_max/L_min', 'R_c / R_max' , 'E_coul/E_coul_max' , 'D_E_Coul'], dtype = float)
        address = abs(dif)
        df['E_u'] = address
        df.to_csv(os.path.join(path,"Programas_de_memoria","Data_correlacion" , mol_name + "_" + str(dens) + "_" + str(N_ref) +".csv"))
        script_path = os.path.abspath(__file__)
        General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
        config_path = os.path.join(General_path , "Software" , "config" , "rclone" , "rclone.conf")
        Corr_Path = os.path.join(General_path , "Programas_de_memoria" , "Data_correlacion")
        comando_actualizar_mallas = [
        "rclone", "--config", config_path, "copy",
        os.path.join(Mol_directory, mol_name),
        "Onedrive:Refinamiento_adaptativo/General/Programas_de_memoria/Refinamiento_adaptativo/Molecule/" + str(mol_name),
        "--include", f"*{dens}*", "--ignore-existing"]
        if Use_rclone == True:
         subprocess.run(comando_actualizar_mallas)
         print("Actualizando data")
         comando_actualizar_data= ["rclone", "--config", config_path, "copy",
          Corr_Path,
          "Onedrive:Refinamiento_adaptativo/General/Programas_de_memoria/Data_correlacion" ,
          "--include", f"*{dens}*" , "--ignore-existing"]
         subprocess.run(comando_actualizar_data)
        print("Se completo la rutina main")
        if os.path.exists(ruta_historial):
          texto_buscar = f"{mol_name}_{dens}"
          with open(ruta_historial, 'r') as archivo:
              lineas = archivo.readlines()

          with open(ruta_historial, 'w') as archivo:
              print("Se abrió la ruta historial en modo w ")
              for linea in lineas:
                  if texto_buscar in linea and "Procces_end:False" in linea:
                      nueva_linea = linea.replace("Procces_end:False", "Procces_end:True")
                      archivo.write(nueva_linea)
                  else:
                      archivo.write(linea)
        print("Se completo la rutina main")
    except:
       if Use_rclone == True:
        comando_actualizar_mallas = [
            "rclone", "--config", config_path, "copy",
            os.path.join(Mol_directory, mol_name),
            "Onedrive:Refinamiento_adaptativo/General/Programas_de_memoria/Refinamiento_adaptativo/Molecule/" + str(mol_name),
            "--include", f"*{dens}*"
        ]
        subprocess.run(comando_actualizar_mallas)
def Limpiar_directorios(carpetas_con_archivo_txt):
    Directorio_similitud = os.path.join(General_path ,"Programas_de_memoria" ,"Data_correlacion","Evaluacion_modelos","Interpolacion_moleculas_similares")
    Data_correlacion = os.path.join(General_path, "Programas_de_memoria" ,"Data_correlacion")
    carpetas_sin_archivos_en_data_correlacion = []
    for carpeta in carpetas_con_archivo_txt:
        encontrado = False
        for root, dirs, files in os.walk(Data_correlacion):
            for file in files:
                if carpeta in file:
                    encontrado = True
                    break
            if encontrado:
                break
        if not encontrado:
            carpetas_sin_archivos_en_data_correlacion.append(carpeta)
    for carpeta in carpetas_sin_archivos_en_data_correlacion:
        print("Removiendo carpeta de archivos con carpeta: " , carpeta )
        carpetas_con_archivo_txt.remove(carpeta)
    os.chdir(Directorio_similitud)
    carpetas = next(os.walk('.'))[1]
    for carpeta in carpetas:
        os.chdir(os.path.join(Directorio_similitud, carpeta))
        archivos_txt = glob.glob('*.txt')
        if len(archivos_txt) == 1:
            carpetas_con_archivo_txt.append(carpeta)
    random.shuffle(carpetas_con_archivo_txt)
    return carpetas_con_archivo_txt
def verificar_y_crear_directorio(nombre_directorio):
    # Verificar si el directorio ya existe
    if not os.path.exists(nombre_directorio):
        # Si no existe, crear el directorio
        os.makedirs(nombre_directorio)
        print(f"making dir '{nombre_directorio}'.")
    else:
        print(f"The directory '{nombre_directorio}' already exist.")
def Out_range():
    texto_extraido = []
    extraer_siguiente_texto = False
    nombre_archivo = "texto_extraido.txt"
    with open(nombre_archivo, "r", encoding="utf-8") as file:
        lineas = file.readlines()
        indice = 0
        while indice < len(lineas):
            if "_entity_poly.pdbx_seq_one_letter_code_can" in lineas[indice]:
                if indice + 1 < len(lineas) and ';' in lineas[indice + 1]:
                    primera_linea_con_coma = lineas[indice + 1].strip()
                    siguiente_linea_con_coma = None
                    for i in range(indice + 2, len(lineas)):
                        if ';' in lineas[i]:
                            siguiente_linea_con_coma = lineas[i].strip()
                            break
                    if siguiente_linea_con_coma:
                        lineas_en_medio = lineas[indice + 2:i]
                        resultado = "\n".join([primera_linea_con_coma, siguiente_linea_con_coma] + lineas_en_medio).replace(";", "").replace("\n", "")
                        print("Chemical composition founded")
                        return(resultado)
                        break
                    else:
                        print("Chemical composition is not detailed in the .cif file from the proteindatabank.")
                else:
                    print("Error while reading chemical sequence from .cif file.")
                break
            indice += 1
        else:
            print("There is not a line that specifies '_entity_poly.pdbx_seq_one_letter_code_can'. inside the .cif molecule file from the protein data bank")
    return()

def Query(Nombre_mol):
        Min_similitud = 0.0001
        Numero_similitudes = 1000
        url = f"https://files.rcsb.org/view/{Nombre_mol}.cif"
        print(f"Querying information from the Protein Data Bank about : {Nombre_mol}")
        response = requests.get(url)
        if response.status_code == 200:        
            # Usar BeautifulSoup para parsear el contenido HTML de la respuesta
            soup = BeautifulSoup(response.text, 'html.parser')

            # Obtener todo el texto de la página
            texto = soup.get_text()
        else:
            print(f"Error al acceder a la página. Código de estado: {response.status_code}")
            return(None)
        with open("texto_extraido.txt", "w", encoding="utf-8") as file:
            file.write(texto)
        tiene_mas_de_25_letras = False
        with open("texto_extraido.txt", "r", encoding="utf-8") as file:
            for line in file:
                if "seq_one_letter_code_can" in line:
                    words = re.split(r'\s+', line.strip())
                    try:
                       second_word = words[1]
                    except Exception as e:
                     second_word = Out_range()
                    if len(second_word) > 25:
                        tiene_mas_de_25_letras = True
                    break
        try:
            structural_similarity_query = StructSimilarityQuery(entry_id=Nombre_mol)
            cantidad_resultados = len(list(structural_similarity_query(results_verbosity="minimal")))
        except:
            cantidad_resultados = 0
            structural_similarity_query = "0"
        print("We find a total of " + str(cantidad_resultados -1) + " Molecular enssambles whit chemical geometrical similarity min(25 chains)")
        if tiene_mas_de_25_letras:
            print("The element is large enought to try a chemical similiraty query")
            chemical_similarity_query = SequenceQuery(second_word, 1, Min_similitud)
            results = structural_similarity_query & chemical_similarity_query
            if Numero_similitudes > len(list(results(results_verbosity="minimal"))):
                Numero_similitudes = len(list(results(results_verbosity="minimal")))
                print("We find a total of " + str(len(list(results(results_verbosity="minimal")))) + " enssambles whit chemical similarity" )
            sorted_results = sorted(list(results(results_verbosity="minimal")), key=lambda x: x['score'], reverse=True)
            top_n_results = sorted_results[:Numero_similitudes]
            return(results)
        else:
            results = structural_similarity_query
            print("The molecule is not big enought to realize a chemical similarity Query min(25 chains)")
        return(results)
def obtener_mejores_puntajes(resultados, tipo_servicio, Numero_similitudes):
    puntajes = []
    for identifier, servicios in resultados.items():
        if tipo_servicio in servicios:
            puntajes.append((identifier, servicios[tipo_servicio]))
    puntajes.sort(key=lambda x: x[1], reverse=True)
    mejores_puntajes = puntajes[1:(Numero_similitudes + 1)]
    return mejores_puntajes
def analizar_similitudes(Nombre_mol , Numero_similudes):
    print("Searching similar molecules for: " , Nombre_mol )
    username = getpass.getuser()
    Directorio_similitud = os.path.join(General_path ,"Programas_de_memoria" ,"Data_correlacion","Evaluacion_modelos","Interpolacion_moleculas_similares")
    verificar_y_crear_directorio(os.path.join(Directorio_similitud , Nombre_mol))
    Txt_salida = os.path.join(Directorio_similitud , Nombre_mol,Nombre_mol + str(Numero_similudes) + ".txt")
    results = Query(Nombre_mol)
    resultados_structure = {}
    resultados_sequence = {}
    resultados = {}
    try: 
     lista = list(results(results_verbosity="verbose"))
     for salida in list(results(results_verbosity="verbose")):
        identifier = salida['identifier']
        for servicio in salida['services']:
            service_type = servicio['service_type']
            norm_score = servicio['nodes'][0]['norm_score']
            if identifier in resultados:
                resultados[identifier][service_type] = norm_score
            else:
                resultados[identifier] = {service_type: norm_score}
     mejores_puntajes_structure = obtener_mejores_puntajes(resultados, 'structure', Numero_similudes)
     with open(Txt_salida, 'w') as archivo:
        archivo.write("Identificador\tPuntaje (Structure)\tPuntaje (Sequence)\n")
        for id_structure, score_structure in mejores_puntajes_structure:
            score_sequence = resultados[id_structure].get('sequence', 'N/A')
            archivo.write(f"{id_structure}\t{score_structure}\t{score_sequence}\n")
     print("The top " + str(Numero_similudes) + " assemblies whit structural similarity to " + Nombre_mol + " are:")
     for identifier, score in mejores_puntajes_structure:
        print(identifier)
     for identifier, score in mejores_puntajes_structure:
        print("Checking pqr's files for " + identifier + " in: " +  "Refinamiento_adaptativo/Molecule/" + identifier)
        Texto_elementos = os.path.join(path ,"Programas_de_memoria","Refinamiento_adaptativo","Texto_moleculas.txt")
        carpeta_base = os.path.join(path ,"Programas_de_memoria","Refinamiento_adaptativo","Molecule")
        elemento = identifier
        ruta_carpeta_elemento = os.path.join(carpeta_base, elemento)

        if os.path.exists(ruta_carpeta_elemento) and os.path.isdir(ruta_carpeta_elemento):
                    comando = [
            "pdb2pqr",
            "--titration-state-method=propka",
            "--with-ph=7",
            "--nodebump",
            "--drop-water",
            "--noopt",
            "--ff=PARSE",
        elemento,
         os.path.join(ruta_carpeta_elemento , elemento +".pqr")]
                    subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
         print(f"The directory for {elemento} did'nt exist in {ruta_carpeta_elemento} , creating a new one")
         os.makedirs(ruta_carpeta_elemento)
         os.chdir(ruta_carpeta_elemento)
         comando = [
            "pdb2pqr",
            "--titration-state-method=propka",
            "--with-ph=7",
            "--nodebump",
            "--drop-water",
            "--noopt",
            "--ff=PARSE",
        elemento,
         os.path.join(ruta_carpeta_elemento , elemento +".pqr")]
         subprocess.run(comando , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("ready")
    except:
        print("Impossible to recognice any similar molecule")

def obtener_texto_archivos_msh(ruta_directorio):
    archivos_msh = [archivo for archivo in os.listdir(ruta_directorio) if archivo.endswith(".msh")]
    textos = []
    for archivo_msh in archivos_msh:
        partes = archivo_msh.split("_")
        if len(partes) > 1:
            texto = partes[1].split("-")[0]
            textos.append(float(texto))  # Convertimos a float
    return textos

def subrutina_refinamiento_adapatativo(dens, nombre_mol, N_ref):
    try:
         start_time = time.time()
         result_queue = queue.Queue()
         main(path, nombre_mol, "NanoShaper", N_ref, dens, result_queue)
         try:
            result = result_queue.get(timeout=10)  # Espera hasta 10 segundos
            if result is False:
                print("main retornó False. Deteniendo el ciclo.")
         except queue.Empty:
            print("Error: No se pudo generar la data de forma satisfactoria")
    except:
         print("No se pudo iniciar la rutina de refinamiento y calculo de error")


def Generar_malla_adaptativa(nombre_mol , N_ref):
        username = getpass.getuser()
        script_path = os.path.abspath(__file__)
        General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
        directorio_principal = os.path.join(General_path ,"Programas_de_memoria","Refinamiento_adaptativo","Molecule")
        print("Searching for .pqr and .msh files of", nombre_mol)
        ruta_refinador = os.path.join(General_path ,"Programas_de_memoria","Refinamiento_adaptativo","Refinamiento.py")
        ruta_directorio = os.path.join(directorio_principal ,nombre_mol )
        ruta_historial = os.path.join(ruta_directorio , nombre_mol + "_History" )
        print("Searching for directory : " + ruta_directorio )
        carpeta_mol  = os.path.join(General_path , "Programas_de_memoria" , "Refinamiento_adaptativo" , "Molecule" ,nombre_mol )
        archivo_historia = carpeta_mol + "/" + nombre_mol + "_History"
        i = 0
        while True:
            i = i +1
            print("Ciclo numero " + str(i) + " de Generar_malla_adaptativa ")
            if os.path.exists(ruta_directorio):
                print("Buscando en " + ruta_directorio)
                archivos = os.listdir(ruta_directorio)
                tiene_pqr = False
                tiene_msh = False
                for archivo in archivos:
                    if archivo.endswith(".pqr"):
                        tiene_pqr = True
                    elif archivo.endswith(".msh"):
                        tiene_msh = True
                if tiene_pqr and not tiene_msh and not os.path.exists(archivo_historia):
                    print("Ruta del archivo historia:", archivo_historia)
                    print("El archivo existe:", os.path.exists(archivo_historia))
                    print(f"We find a .pqr file but not a .msh file in {ruta_directorio}. Ejecutando subrutina...")
                    dens = encontrar_mayor_decimal(ruta_directorio)
                    output_name = nombre_mol + "_" + str(dens) + "-0"
                    nueva_linea = (
    f"{output_name}    NanoShaper_Status_Done:{str(False)}    "
    "Memory_Exception:False    Time_Exception:False    Procces_end:False"
)
                    with open(archivo_historia, 'a') as archivo:
                        archivo.write(nueva_linea + "\n")
                        print("Se creó la primera linea, del archivo history :" , nueva_linea)
                    subrutina_refinamiento_adapatativo(dens, nombre_mol , N_ref)
                elif tiene_pqr and tiene_msh:
                 print("We find both .pqr and .msh file for :"+ nombre_mol)
                if os.path.exists(ruta_historial):
                    resultados = []
                    with open(ruta_historial, 'r') as file:
                     for line in file:
                        line = line.strip()
                        parts = line.split('    ')
                        id_mol = parts[0]
                        dens_molecule = (id_mol.split("-")[0]).split("_")[1]
                        Nombre_mol =    (id_mol.split("-")[0]).split("_")[0]
                        nano_shaper_status_done = parts[1].split(':')[1] == 'True'
                        memory_exception = parts[2].split(':')[1] == 'True'
                        time_exception = parts[3].split(':')[1] == 'True'
                        process_end = parts[4].split(':')[1] == 'True'
                        if process_end:
                         clasificacion = "COMPLETADO_CON_EXITO"
                        elif not process_end and not nano_shaper_status_done:
                         clasificacion = "ERROR_AL_GENERAR_MALLA"
                        elif (time_exception or memory_exception) and nano_shaper_status_done:
                         clasificacion = "SOBRECONSUMO"
                        else:
                         clasificacion = "NO-CLASIFICABLE"
                        resultados.append({
                            "ID Molecule": id_mol,
                            "Mol_Name" : Nombre_mol ,
                            "Mesh_Density": dens_molecule,
                            "NanoShaper_Status_Done": nano_shaper_status_done,
                            "Memory_Exception": memory_exception,
                            "Time_Exception": time_exception,
                            "Process_End": process_end,
                            "Clasificacion": clasificacion,
                        })
                    #for resultado in resultados:
                    #    if resultado["Clasificacion"] == "PENDIENTE":
                    #        print("Se encontro trabajo pendiente")
                    #        subrutina_refinamiento_adapatativo(float(resultado["Mesh_Density"]), resultado["Mol_Name"])
                    if resultados:
                     max_density_result = max(resultados, key=lambda x: x["Mesh_Density"])
                     if max_density_result["Clasificacion"] == "NO-CLASIFICABLE":
                      print(f"Ocurrio un evento no clasificable en : {max_density_result['ID Molecule']} deteniendo proceso ")
                      break
                     if max_density_result["Clasificacion"] == "SOBRECONSUMO":
                      print(f"Exisitió un sobre-consumo de recursos al calcular: {max_density_result['ID Molecule']} deteniendo proceso")
                      break
                     elif max_density_result["Clasificacion"] == "COMPLETADO_CON_EXITO":
                      print(f"Resultado con mayor densidad de malla: {max_density_result['ID Molecule']} se completó con exito, intentado con malla más refinada")
                      subrutina_refinamiento_adapatativo(round(1.3*float(max_density_result["Mesh_Density"]),3), max_density_result["Mol_Name"] , N_ref)
                     elif max_density_result["Clasificacion"] == "ERROR_AL_GENERAR_MALLA":
                      subrutina_refinamiento_adapatativo(round(1.3*float(max_density_result["Mesh_Density"]),3), max_density_result["Mol_Name"] , N_ref)
                      print(f"Resultado con mayor densidad de malla: {max_density_result['ID Molecule']} no pudo generar malla, intentado con malla más refinada")
                      
                     
                    else:
                        print("No se encontro una salida capaz de mostrarnos el error obtenido, buscando en el historial")
                        if os.path.exists(carpeta_mol):
                            print(f"La carpeta '{carpeta_mol}' existe.")
                            if os.path.isfile(archivo_historia):
                                print(f"El archivo '{archivo_historia}' existe.")
                                if os.path.isfile(archivo_historia):
                                  with open(archivo_historia, 'r') as file:
                                      # Leer todas las líneas del archivo
                                      lineas = file.readlines()
                                      print("Buscando lineas en el archivo","Lineas = " , lineas)
                                      ultima_linea_con_texto = None
                                      for linea in (lineas):
                                          print("Ultima linea encontrada :" , lineas)
                                          if linea.strip():  # Verifica si la línea no está vacía
                                              ultima_linea_con_texto = linea.strip()
                                              break

                                if ultima_linea_con_texto:
                                          linea = ultima_linea_con_texto
                                          print(f"La última línea con texto en el archivo es:\n{ultima_linea_con_texto}")
                                          resultado = linea.split()[0].split("_")[1].split("-") if len(linea.split()) > 0 and len(linea.split()[0].split("_")) > 1 else None
                                          print(f"la densidad de la ultima refinación fue '-': {resultado}" if resultado else "No se pudo encontrar la ultima refinacion")
                                          dens = round(1.3*(float(resultado)),3)
                                else:
                                          dens = 0.025
                                          print("El archivo está vacío o no contiene líneas con texto.")
                                          output_name = nombre_mol + "_" + str(dens) + "-0"
                                          nueva_linea = (
    f"{output_name}    NanoShaper_Status_Done:{str(False)}    "
    "Memory_Exception:False    Time_Exception:False    Procces_end:False"
)
                                          with open(archivo_historia, 'a') as archivo:
                                            archivo.write(nueva_linea + "\n")
                                            print("Se creó la primera linea, del archivo history :" , nueva_linea)
                                          subrutina_refinamiento_adapatativo(dens, nombre_mol , N_ref)
                                          with open(archivo_historia, 'r') as file:
                                            # Leer todas las líneas del archivo
                                            lineas = file.readlines()
                                            for linea in lineas:
                                                print("Linea encontrada :" , linea)
                            else:
                                print(f"El archivo '{archivo_historia}' no existe.")
                        else:
                            print(f"La carpeta '{carpeta_mol}' no existe.")
                            dens = 0.025
                            subrutina_refinamiento_adapatativo(dens, nombre_mol , N_ref)
                else:
                    print("No se encontró archivo de ruta al historial, fijando densidad de 0.025")
                    dens = 0.025
                    subrutina_refinamiento_adapatativo(dens, nombre_mol , N_ref)
            else:
             print("Error route: = " + ruta_directorio + " Not exist")
             break
def interpolacion(path=path ,white_lists = False , white_list_predictor = [] ,white_lists_predicted = [] ):
 script_path = os.path.abspath(__file__)
 General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
 path = General_path
 print("Iniciando interpolacion ")
 Carpeta_Data_Correlacion =  os.path.join(General_path ,"Programas_de_memoria","Data_correlacion")
 List_path = os.listdir(Carpeta_Data_Correlacion)
 random.shuffle(List_path)
 for h in List_path:
  os.chdir(Carpeta_Data_Correlacion)
  if str(h) != "Evaluacion_modelos" and str(h) != "Graficos" and str(h) != "texto_extraido.txt":
   hh = str(h)
   name_1 = (hh.split(sep='_'))[0]
   dens_1 = (hh.split(sep='_'))[1]
   df_1 = pd.read_csv(hh)
   if white_lists == True and name_1 not in white_lists_predicted:
    continue
   lista_aux_1  = os.listdir(Carpeta_Data_Correlacion)
   random.shuffle(lista_aux_1)
   for p in lista_aux_1:
     if str(p) != "Evaluacion_modelos" and str(p) != "Graficos":
      pp = str(p)
      print(pp)
      name_2 = (pp.split(sep='_'))[0]
      dens_2 = (pp.split(sep='_'))[1]
      lista_interp = []
      path_final =os.path.join( General_path ,"Programas_de_memoria","Data_correlacion","Evaluacion_modelos","interpolacion_misma_malla")
      for Nombres_archivos_interpolacion in os.listdir(path_final):
        Malla_gruesa , Malla_fina = (Nombres_archivos_interpolacion.split(sep='-'))[1] , (Nombres_archivos_interpolacion.split(sep='-'))[2]
        dens_malla_fina , dens_malla_gruesa , Nombre_mol = (Malla_fina.split(sep='_')[1]), (Malla_gruesa.split(sep='_')[1]) , (Malla_gruesa.split(sep='_')[0])
        a = [Nombre_mol , dens_malla_gruesa, dens_malla_fina]
        lista_interp.append(a)
      Lista_a_comprobar = [name_2,dens_1 , dens_2]
      print(name_1, Lista_a_comprobar)
      if name_2 == name_1 and dens_1 < dens_2 and Lista_a_comprobar not in lista_interp:
        Checkeo = False
        lista_u = os.listdir(Carpeta_Data_Correlacion)
        random.shuffle(lista_u)
        for u in lista_u :
         if str(u) != "Evaluacion_modelos" and str(u) != "Graficos":
          uu = str(u)
          name_3 = (uu.split(sep='_'))[0]
          dens_3 = (uu.split(sep='_'))[1]
          if name_3 == name_2 and dens_3< dens_2 and dens_3 > dens_1:
            Checkeo = True
            if white_lists == True and name_3 not in white_list_predictor:
             Checkeo = False
        if Checkeo:
            print("Starting the MachineLearning prediction whit data from " + name_3 + " y " + name_1)
            print("Reading data from " + hh)
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
            if white_lists == False:
                os.chdir(os.path.join(path ,"Programas_de_memoria","Data_correlacion","Evaluacion_modelos","interpolacion_misma_malla"))
                df.to_csv("Interpolacion"+"-" + hh +"-"+ pp +"-"+ str(TR_size*100) +"%_Ref_percent_" + str(Percent) + "%")
                os.chdir(os.path.join(path ,"Programas_de_memoria","Data_correlacion"))
            else:
                Directorio_similitud = os.path.join(General_path ,"Programas_de_memoria" ,"Data_correlacion","Evaluacion_modelos","Interpolacion_moleculas_similares")
                verificar_y_crear_directorio(os.path.join(Directorio_similitud , Nombre_mol))
                os.chdir(os.path.join(Directorio_similitud , Nombre_mol))
                df.to_csv("Interpolacion"+"-" + hh +"-"+ pp +"-"+ str(TR_size*100) +"%_Ref_percent_" + str(Percent) + "%")
                os.chdir(os.path.join(path ,"Programas_de_memoria","Data_correlacion"))
def rutina_completa(Usuario , Numero_similudes , N_ref):
  print(f"Iniciando programa")
  carpetas_con_archivo_txt = []
  Directorio_Objetivos = os.path.join(General_path ,"Programas_de_memoria" ,"Programas_analisis","Moleculas_Objetivo_" + Usuario + ".txt")
  Directorio_similitud = os.path.join(General_path ,"Programas_de_memoria" ,"Data_correlacion","Evaluacion_modelos","Interpolacion_moleculas_similares")
  with open(Directorio_Objetivos, 'r') as archivo:
    Objetivos = [linea.strip() for linea in archivo]
  for objetivo in Objetivos:
    analizar_similitudes(objetivo , Numero_similudes)
  os.chdir(Directorio_similitud)
  carpetas = next(os.walk('.'))[1]
  random.shuffle(carpetas)
  for carpeta in carpetas:
      print("Buscando datos de similitud en Interpolacion_moleculas_similares/" + carpeta )
      os.chdir(os.path.join(Directorio_similitud, carpeta))
      archivos_txt = glob.glob('*.txt')
      carpetas_con_archivo_txt.append(carpeta)
  carpetas_con_archivo_txt = Limpiar_directorios(carpetas_con_archivo_txt)
  carpetas_con_archivo_txt = list(set(carpetas_con_archivo_txt) & set(Objetivos))
  random.shuffle(carpetas_con_archivo_txt)
  print("carpetas_con_archivo_txt: " , carpetas_con_archivo_txt)
  for molecules in carpetas_con_archivo_txt:
      Nombre_mol = molecules
      analizar_similitudes(Nombre_mol , Numero_similudes)
      Directorio_similitud = os.path.join(General_path ,"Programas_de_memoria" ,"Data_correlacion","Evaluacion_modelos","Interpolacion_moleculas_similares")
      verificar_y_crear_directorio(os.path.join(Directorio_similitud , Nombre_mol))
      Txt_salida = os.path.join(Directorio_similitud , Nombre_mol, Nombre_mol + str(Numero_similudes) + ".txt")
      Moleculas_similares = []
      try:
       with open(Txt_salida, 'r') as file:
          lines = file.readlines()
          for line in lines[1:]:
              Moleculas = line.split()[0]
              if Moleculas:
                Moleculas_similares.append(Moleculas)
       random.shuffle(Moleculas_similares)
      except:
       print("Error al buscar similitudes")
       continue
      for molecula in Moleculas_similares:
               MEMORY_LIMIT = 8000 * 1024 * 1024
               pid = os.getpid()
               #molecula = "arg"
               Generar_malla_adaptativa(molecula , N_ref)
               print("Succesfully mesh generated for: " +Moleculas )
      try:
        interpolacion(white_lists=True , white_list_predictor =[Nombre_mol], white_lists_predicted = Moleculas_similares)
      except:
        print("Error al interpolar")
        
if __name__ == '__main__':
    # Verifica que el usuario ha pasado un argumento
    if len(sys.argv) != 2:
        print("Uso: python ~/Sharepoint/General/Programas_de_memoria/Predicciones/Rutina_completa.py <usuario>")
        sys.exit(1)
    
    # Toma el argumento de la línea de comandos
    usuario = sys.argv[1]

    # Define el archivo de salida
    output_path = os.path.expanduser("~/salida_rutina_completa.txt")
    with open(output_path, 'w') as f:
        tee = Tee(sys.stdout, f)
        sys.stdout = tee

        # Ejecuta la rutina con el usuario proporcionado
        rutina_completa(str(usuario), 4, 4)
        sys.stdout = sys.__stdout__
