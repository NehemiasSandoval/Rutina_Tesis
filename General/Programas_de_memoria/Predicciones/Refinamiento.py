import argparse
import platform
import bempp.api
import numpy as np
import os, time
from quadrature import *
from constants import *
from Grid_Maker import *
from Mesh_refine import *
import trimesh
from bempp.api.operators.potential import laplace as lp
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import pygamer
import threevis
from numba import jit
import sys
import argparse

def Calculo_parametros_geometricos(grid, PQR_File):
  '''
  Librerias dependientes = Bempp.api ; trimesh
  Input
  Malla = Formato .msh
  Output = [Id_Elemento , Radio_de_curvatura , normal_punto_Radio_curvatura , Radio_curv/R_max , Area , Lmax_Lmin ]
  '''
  import numpy as np
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
import platform
from pathos.multiprocessing import ProcessingPool as Pool
import bempp.api
import numpy as np
import os, time
from quadrature import *
from constants import *
from Grid_Maker import *
from Mesh_refine import *
import trimesh
from bempp.api.operators.potential import laplace as lp
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import pygamer
import threevis
from numba import jit


#from preprocess import PARAMS
#from scipy.sparse import diags, bmat, block_diag
#from scipy.sparse.linalg import aslinearoperator

#from random import randint

def base ( name, dens, input_suffix, output_suffix, percentaje, N, N_ref, N_it_smoothing , smooth=True, Mallador='Nanoshaper', refine=True,  Use_GAMer = True,
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
            else:
                print("Loading the built grid.")

                grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density ,  mesh_info.suffix , 'Self')
            t_0 = time.time() - init_time_0
            print('Total time to create and load the mesh {0:.6f} [s]'.format(t_0))
            init_time_solving_out = time.time()
            U, dU , spaces_time_U , operators_time_U , assembly_time_U , GMRES_time_U , UdU_time, it_count_U = U_tot_boundary(grid, mesh_info.q, mesh_info.x_q
                                                        , return_time = True , save_plot=True , tolerance = 1e-5)
            #print (U.coefficients.real)
            #print (dU.coefficients.real)

            total_solving_time = time.time() - init_time_solving_out
            print('Total solving time for U and dU: {0:.4f}'.format(total_solving_time))
            print('Total surface is: {0:4f}'.format(np.sum(grid.volumes)))
            init_time_S_trad = time.time()
            S_trad   = S_trad_calc_R( potential.dirichl_space_u, potential.neumann_space_u , U , dU, mesh_info.x_q )
            t_S_trad = time.time()-init_time_S_trad
            print('Measured time to obtain S_trad : {0:.4f}'.format(t_S_trad))

            init_time_S_Ap = time.time()
            S_Ap , S_Ap_i = delta_G_tent_Pool( grid , U.coefficients.real , dU.coefficients.real , mesh_info.u_space ,
                                           mesh_info.u_order  , N, mesh_info.q, mesh_info.x_q )
            S_Ap_time = time.time() - init_time_S_Ap
            print('Time to calculate S_ap: {0:.2f}'.format(S_Ap_time))
            [S_Ex , S_Ex_i , it_count_phi , N_El_adj , flat_ref_time_adj , spaces_time_adj , operators_time_adj ,
             #(mol_name , grid  , dens , input_suffix , N ,save_energy_plot=False  , test_mode = False , return_times = False)
            matrix_time_adj , GMRES_time_adj , phidphi_time , S_Ex_time]  = S_Exact_in_Adjoint_Mesh_with_N_Ref_Pool(name , grid , dens , input_suffix , N , N_ref, mesh_info.q
                                                  ,mesh_info.x_q, Mallador , save_energy_plot=True , test_mode=True , return_times = True)
            print("Se generaron :" + str(N_El_adj) + "Elementos de la malla adjunta")
            total_phi_time = matrix_time_adj + GMRES_time_adj + phidphi_time
            print('Time to solve the adjoint and dependencies: {0:.2f}'.format(total_phi_time))
            print('Time to calculate S_ex: {0:.2f}'.format(S_Ex_time))


            init_time_E = time.time()
            const_space = bempp.api.function_space(grid,  "DP", 0)
            S_Ap_bempp = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ap_i[:,0])
            S_Ex_bempp    = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ex_i)

            dif =S_Ap_i[:,0]-S_Ex_i
            Parametros_geometricos = Calculo_parametros_geometricos(grid, PQR_File)
            error_max, error_max_area, ratio_max, pot_max, pot = error_test(dif, grid, mesh_info.q, mesh_info.x_q)

            dif_F = bempp.api.GridFunction(const_space, fun=None, coefficients=np.abs(dif) )
            bempp.api.export(( path_for_bempp+ '_{0}{1}_N_ref_{2:d}.msh'.format( dens,
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
            smoothing_time = 0
            mesh_refiner_time = 0
            GAMer_time = 0
            if refine:
                init_time_mesh_refiner = time.time()
                new_face_array , new_vert_array = mesh_refiner(face_array , vert_array , np.abs(dif) , percentaje )
                vert_and_face_arrays_to_text_and_mesh( name , new_vert_array , new_face_array.astype(int)[:] ,output_suffix, dens , Self_build=True)

                grid = Grid_loader( name , dens ,output_suffix,'Self')
                mesh_refiner_time  = time.time()-init_time_mesh_refiner
                if smooth:
                    print("Inicializando rutina smooth")
                    init_time_smoothing = time.time()
                    print("Buscando Archivos en: " + path_for_bempp +str(dens)+output_suffix + '.vert'.format(mesh_info.mol_name))
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
                print("Inicializando rutina GAMer")
                init_time_GAMer = time.time()
                new_face_array, new_vert_array = use_pygamer (new_face_array , aux_vert_array , mesh_info.path ,
                                                                  mesh_info.mol_name+ '_' + str(dens) + output_suffix, N_it_smoothing)

                vert_and_face_arrays_to_text_and_mesh( name , new_vert_array , new_face_array.astype(int)[:] ,
                                                        output_suffix, dens , Self_build=True)

                grid = Grid_loader( name , dens , output_suffix ,'Self')
                GAMer_time = time.time()-init_time_GAMer

            t_ref =time.time()- init_time_ref
            times = np.array([ t_0 , spaces_time_U , operators_time_U , assembly_time_U ,
                              GMRES_time_U , UdU_time, t_S_trad, S_Ap_time, flat_ref_time_adj ,spaces_time_adj ,
                              operators_time_adj , matrix_time_adj , GMRES_time_adj , phidphi_time ,
                              S_Ex_time , E_time , status_time , mesh_refiner_time , smoothing_time  ,
                              GAMer_time ])

            return ( S_trad , S_Ap , S_Ex , N_elements , N_El_adj , it_count_U  , times, dif , Parametros_geometricos )
            #return ( S_trad , S_Ap , S_Ex , N_elements , N_El_adj , total_solving_time , S_trad_time , S_Ap_time ,
            #         S_Ex_time , operators_time_U , assembly_time_U , solving_time_U , it_count_U , t_ref)

        elif estimator == 'E_phi':

            mesh_info.mol_name     = name
            mesh_info.mesh_density = dens
            mesh_info.suffix       = input_suffix
            mesh_info.path         = os.path.join('Molecule' , mesh_info.mol_name)

            mesh_info.q , mesh_info.x_q = run_pqr(mesh_info.mol_name)

            init_time_0 = time.time()
            mesh_info.u_space , mesh_info.u_order     = 'DP' , 0
            mesh_info.phi_space , mesh_info.phi_order = 'P' , 1
            mesh_info.u_s_space , mesh_info.u_s_order = 'P' , 1

            if input_suffix == '-0' and not sphere:
                grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density , mesh_info.suffix , Mallador, GAMer = False)
            else:
                print('Loading previus mesh')
                grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density , mesh_info.suffix, 'Self')
            print('Total time to create and load the mesh {0:.6f} [s]'.format(t_0))
            t_0 = time.time() - init_time_0

            face_array = np.transpose(grid.elements)
            vert_array = np.transpose(grid.vertices)

            init_time = time.time()
            U, dU , spaces_time_U , operators_time_U , assembly_time_U , GMRES_time_U , UdU_time, it_count_U = U_tot_boundary(grid, mesh_info.q, mesh_info.x_q
                                                        , return_time = True , save_plot=True , tolerance = 1e-5)
            total_solving_time = time.time() - init_time
            print('Total solving time for U and dU: {0:.4f}'.format(total_solving_time))

            phi , dphi , adj_grid = phi_with_N_ref(name , grid , face_array , vert_array ,
                            dens , input_suffix , N_ref , mesh_info.q, mesh_info.x_q, return_grid = True)

            U_R , dU_R = U_Reac( U, dU , potential.dirichl_space_u , potential.neumann_space_u, mesh_info.q, mesh_info.x_q)

            init_time = time.time()
            S_trad = S_trad_calc_R( potential.dirichl_space_u, potential.neumann_space_u , U , dU, mesh_info.x_q )
            t_S_trad = time.time()-init_time
            print('Measured time to obtain S_trad : {0:.4f}'.format(t_S_trad))

            adj_face_array = np.transpose(adj_grid.elements)
            adj_vert_array = np.transpose(adj_grid.vertices)

            init_time = time.time()
            S_Ap , S_Ap_i , relation = Aproximated_Sol_Adj_UDP0( U_R , dU_R , phi , dphi , face_array , vert_array ,
                                     adj_face_array , adj_vert_array , 1 , grid , adj_grid , N_ref ,
                                                        return_relation=True)
            S_Ap_time = time.time() - init_time

            [S_Ex , S_Ex_i , it_count_phi , N_El_adj , flat_ref_time_adj , spaces_time_adj , operators_time_adj ,
             #(mol_name , grid  , dens , input_suffix , N ,save_energy_plot=False  , test_mode = False , return_times = False)
            matrix_time_adj , GMRES_time_adj , phidphi_time , S_Ex_time]  = S_Exact_in_Adjoint_Mesh_with_N_Ref_Pool(name , grid , dens , input_suffix , N , N_ref, mesh_info.q
                                                  , mesh_info.x_q,Mallador , save_energy_plot=False , test_mode=True , return_times = True)
            total_phi_time = matrix_time_adj + GMRES_time_adj + phidphi_time
            print('Time to solve the adjoint and dependencies: {0:.2f}'.format(total_phi_time))
            print('Time to calculate S_ex: {0:.2f}'.format(S_Ex_time))

            init_time_E = time.time()
            const_space = bempp.api.function_space(grid,  "DP", 0)
            S_Ap_bempp = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ap_i[:,0])
            S_Ex_bempp    = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ex_i)

            dif =S_Ap_i[:,0]-S_Ex_i
            error_max, error_max_area, ratio_max, pot_max, pot = error_test(dif, grid, mesh_info.q, mesh_info.x_q)
            dif_F = bempp.api.GridFunction(const_space, fun=None, coefficients=np.abs(dif))

            bempp.api.export('Molecule/' + name +'/' + name + '_{0}{1}_N_ref_{2:d}.msh'.format( dens,
                                                                input_suffix , N_ref )
                             , grid_function = dif_F , data_type = 'element')
            E_time = time.time()-init_time_E


            if True: #Marked elements
                init_time_status = time.time()
                face_array = np.transpose(grid.elements) + 1
                status = value_assignor_starter(face_array , np.abs(dif[:,0]) , percentaje)
                const_space = bempp.api.function_space(grid,  "DP", 0)
                Status    = bempp.api.GridFunction(const_space, fun=None, coefficients=status)
                status_time = time.time()-init_time_status
                bempp.api.export('Molecule/' + name +'/' + name + '_{0}{1}_Marked_elements_{2}.msh'.format(
                                                dens, input_suffix , N_ref )
                             , grid_function = Status , data_type = 'element')

            face_array = np.transpose(grid.elements)+1
            vert_array = np.transpose(grid.vertices)

            N_elements = len(face_array)
            smoothing_time = 0
            mesh_refiner_time = 0
            GAMer_time = 0
            if refine:
                init_time_mesh_refiner = time.time()
                new_face_array , new_vert_array = mesh_refiner(face_array , vert_array , np.abs(dif) , percentaje )
                vert_and_face_arrays_to_text_and_mesh( name , new_vert_array , new_face_array.astype(int)[:] ,
                                                        output_suffix, dens , Self_build=True)

                grid = Grid_loader( name , dens , output_suffix ,'Self')
                mesh_refiner_time  = time.time()-init_time_mesh_refiner
                if smooth:
                    init_time_smoothing = time.time()

                    fine_vert_array = np.loadtxt('Molecule/{0}/{0}_40.0-1.vert'.format(mesh_info.mol_name))[:,:3]
                    #fine_vert_array = text_to_list(name , '_40.0-0' , '.vert' , info_type=float)
                    aux_vert_array  = smoothing_vertex( new_vert_array , fine_vert_array )
                    vert_and_face_arrays_to_text_and_mesh( name , aux_vert_array , new_face_array.astype(int)[:] ,
                                                        output_suffix, dens , Self_build=True)
                    smoothing_time      = time.time()-init_time_smoothing
                elif not smooth:
                    aux_vert_array = new_vert_array.copy()


            if Use_GAMer:
                init_time_GAMer = time.time()
                new_face_array, new_vert_array = use_pygamer (new_face_array , aux_vert_array , mesh_info.path ,
                                                                  mesh_info.mol_name+ '_' + str(dens) + output_suffix, N_it_smoothing)
                vert_and_face_arrays_to_text_and_mesh( name , new_vert_array , new_face_array.astype(int)[:] ,
                                                        output_suffix, dens , Self_build=True)

                grid = Grid_loader( name , dens , output_suffix ,'Self')
                GAMer_time = time.time()-init_time_GAMer

            t_ref =time.time()- init_time_ref
            times = np.array([ spaces_time_U , operators_time_U , assembly_time_U ,
                              GMRES_time_U , UdU_time, t_S_trad, S_Ap_time, flat_ref_time_adj ,spaces_time_adj ,
                              operators_time_adj , matrix_time_adj , GMRES_time_adj , phidphi_time ,
                              S_Ex_time , E_time , status_time , mesh_refiner_time , smoothing_time  ,
                              GAMer_time ])

            return ( N_El_adj, S_trad , S_Ap , S_Ex , N_elements , N_El_adj , it_count_U  , times , Parametros_geometricos )


def U_tot_boundary(grid , q, x_q, return_time =False , save_plot=False , tolerance = 1e-8):
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
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s)

    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dual_to_dir_s, k)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, k)
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

def S_Exact_in_Adjoint_Mesh_with_N_Ref_Pool(mol_name , grid  , dens , input_suffix , N , N_ref , q, x_q,Mallador,
                                        save_energy_plot=True  , test_mode = False , return_times = False):

    face_array = np.transpose(grid.elements)
    vert_array = np.transpose(grid.vertices)

    aux_face = face_array.copy()
    aux_vert = vert_array.copy()
    flat_ref_time_init = time.time()
    if N_ref == 0:
        adj_grid = grid
        flat_ref_time = time.time()-flat_ref_time_init
        i=1
    elif N_ref>=1:
        flat_ref_time_init = time.time()

        aux_grid = grid
        i=1
        while i <= N_ref:

            #new_face , new_vertex = mesh_refiner(  aux_face +1 , aux_vert , np.ones((len(aux_face[0:,]))) , 1.5 )

            #aux_face , aux_vert   = new_face.copy() , new_vertex.copy()

            #aux_grid =  aux_grid.refine()
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

    print('The grid was uniformly refined in {0:.2f} seconds'.format(flat_ref_time))

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

    S_Ex , S_Ex_j = Exact_aproach_with_u_s_Teo_Pool( adj_face_array , adj_vert_array ,
                                                    phi.coefficients.real , dphi.coefficients.real , N, q, x_q )
    #S_Ex    , rearange_S_Ex_i  , S_Ex_i , _= Exact_aproach_with_u_s_Teo( adj_face_array , adj_vert_array , phi , dphi , N ,
    #                                             grid_relation = adj_el_pos , return_values_on_Adj_mesh = True)

    N_el_adjoint = len(adj_face_array)


    if test_mode:

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
        const_space = bempp.api.function_space(adj_grid,  "DP", 0)
        S_Ex_BEMPP  = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Ex_j)
        bempp.api.export(Org_path  + "eneergy.msh", grid_function=S_Ex_BEMPP)
        #bempp.api.export('Molecule/' + mol_name +'/' + mol_name + '_{0}{1}_S_Exact_{2}.msh'.format(
                                        #dens, input_suffix , N_ref )
                     #, grid_function = S_Ex_BEMPP , data_type = 'element')


    print('Exact solvation energy {0:.5f} [kcal/kmol]'.format(S_Ex))
    rearange_S_Ex_i = np.sum( np.reshape(S_Ex_j , (-1,4**N_ref) )  , axis = 1)
    S_Ex_time      = time.time()-init_time_S_Ex


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
        print("Generando malla adjunta gruesa")


    elif N_ref>=1:

        for i in range(1,N_ref+1):

            new_face , new_vertex = mesh_refiner(aux_face +1 , aux_vert , np.ones((len(aux_face[0:,]))) , 1.5 )

            vert_and_face_arrays_to_text_and_mesh( mol_name , new_vertex , new_face.astype(int), input_suffix +
                                                  '_adj_'+ str(i), Mallador, dens=dens, Self_build=True)

            aux_face , aux_vert = new_face.copy()- 1 , new_vertex.copy()
        print("Generando malla adjunta")
        adj_grid = Grid_loader( mol_name , dens , input_suffix + '_adj_' + str(N_ref) )

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
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dual_to_dir_s, k)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, k)
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

    mesh    = trimesh.Trimesh(vertices=vert_array , faces=face_array)
    normals = mesh.face_normals
    Areas   = mesh.area_faces

    quadrule = quadratureRule_fine(N)
    X_K , W  = quadrule[0].reshape(-1,3) , quadrule[1]

    def integrate_i(c):
        return S_ex_integrate_face(c , face_array , vert_array  , normals, phi ,
                                   dphi, X_K , W , N, q, x_q)

    Integrates = np.array(list(map( integrate_i , np.arange(len(face_array)) )))
    #Integrates = np.array(Pool().map( integrate_i , np.arange(len(face_array)) ))

    Solv_Exact_i = K * Integrates * ep_m * Areas


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
    face_and_vert_to_off(face_array , vert_array , path , file_name)
    #mesh = pygamer.readOFF('Molecule/sphere_cent/sphere_cent_0-0.off')
    mesh = pygamer.readOFF(os.path.join( path , file_name + '.off' ))
    components, orientable, manifold = mesh.compute_orientation()
    mesh.correctNormals()
    # Set selection of all vertices to True so smooth will operate on them.
    for v in mesh.vertexIDs:
        v.data().selected = True
    # Apply N_it_smoothing iterations of smoothing
    mesh.smooth(max_iter=N_it_smoothing, preserve_ridges=True, verbose=True)
    print(F"The mesh currently has {mesh.nVertices} vertices, \
    {mesh.nEdges} edges, and {mesh.nFaces} faces.")

    pygamer.writeOFF(os.path.join(path, file_name + '.off'), mesh)

    new_off_file = open(os.path.join( path , file_name + '.off' ))
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
def main(path , mol_name , mallador , N_ref , dens):
    Org_path = path + r"Programas_de_memoria/Refinamiento_adaptativo"
    os.chdir(Org_path)
    print("Iniciando calculos")
    import platform
    (S_trad , S_Ap , S_Ex , N_elements , N_El_adj , it_count_U  , times, dif,Parametros_geometricos) = base(mol_name, dens, '-0', '-1', 0.1, 25, N_ref, 6, smooth=True, Mallador=mallador, refine =True, Use_GAMer = True, sphere=False , Estimator = 'E_u', x_q = None , q= None, r = np.nan , fine_vert_array = None)
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame(Parametros_geometricos, columns = ['Area', 'R_c', 'Concavidad', 'E_coul_cent' , "E_coul_vert_1" ,"E_coul_vert_2" , "E_coul_vert_3" , 'L_max/L_min', 'R_c / R_max' , 'E_coul/E_coul_max' , 'D_E_Coul'], dtype = float)
    address = abs(dif)
    df['E_u'] = address
    df.to_csv(path + r"/Programas de memoria/Data_correlacion/" + mol_name + "_" + str(dens) + "_" + str(N_ref) +".csv")
    df.corr()

main("/home/nehemias/Sharepoint/General/" , "100D" , "NanoShaper" , 0 , 0.1)
