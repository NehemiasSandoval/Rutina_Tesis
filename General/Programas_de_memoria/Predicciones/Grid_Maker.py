import bempp.api, numpy as np, time, os, matplotlib.pyplot as plt
from math import pi
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from matplotlib import pylab as plt
from numba import jit
import trimesh
import numpy as np
import subprocess
import os
import bempp.api
import shutil
import getpass
from subprocess import Popen, PIPE
import re
import platform
from constants import *
import time
script_path = os.path.abspath(__file__)
General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
Mol_directories = os.path.join(General_path , "Programas_de_memoria" , "Refinamiento_adaptativo" , "Molecule")
def corregir_ruta(ruta_wsl):
    if platform.system() == "Linux":
        return(ruta_wsl)
    else:
        # Buscar si hay coincidencia con el patrón '/mnt/' seguido de cualquier cosa
        match = re.match(r'^/mnt/([a-z])/(.*)', ruta_wsl)
        if match:
            # Si hay coincidencia, obtener la letra de la unidad y el resto de la ruta
            unidad = match.group(1)
            resto_ruta = match.group(2)
            # Construir la ruta de Windows usando la letra de la unidad y reemplazar barras inclinadas
            ruta_windows = unidad.upper() + ":\\" + resto_ruta.replace('/', '\\')
            return ruta_windows
        else:
            # Si no hay coincidencia con el patrón, simplemente reemplazar barras inclinadas
            return ruta_wsl.replace('/', '\\')
# This python must be saved in a directory where you have a folder named
# /Molecule/Molecule_Name, as obviusly Molecule_Name holds a .pdb or .pqr file
# ; otherwise, this function won't do anything.

# Data is saved in format {mol_name}_{mesh_density}-{it_count}
# Where mol_name is the abreviated name of the molecule
# mesh_density is the density of the mesh in elements per square amstrong
# it_count is for the mesh ref pluggin and will be treaten as -0 when is the first grid made

# IMPORTANT BUGS - 1. NANOSHAPER MUST BE REPAIRED - ONLY MSMS ALLOWED
# 2. print('juan') not printing in xyzr_to_msh !!!!!!!!!!!! This may be a reason
#    why .msh file is not being created.

#With the .pdb file we can build .pqr & .xyzr files, and they don't change when the mesh density is changed.

def face_and_vert_to_off(face_array , vert_array , path , file_name ):
    '''
    Creates off file from face and vert arrays.
    '''
    off_file = open( os.path.join( path , file_name + '.off' ) , 'w+')
    
    off_file.write( 'OFF\n' )
    off_file.write( '{0} {1} 0 \n'.format(len(vert_array) , len(face_array)) )
    for vert in vert_array:
        off_file.write( str(vert)[1:-1] +'\n' )
    
    for face in face_array:
        off_file.write( '3 ' + str(face - 1)[1:-1] +'\n' )
        
    off_file.close()
    
    return None

def Improve_Mesh(face_array , vert_array , path , file_name ):
    '''
    Executes ImproveSurfMesh and substitutes files.
    '''
    
    #os.system('export LD_LIBRARY_PATH=/vicenteramm/lib/')
    face_and_vert_to_off(face_array , vert_array , path , file_name)
    
    Improve_surf_Path = path + 'fetk/gamer/tools/ImproveSurfMesh/ImproveSurfMesh'
    os.system( Improve_surf_Path + ' --smooth --correct-normals ' + os.path.join(path , file_name +'.off')  )
    
    os.system('mv  {0}/{1}'.format(path, file_name + '_improved_0.off ') + 
                                 '{0}/{1}'.format(path, file_name + '.off '))
    
    new_off_file =  open( os.path.join( path , file_name + '.off' ) , 'r').read().split('\n')
    #print(new_off_file)
    
    num_verts = int(new_off_file[1].split()[0])
    num_faces = int(new_off_file[1].split()[1])

    new_vert_array = np.empty((0,3))
    for line in new_off_file[2:num_verts+2]:
        new_vert_array = np.vstack((new_vert_array , line.split() ))


    new_face_array = np.empty((0,3))
    for line in new_off_file[num_verts+2:-1]:
        new_face_array = np.vstack((new_face_array , line.split()[1:] ))


    new_vert_array = new_vert_array.astype(float)
    new_face_array = new_face_array.astype(int  ) + 1
    
    return new_face_array , new_vert_array

def pdb_to_pqr(mol_name , stern_thickness , method = 'amber' ):
    '''
    Function that makes .pqr file from .pdb using Software/apbs-pdb2pqr-master/pdb2pqr/main.py
    Be careful of the version and the save directory of the pdb2pqr python shell.
    mol_name : Abreviated name of the molecule
    stern_thicness : Length of the stern layer
    method         : This parameter is an 
    '''
    path = os.getcwd()
        
    pdb_file , pdb_directory = mol_name+'.pdb' , os.path.join('Molecule',mol_name)
    pqr_file , xyzr_file     = mol_name+'.pqr' , mol_name+'.xyzr'
    
    if os.path.isfile(os.path.join('Molecule',mol_name,pqr_file)):
        print('File already exists in directory.')
        return None
    
    # The apbs-pdb2pqr rutine, allow us to generate a .pqr file
    pdb2pqr_dir = os.path.join('Software','apbs-pdb2pqr-master','pdb2pqr','main.py')
    exe=('python2.7  ' + pdb2pqr_dir + ' '+ os.path.join(pdb_directory,pdb_file) +
         ' --ff='+method+' ' + os.path.join(pdb_directory,pqr_file)   )
    
    os.system(exe)
    
    # Now, .pqr file contains unneeded text inputs, we will save the rows starting with 'ATOM'.
    
    pqr_Text = open( os.path.join(pdb_directory , pqr_file) ).read()
    pqr_Text_xyzr = open(os.path.join(pdb_directory , xyzr_file )  ,'w+')

    
    for i in pqr_Text.split('\n'):
        row=i.split()
        if row[0]=='ATOM':
            aux=row[5]+' '+row[6]+' '+row[7]+' '+row[-1]
            pqr_Text_xyzr.write(aux + '\n')   
    pqr_Text_xyzr.close()
    
    print('Global .pqr & .xyzr ready.')
    
    # The exterior interface is easy to add, by increasing each atom radii
    if stern_thickness>0: 
        xyzr_file_stern = os.path.join(pdb_directory , mol_name +'_stern.xyzr')
        pqr_Text_xyzr_s = open(xyzr_file_stern ,'w')
        
        for i in pqr_Text.split('\n'):
            row=i.split()
            if row[0]=='ATOM':
                R_vv=float(row[-1])+stern_thickness
                pqr_Text_xyzr_s.write(row[5]+' '+row[6]+' '+row[7]+' '+str(R_vv)+'\n' )      
        pqr_Text_xyzr_s.close()
        print('Global _stern.pqr & _stern.xyzr ready.')
    
    return 

def pqr_to_xyzr(mol_name , stern_thickness , method = 'amber' ):
    '''
    Extracts .xyzr information from .pqr
    mol_name : Abreviated name of the molecule
    stern_thickness : Length of the stern layer
    method          : amber by default , a pdb2pqr parameter to build the mesh.
    '''
    path = os.getcwd()
    
    pqr_directory = os.path.join('Molecule',mol_name)
    pqr_file , xyzr_file     = mol_name+'.pqr' , mol_name+'.xyzr'
     
    # Now, .pqr file contains unneeded text inputs, we will save the rows starting with 'ATOM'.
    
    pqr_Text = open( os.path.join(pqr_directory , pqr_file) ).read()
    pqr_Text_xyzr = open(os.path.join(pqr_directory , xyzr_file )  ,'w+')

    
    for i in pqr_Text.split('\n'):
        row=i.split()
        if len(row)==0: continue
            
        if row[0]=='ATOM':
            aux=' '.join( [row[5],row[6],row[7],row[-1]] )
            pqr_Text_xyzr.write(aux + '\n')   
    pqr_Text_xyzr.close()
    
    print('.xyzr File from .pqr ready.')
    
    # The exterior interface is easy to add, by increasing each atom radii
    if stern_thickness>0: 
        xyzr_file_stern = os.path.join(pqr_directory , mol_name +'_stern.xyzr')
        pqr_Text_xyzr_s = open(xyzr_file_stern ,'w')
        
        for i in pqr_Text.split('\n'):
            row=i.split()
            if row[0]=='ATOM':
                R_vv=float(row[-1])+stern_thickness
                pqr_Text_xyzr_s.write(row[5]+' '+row[6]+' '+row[7]+' '+str(R_vv)+'\n' )      
        pqr_Text_xyzr_s.close()
        print('Global _stern.pqr & _stern.xyzr ready.')
        
    return None

def NanoShaper_config(xyzr_file , dens , probe_radius):
    '''
    Yet in beta version. Changes some data to build the mesh with NanoShaper
    xyzr_file : Directory of the xyzr_file
    dens      : mesh density
    probe_radius : might be set to 1.4
    '''
    t1 = (  'Grid_scale = {:s}'.format(str(dens)) 
                #Specify in Angstrom the inverse of the side of the grid cubes  
              , 'Grid_perfil = 80.0 '                     
                #Percentage that the surface maximum dimension occupies with
                # respect to the total grid size,
              , 'XYZR_FileName = {:s}'.format(xyzr_file)  
              ,  'Build_epsilon_maps = false'              
              , 'Build_status_map = false'                
              ,  'Save_Mesh_MSMS_Format = true'            
              ,  'Compute_Vertex_Normals = true'           
              ,  'Surface = ses  '                         
              ,  'Smooth_Mesh = true'                      
              ,  'Skin_Surface_Parameter = 0.45'           
              ,  'Cavity_Detection_Filling = false'        
              ,  'Conditional_Volume_Filling_Value = 11.4' 
              ,  'Keep_Water_Shaped_Cavities = false'      
              ,  'Probe_Radius = {:s}'.format( str(probe_radius) )                
              ,  'Accurate_Triangulation = true'           
              ,  'Triangulation = true'                    
              ,  'Check_duplicated_vertices = true'        
              ,  'Save_Status_map = false'                 
              ,  'Save_PovRay = false'                     )
    return t1

def xyzr_to_msh(mol_name , dens , probe_radius , stern_thickness , min_area , Mallador ,
               suffix = '' , build_msh=True):
    '''
    Makes msh (mesh format for BEMPP) from xyzr file
    mol_name : Abreviated name of the molecule
    dens     : Mesh density
    probe_radius : might be set to 1.4[A]
    stern_thickness : Length of the stern layer
    min_area        : Discards elements with less area than this value
    Mallador        : MSMS or NanoShaper
    
    outputs : Molecule/{mol_name}/{mol_name}_{dens}-0.msh
    Where -0 was added because of probable future mesh refinement and easier handling of meshes.
    '''
    script_path = os.path.abspath(__file__)
    one_level_up = os.path.abspath(os.path.join(script_path, "../.."))
    mol_directory = os.path.join(one_level_up , "Refinamiento_adaptativo" ,"Molecule" , mol_name)
    xyzr_file     = os.path.join(mol_directory , mol_name + ".xyzr" )
    if stern_thickness > 0:  xyzr_s_file = os.path.join(mol_directory , mol_name + '_stern.xyzr'  )
    
    # The executable line must be:
    #  path/Software/msms/msms.x86_64Linux2.2.6.1 
    # -if path/mol_name.xyzr       (Input File)
    # -of path/mol_name -prob 1.4 -d 3.    (Output File)
   
    # The directory of msms/NS needs to be checked, it must be saved in the same folder that is this file
    if Mallador == 'MSMS':  
              path = os.path.join(mol_directory , mol_name )
              msms_dir = os.path.join(PBJ_PATH, "mesh", "ExternalSoftware", "MSMS", "")
              external_file = "msms"
              os.system("chmod +x " + msms_dir + external_file)
              command = (msms_dir
               + external_file
               + " -if "
               + xyzr_file
               + " -of "
               + path+'_{0:s}-0'.format(str(dens))
               + " -p "
               + str(probe_radius)
               + " -d "
               + str(dens)
               + " -no_header")
              
              print("El comando es : " + command)
              os.system(command)
        #(M_path+ " -if " + xyzr_file + " -of " + os.path.join(mol_directory , mol_name) + " -p " + str(prob_rad) + " -d " + str(dens_msh) + " -no_header")
              print('Normal .vert & .face Done')
              grid = factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador , suffix = '-0', build_msh=build_msh)
              print('Normal .msh Done')
        
        # As the possibility of using a stern layer is available:
              if stern_thickness > 0:
               prob_rad, dens_msh = ' -prob ' + str(probe_radius), ' -d ' + str(dens)
               exe= (M_path+' -if '  + xyzr_s_file + 
              ' -of ' +str(os.path.join(mol_directory , mol_name +'_stern')) + prob_rad  + dens_msh  + mode )
               os.system(exe)
               print('Stern .vert & .face Done')
               stern_grid= factory_fun_msh( mol_directory , mol_name+'_stern', min_area )
               print('Stern .msh Done')
        
    elif Mallador == 'NanoShaper': 
        print("Starting NanoShaper")
        Proceso_terminado = generate_nanoshaper_mesh(str(os.path.join(mol_directory , mol_name + ".xyzr" ))   ,str(mol_directory),  str(mol_name+'_{0:s}-0'.format(str(dens))) , dens   , probe_radius,True,)
        if Proceso_terminado == False:
           print("Error al crear malla con NanoShaper")
           return(Proceso_terminado)
        print("Proceso de generación de malla con NanoShaper terminado")
        Face_path  = str(os.path.join(mol_directory , mol_name +'_{0:s}-0'.format(str(dens)) + ".face"))
        Vert_path = str(os.path.join(mol_directory , mol_name +'_{0:s}-0'.format(str(dens)) + ".vert"))
        grid = import_msms_mesh_1(Face_path , Vert_path , mol_directory , mol_name , suffix , dens)
        print('Mesh Ready ')
        
    return(Proceso_terminado)

def factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador , suffix, build_msh = True):
    '''
    This functions builds msh file adding faces and respective vertices.
    mol_directory : Directory of the molecule
    mol_name      : Abreviated name of the molecule
    min_area      : Min. area set to exclude small elements
    dens          : mesh density
    Mallador      : MSMS - NanoShaper or Self (if doing the GOMR)
    suffix        : Suffix of the .vert and .face file after the mesh density ({mol_name}_{d}{suffix})
                    might be used as -{it_count}
    '''
    # .vert and .face files are readed    
    if Mallador == 'MSMS':
        print('Loading the MSMS grid.')
        vert_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix) ) , usecols=(0,1,2))
        face_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.face'.format(str(dens),suffix) ) , dtype=int , usecols=(0,1,2)) -1
        grid = bempp.api.Grid (np.transpose(vert_Text), np.transpose(face_Text))
        
    elif Mallador == 'NanoShaper':
        print('Loading the NanoShaper grid.')
        script_path = os.path.abspath(__file__)
        vert_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix) ) , skiprows=3 ,usecols=(0,1,2))
        face_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix) ) ,skiprows=3 ,usecols=(0,1,2) , dtype=int)-1
        grid = bempp.api.Grid (np.transpose(vert_Text), np.transpose(face_Text))

    elif Mallador == 'Self':
        print('Loading the built grid.')
        vert_Text = np.loadtxt( os.path.join(Mol_directories , mol_name , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix)), usecols=(0,1,2) )
        face_Text = np.loadtxt( os.path.join(Mol_directories , mol_name , mol_name +'_{0:s}{1}.face'.format(str(dens),suffix)), usecols=(0,1,2) , dtype=int)-1
        grid = bempp.api.Grid (np.transpose(vert_Text), np.transpose(face_Text))
    export_file = os.path.join(Mol_directories, mol_name , mol_name +'_'+str(dens)+ suffix +'.msh') ##cambiar
    bempp.api.export(export_file, grid=grid)
    return grid

def triangle_areas(mol_directory , mol_name , dens , return_data = False , suffix = '', Self_build = False):
    """
    This function calculates the area of each element.
    Avoid using this with NanoShaper, only MSMS recomended
    Self_build : False if using MSMS or NanoShaper - True if building with new methods
    Has a BUG! probably not opening .vert or .face or not creating .txt or both :P .
    """
    
    vert_Text = open( os.path.join(mol_directory , mol_name +'_'+str(dens)+suffix+'.vert' ) ).read().split('\n')
    face_Text = open( os.path.join(mol_directory , mol_name +'_'+str(dens)+suffix+'.face' ) ).read().split('\n')
    area_list = np.empty((0,1))
    area_Text = open( os.path.join(mol_directory , 'triangleAreas_'+str(dens)+suffix+'.txt' ) , 'w+')
    
    vertex = np.empty((0,3))
    
    if not Self_build:
        for line in vert_Text:
            line = line.split()
            if len(line) !=9: continue
            vertex = np.vstack(( vertex, np.array(line[0:3]).astype(float) ))

        atotal=0.0
        # Grid assamble
        for line in face_Text:
            line = line.split()
            if len(line)!=5: continue
            A, B, C, _, _ = np.array(line).astype(int)
            side1, side2  = vertex[B-1]-vertex[A-1], vertex[C-1]-vertex[A-1]
            face_area = 0.5*np.linalg.norm(np.cross(side1, side2))

            area_Text.write( str(face_area)+'\n' )

            area_list = np.vstack( (area_list , face_area ) )
            atotal += face_area

        area_Text.close()

        if return_data:
            return area_list
        
    elif Self_build:
        
        for line in vert_Text[:-1]:
            line = line.split()
            
            vertex = np.vstack(( vertex, np.array(line[0:3]).astype(float) ))

        atotal=0.0
        # Grid assamble
        for line in face_Text[:-1]:
            line = line.split()
            A, B, C = np.array(line[0:3]).astype(int)
            side1, side2  = vertex[B-1]-vertex[A-1], vertex[C-1]-vertex[A-1]
            face_area = 0.5*np.linalg.norm(np.cross(side1, side2))
            area_Text.write( str(face_area)+'\n' )

            area_list = np.vstack( (area_list , face_area ) )
            atotal += face_area

        area_Text.close()

        if return_data:
            return area_list
    
    return None

def normals_to_element( face_array , vert_array , check_dir = False ):
    '''
    Calculates normals to a given element, pointint outwards.
    face_array : Array of vertex position for each triangle
    vert_array : Array of vertices
    check_dir  : checks direction of normals. WORKS ONLY FOR A SPHERE WITH RADII 1!!!!!!!!!!!!
    '''

    normals = np.empty((0,3))
    element_cent = np.empty((0,3))
    
    check_list = np.empty((0,1))
    
    for face in face_array:
        
        f1,f2,f3 = face-1
        v1 , v2 , v3 = vert_array[f1] , vert_array[f2] , vert_array[f3]
        n = np.cross( v2-v1 , v3-v1 ) 
        normals = np.vstack((normals , n/np.linalg.norm(n) )) 
        element_cent = np.vstack((element_cent, (v1+v2+v3)/3. ))
        
        if check_dir:
            v_c = v1 + v2 + v3
            pdot= np.dot( v_c , n )
            if pdot>0:
                check = True
            else:
                check = False
            check_list = np.vstack( (check_list , check ) )
            

    return normals , check_list[:,0]


def vert_and_face_arrays_to_text_and_mesh(mol_name , vert_array , face_array , suffix
                                          , dens , Self_build=True):
    '''
    This rutine saves the info from vert_array and face_array and creates .msh and areas.txt files
    mol_name : Abreviated name for the molecule
    dens     : Mesh density, anyway is not a parameter, just a name for the file
    vert_array: array containing verts
    face_array: array containing verts positions for each face
    suffix    : text added to diference the meshes.
    
    Returns None but creates Molecule/{mol_name}/{mol_name}_{mesh_density}{suffix}.msh file.
    '''

    normalized_path = os.path.join(Mol_directories, mol_name,mol_name+'_'+str(dens)+suffix)
    vert_txt = open( normalized_path+'.vert' , 'w+' )

    for vert in vert_array:
        txt = ' '.join( vert.astype(str) )
        vert_txt.write( txt + '\n')
    vert_txt.close()
    
    face_txt = open( normalized_path+'.face' , 'w+' )
    for face in face_array:
        txt = ' '.join( face.astype(int).astype(str) )
        face_txt.write( txt + '\n')
    face_txt.close()
    script_path = os.path.abspath(__file__)
    General_path =  os.path.abspath(os.path.join(script_path, "../../.."))
    mol_directory = os.path.join(General_path ,"Programas_de_memoria" , "Refinamiento_adaptativo" , "Molecule" , mol_name)
    min_area = 0

    factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador='Self', suffix=suffix)
    #triangle_areas(mol_directory , mol_name , str(dens) , suffix = suffix , Self_build = Self_build)
    
    return None

def Grid_loader(mol_name , mesh_density , suffix , Mallador , GAMer=False, build_msh = True):
    grid_name_File = os.path.join(Mol_directories , mol_name ,mol_name + '_'+str(mesh_density)+suffix+'.msh')
    if os.path.isfile(grid_name_File) and suffix == '-0':
        
        pqr_directory = os.path.join(Mol_directories,mol_name, mol_name+'.pqr' )
        
        if not os.path.isfile(pqr_directory):
            pdb_to_pqr(mol_name , stern_thickness , method = 'amber' )
       
    if suffix == '-0':
        pqr_to_xyzr(mol_name , stern_thickness=0 , method = 'amber' )
        Estado = xyzr_to_msh(mol_name , mesh_density , 1.5 , 0 , 0
                    , Mallador, suffix = suffix , build_msh = build_msh)
        if Estado == False:

            return("Abort")
    if not build_msh:
        
        return None    
    
    #print('Working on '+grid_name_File )
    grid = bempp.api.import_grid(grid_name_File)
    
    
    
    if GAMer:
        face_array = np.transpose(grid.elements)+1
        vert_array = np.transpose(grid.vertices)
        
        new_face_array , new_vert_array = Improve_Mesh(face_array , vert_array , path , mol_name + '_'+str(mesh_density)+suffix )
        
        vert_and_face_arrays_to_text_and_mesh(mol_name , new_vert_array , new_face_array , suffix 
                                          , dens=mesh_density , Self_build=True)
        
        grid = bempp.api.import_grid(grid_name_File)
    
    return grid

def fine_grid_maker(mol_name , dens_f=40.0):
    '''
    Does a 40.0 grid
    Input 
    mol_name : Name of the molecule
    dens     : Mesh density
    Output
    None
    '''
    
    path = os.path.join('Molecule' , mol_name , mol_name + '_{0:.1f}'.format(dens_f))
    if os.path.isfile( path + '.vert' ):
        return None        
    

    x_q , q = run_pqr(mol_name)
    Grid_loader( mol_name , dens_f , '-0' , 'MSMS' , GAMer = False  , build_msh = False)
    
    return None

def run_pqr(mol_name):
    global q , x_q
    q, x_q = np.empty(0), np.empty((0,3))
    print("Starting mesh for :" +mol_name)
    path = os.path.join(Mol_directories, mol_name, mol_name + ".pqr")
    pqr_file = path
    charges_file = open(pqr_file,'r').read().split('\n')
    for line in charges_file:
        line = line.split()
        if len(line)==0: continue
        if line[0]!='ATOM': continue
        q = np.append( q, float(line[8]))
        x_q = np.vstack( ( x_q, np.array(line[5:8]).astype(float) ) )  

    return q , x_q

def error_test(dif, grid, q, x_q):
    
    '''
    Calculate maximum error, maximum potential, maximum ratio, maximum area in maximum error triangle
    Generates a matrix with all potentials, another with all ratio's.
    This is to select the criterion that most influences has in the error.
    
    Parameters:
    dif: error matrix
    grid
    q
    x_q
    
    returns:
    error, area, ratio and potential from maximum error triangle
    '''
    global ep_m
    total_error = np.abs(np.sum (dif))
    print ('Total Error is: {0:.7f}'.format(total_error))
    error_max = np.max(dif)
    print ('Maximum Error is: {0:.7f}'.format(error_max))
    index_error_max = np.where(np.abs(error_max-dif)<1e-12)[0]
    vert_max = np.transpose(grid.vertices)[np.transpose(grid.elements)[index_error_max]]
    #print (np.transpose(grid.elements)[index_error_max])
    #print (index_error_max)
    #print (vert_max)
    error_max_area = grid.volumes[index_error_max]
    print ('Area in max error triangle is:', error_max_area)
    #print (np.sum(grid.volumes))
    all_area = grid.volumes
            
    triangles = (np.transpose(grid.vertices)[np.transpose(grid.elements)]) 
            
    #ratio
    ratio = np.empty(grid.number_of_elements)
    for i in range (grid.number_of_elements):
        L_ab = triangles[i][1] - triangles[i][0]
        L_bc = triangles[i][2] - triangles[i][1]
        L_ca = triangles[i][0] - triangles[i][2]
        A = np.linalg.norm(np.cross(L_ab, L_bc))/np.linalg.norm(L_bc)
        B = np.linalg.norm(np.cross(L_bc, L_ca))/np.linalg.norm(L_ca)
        C = np.linalg.norm(np.cross(L_ca, L_ab))/np.linalg.norm(L_ab)
        values = np.array([A,B,C])
        h_max = np.max(values)
        h_min = np.min(values)
        ratio[i] = h_max / h_min #all ratio
                
    L_ab = vert_max[0][1] - vert_max[0][0]
    L_bc = vert_max[0][2] - vert_max[0][1]
    L_ca = vert_max[0][0] - vert_max[0][2]
            
    A = np.linalg.norm(np.cross(L_ab, L_bc))/np.linalg.norm(L_bc)
    B = np.linalg.norm(np.cross(L_bc, L_ca))/np.linalg.norm(L_ca)
    C = np.linalg.norm(np.cross(L_ca, L_ab))/np.linalg.norm(L_ab)
    values = np.array([A,B,C])
    h_max = np.max(values)
    h_min = np.min(values)
            
    print ('Distance Ratio in max_error_triangle:', h_max/h_min)
            
    #potential
    pot = np.empty(grid.number_of_elements)
    variable = np.empty(len(q))
    for i in range (grid.number_of_elements):
        x = (triangles[i][0][0] + triangles[i][1][0] + triangles[i][2][0]) / 3
        y = (triangles[i][0][1] + triangles[i][1][1] + triangles[i][2][1]) / 3
        z = (triangles[i][0][2] + triangles[i][1][2] + triangles[i][2][2]) / 3
        r_c = np.array([x,y,z])
        for j in range (len(q)):
            variable[j] = q[j] / (4*np.pi*ep_m*np.linalg.norm(r_c - x_q[j]))
        pot[i] = np.sum(variable) #all potentials
        
    x = (vert_max[0][0][0] + vert_max[0][1][0] + vert_max[0][2][0]) / 3
    y = (vert_max[0][0][1] + vert_max[0][1][1] + vert_max[0][2][1]) / 3
    z = (vert_max[0][0][2] + vert_max[0][1][2] + vert_max[0][2][2]) / 3
    r_center = np.array([x,y,z])
    pot_max = np.sum (q / (4*np.pi*ep_m*np.linalg.norm(r_center - x_q, axis = 1)))
    print ('Potential in max_error_triangle:',(pot_max))
            
    error_sort = np.argsort(np.abs(dif))

    potential_sort = np.argsort(np.abs(pot))
    #np.savetxt ('error', error_sort[::-1], fmt='%10.0f')
    #np.savetxt ('potential', potential_sort[::-1], fmt='%10.0f')
    ratio_sort = np.argsort(np.abs(ratio))
    area_sort = np.argsort(np.abs(all_area))
    pot_ratio_sort = np.argsort (np.abs(pot*ratio))
    pot_area_sort = np.argsort (np.abs(pot*all_area))
    ratio_area_sort = np.argsort (np.abs(all_area*ratio))
            
    div_pot_ratio = np.argsort (np.abs(pot/ratio))
    div_ratio_pot = np.argsort (np.abs(ratio/pot))
    div_area_ratio = np.argsort (np.abs(all_area/ratio))
    div_ratio_area = np.argsort (np.abs(ratio/all_area))
    div_pot_area = np.argsort (np.abs(pot/all_area))
    div_area_pot = np.argsort (np.abs(all_area/pot))
    ####
    error_pot = np.linalg.norm(error_sort - potential_sort)
    error_ratio = np.linalg.norm(error_sort - ratio_sort)
    error_area = np.linalg.norm(error_sort - area_sort)
    error_pot_ratio = np.linalg.norm(error_sort - pot_ratio_sort)
    error_pot_area = np.linalg.norm(error_sort - pot_area_sort)
    error_ratio_area = np.linalg.norm(error_sort - ratio_area_sort)
            
    error_pot_ratio2 = np.linalg.norm(error_sort - div_pot_ratio)
    error_ratio_pot = np.linalg.norm(error_sort - div_ratio_pot)
    error_area_ratio = np.linalg.norm(error_sort - div_area_ratio)
    error_ratio_area2 = np.linalg.norm(error_sort - div_ratio_area)
    error_pot_area2 = np.linalg.norm(error_sort - div_pot_area)
    error_area_pot = np.linalg.norm(error_sort - div_area_pot)
            
            
    #print ('Euclidean distance for error vs potential is:', error_pot)
    #print ('Euclidean distance for error vs ratio is:',error_ratio)
    #print ('Euclidean distance for error vs area is:',error_area)
    #print ('Euclidean distance for error vs pot*ratio is:',error_pot_ratio)
    #print ('Euclidean distance for error vs pot*area is:',error_pot_area)
    #print ('Euclidean distance for error vs ratio*area is:',error_ratio_area)
            
    #print ('Euclidean distance for error vs pot/ratio is:',error_pot_ratio2)
    #print ('Euclidean distance for error vs ratio/pot is:',error_ratio_pot)
    #print ('Euclidean distance for error vs area/ratio is:',error_area_ratio)
    #print ('Euclidean distance for error vs ratio/area is:',error_ratio_area2)
    #print ('Euclidean distance for error vs pot/area is:',error_pot_area2)
    #print ('Euclidean distance for error vs area/pot is:',error_area_pot)
    
    Array = np.array([error_pot, error_ratio, error_area, error_pot_ratio, error_pot_area, error_ratio_area,
                     error_pot_ratio2, error_ratio_pot, error_area_ratio, error_ratio_area2, error_pot_area2,
                     error_area_pot])
    #print ('Potential, Ratio, Area, Pot*Ratio, Pot*area, Ratio*Area,\
 #Pot/Ratio, Ratio/Pot, Area/Ratio, Ratio/Area, Pot/Area, Area/Pot')
    
    #print (np.argsort(Array)[0], np.argsort(Array)[1], np.argsort(Array)[2], np.argsort(Array)[3])
    
    
    return error_max, error_max_area, h_max/h_min, pot_max, pot

def potential_calc(grid, q , x_q ):
    '''
    This function calculates the potential of all elements
    
    Parameters:
    grid
    q
    x_q
    
    '''
    triangles = (np.transpose(grid.vertices)[np.transpose(grid.elements)]) 
    pot = np.empty(grid.number_of_elements)
    variable = np.empty(len(q))
    for i in range (grid.number_of_elements):
        x = (triangles[i][0][0] + triangles[i][1][0] + triangles[i][2][0]) / 3
        y = (triangles[i][0][1] + triangles[i][1][1] + triangles[i][2][1]) / 3
        z = (triangles[i][0][2] + triangles[i][1][2] + triangles[i][2][2]) / 3
        r_c = np.array([x,y,z])
        for j in range (len(q)):
            variable[j] = q[j] / (4*np.pi*ep_m*np.linalg.norm(r_c - x_q[j]))
        pot[i] = np.sum(variable) #all potentials
    
    return pot

def fix_mesh(mesh):
    """
    Receives a trimesh mesh object and tries to fix it iteratively using the trimesh.repair.broken_faces() function.
    Prints a message if the mesh couldn't be fixed.
    Parameters
    ---------
    mesh : trimesh mesh object
        Original mesh object.
    Returns
    ----------
    mesh : trimesh mesh object
        Mesh after trying to fix it.
    """
    mesh.fill_holes()
    mesh.process()
    iter_limit = 20
    iteration = 0
    while not mesh.is_watertight and iteration < iter_limit:
        merge_tolerance = 0.05
        needy_faces = trimesh.repair.broken_faces(mesh)
        for vert_nf in mesh.faces[needy_faces]:
            for nf in vert_nf:
                for c, check in enumerate(
                    np.linalg.norm(mesh.vertices[vert_nf] - mesh.vertices[nf], axis=1)
                ):
                    if (check < merge_tolerance) & (0 < check):
                        mesh.vertices[nf] = mesh.vertices[vert_nf[c]]
        iteration += 1
    if iteration > iter_limit - 1:
        print(" not watertight")
    mesh.fill_holes()
    mesh.process()
    return mesh


def convert_pdb2pqr(mesh_pdb_path, mesh_pqr_path, force_field, str_flag=""):
    """
    Using pdb2pqr from APBS (pdb2pqr30 on bash) creates a pqr file from a pdb file.
    Parameters
    ----------
    mesh_pdb_path : str
        Absolute path of pdb file.
    mesh_pqr_path : str
        Absolute path of pqr file.
    force_field : str
        Indicates selected force field to create pqr file, e.g. {AMBER,CHARMM,PARSE,TYL06,PEOEPB,SWANSON}
    str_flag : str, default '' (empty string)
        Indicates additional flags to be used in bash with pdb2pqr30
    Returns
    ----------
    None
    """
    force_field = force_field.upper()
    if str_flag:
        subprocess.call(
            ["pdb2pqr30", str_flag, "--ff=" + force_field, mesh_pdb_path, mesh_pqr_path]
        )
    else:
        subprocess.call(
            ["pdb2pqr30", "--ff=" + force_field, mesh_pdb_path, mesh_pqr_path]
        )


# Funciona bien:
def convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path):
    """
    Creates a xyzr format file from a pqr format file.
    Parameters
    ----------
    mesh_pqr_path : str
        Absolute path of pqr file
    mesh_xyzr_path : str
        Absolute path of xyzr file
    Returns
    ----------
    None
    """
    pqr_file = open(mesh_pqr_path, "r")
    pqr_data = pqr_file.read().split("\n")
    xyzr_file = open(mesh_xyzr_path, "w")
    for line in pqr_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        xyzr_file.write(
            line[5] + "\t" + line[6] + "\t" + line[7] + "\t" + line[9] + "\n"
        )
    pqr_file.close()
    xyzr_file.close()
def generate_nanoshaper_mesh(
    mesh_xyzr_path,
    output_dir,
    output_name,
    density,
    probe_radius,
    save_mesh_build_files,
):  
    
    print("Making vert. and face. using NanoShaper")
    mol_name = output_name.split("_")[0]
    NanoShaper_Terminado = False
    script_path = os.path.abspath(__file__)
    General_path = os.path.abspath(os.path.join(script_path, "../../.."))
    nanoshaper_dir = os.path.join(General_path , "Software" , "pkg_nanoshaper_0.7.8")
    if platform.system() != "Linux":
     raise EnvironmentError("Error: este script solo puede ejecutarse en Linux.")
    if not os.path.exists(nanoshaper_dir):
        os.makedirs(nanoshaper_dir)
    config_template_file = open(os.path.join(nanoshaper_dir , "config"), "r")
    print("NanoShaper Configuration File =" + nanoshaper_dir + "/surfaceConfiguration.prm")
    config_file = open(os.path.join(nanoshaper_dir , "surfaceConfiguration.prm"), "w")
    for line in config_template_file:
        if "XYZR_FileName" in line:
            line = "XYZR_FileName = " + corregir_ruta(mesh_xyzr_path) + " \n"
        elif "Grid_scale" in line:
            line = "Grid_scale = {:04.1f} \n".format(density)
        elif "Probe_Radius" in line:
            line = "Probe_Radius = {:03.1f} \n".format(probe_radius)
        config_file.write(line)

    config_file.close()
    config_template_file.close()
    time.sleep(1)
    if platform.system() == "Linux":
       permission_comand = ["chmod" , "+x" , os.path.join(nanoshaper_dir, "NanoShaper")]
       permission_comand_2 = ["chmod" , "+x" , os.path.join(nanoshaper_dir, "NanoShaper.bin")]
       subprocess.run(permission_comand, check=True)
       subprocess.run(permission_comand_2, check=True)
       comando = [os.path.join(nanoshaper_dir, "NanoShaper") , os.path.join(nanoshaper_dir , "surfaceConfiguration.prm")]
    else:
       comando = [r'/mnt/c/Windows/System32/cmd.exe', '/C', 'cd', '/D', corregir_ruta(nanoshaper_dir), '&&', corregir_ruta                  (nanoshaper_dir + "NanoShaper32.exe")]
    Terminal_Nano_Shaper_stdout = os.path.join( output_dir ,output_name + "_NanoOutput_TEMP_stdout.txt")
    Terminal_Nano_Shaper_stderr = os.path.join( output_dir ,output_name + "_NanoOutput_TEMP_stderr.txt")
    Directorio_History = os.path.join(output_dir, mol_name + "_History")
    if not os.path.exists(Directorio_History):
        with open(Directorio_History, 'w') as archivo:
         print("Archivo moificado en grid_maker")
         pass
        print(f"Text archive created: {Directorio_History}")
    else:
        print(f"Text archive already exist: {Directorio_History}")
    print("Ejecutando comando:" , comando )
    proceso = subprocess.Popen(comando, stdout=PIPE, stderr=PIPE, text=True , cwd = nanoshaper_dir)
    salida_normal_nano, salida_error_nano = proceso.communicate()
    salida_error = "<<ERROR>>" in salida_normal_nano
    cita_encontrada = "<<CITATION>>" in salida_normal_nano
    with open(Terminal_Nano_Shaper_stdout, "w") as archivo_stdout:
        archivo_stdout.write(salida_normal_nano)
    print("Saving stderr in:" + Terminal_Nano_Shaper_stderr)
    with open(Terminal_Nano_Shaper_stderr, "w") as archivo_stderr:
        archivo_stderr.write(salida_error_nano)
    proceso.wait()
    if cita_encontrada and not salida_error:
     NanoShaper_Terminado = True
     print("Cita encontrada. NanoShaper terminado con exito.")
    else:
     NanoShaper_Terminado = False
     print("Cita no encontrada proceso interrumpido por un error en NanoShaper.")
    
    
    linea = output_name + "    " + "NanoShaper_Status_Done:" + str(NanoShaper_Terminado) + "    " + "Memory_Exception:" +"False"+ "    " + "Time_Exception:" + "False" + "    " + "Procces_end:False"
    archivo_historia = Directorio_History
    # Leer el contenido del archivo
    with open(archivo_historia, 'r') as archivo:
        lineas = archivo.readlines()

    # Buscar y reemplazar la línea que comienza con output_name
    linea_modificada = False
    for i in range(len(lineas)):
        elementos = lineas[i].strip().split()
        if len(elementos) > 0 and elementos[0] == output_name:
            lineas[i] = linea + "\n"
            linea_modificada = True
            break  # Detener el bucle después de encontrar la primera coincidencia

    # Si no se encontró ninguna línea con output_name, agregar la nueva línea al final
    if not linea_modificada:
        lineas.append(linea + "\n")
        print(f"No se encontró ninguna línea con '{output_name}', se agregó la nueva línea.")

    # Escribir el contenido de vuelta al archivo
    with open(archivo_historia, 'w') as archivo:
        archivo.writelines(lineas)
    print("Las siguientes lineas han sido actualziadas :" , lineas)

    try:
        print("Output directory = " ,output_dir , "nanoshaper_dir" , nanoshaper_dir + "/triangulatedSurf.vert")
        print("Output_name =" ,output_name )
        os.chdir(output_dir)
        vert_file = open(os.path.join(nanoshaper_dir , "triangulatedSurf.vert"), "r")
        vert = vert_file.readlines()
        vert_file.close()
        face_file = open(os.path.join(nanoshaper_dir , "triangulatedSurf.face"), "r")
        face = face_file.readlines()
        face_file.close()
        vert_file = open(os.path.join( output_dir ,output_name + ".vert"), "w")
        vert_file.write("".join(vert[3:]))
        vert_file.close()
        face_file = open(os.path.join( output_dir ,output_name + ".face"), "w")
        face_file.write("".join(face[3:]))
        face_file.close()
        print(".vert y .face of " + output_name + " generated")


        if not save_mesh_build_files:
            shutil.rmtree(nanoshaper_dir)

        os.chdir("..")
    
    except(OSError, FileNotFoundError):
        print("The file doesn't exist or it wasn't created by NanoShaper")
        
    return(NanoShaper_Terminado)


def convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path):
    """
    Creates an OFF format mesh file from a .face file and a .vert file.
    Parameters
    ----------
    mesh_face_path : str
        Absolute path of the .face file.
    mesh_vert_file : str
        Absolute path of the .vert file.
    mesh_off_path : str
        Absolute path of the .off file.
    Returns
    ----------
    None
    """
    face = open(mesh_face_path, "r").read()
    vert = open(mesh_vert_path, "r").read()

    faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)

    data = open(mesh_off_path, "w")
    data.write("OFF" + "\n")
    data.write(str(verts.shape[0]) + " " + str(faces.shape[0]) + " " + str(0) + "\n")
    for vert in verts:
        data.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\n")
    for face in faces:
        data.write(
            "3" + " " + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + "\n"
        )


def import_msms_mesh_1(mesh_face_path, mesh_vert_path ,mol_directory , mol_name , suffix, dens):
    """
    Creates a bempp grid object from .face and .vert files.
    Parameters
    ----------
    mesh_face_path : str
        Absolute path of the .face file.
    mesh_vert_file : str
        Absolute path of the .vert file.
    Returns
    ----------
    grid : Grid
        Bempp Grid object.
        
    """
    face = open(mesh_face_path, "r").read()
    vert = open(mesh_vert_path, "r").read()
    faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)
    grid = bempp.api.Grid(verts.transpose(), faces.transpose())
    export_path = os.path.join(mol_directory , mol_name +'_'+str(dens)+ suffix +'.msh')
    bempp.api.export(export_path, grid=grid)
    return grid


def import_off_mesh(mesh_off_path):
    """
    Creates a bempp grid object from a .OFF files.
    Parameters
    ----------
    mesh_off_path : str
        Absolute path of the .off file.
    Returns
    ----------
    grid : Grid
        Bempp Grid object.
    """
    grid = bempp.api.import_grid(mesh_off_path)
    return grid


def density_to_nanoshaper_grid_scale_conversion(mesh_density):
    """
    Converts the grid density value into NanoShaper's grid scale value.
    Parameters
    ----------
    mesh_density : float
        Desired density of the grid.
    Returns
    ----------
    grid_scale : float
        Grid scale value to be used in NanoShaper.
    """
    grid_scale = round(
        0.797 * (mesh_density**0.507), 2
    )  # Emperical relation found by creating meshes using nanoshaper and calculating their density
    return grid_scale
