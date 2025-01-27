#include <iostream>
#include <casc/casc>
#include <gamer/gamer.h>
#include <gamer/SurfaceMesh.h>
#include <memory> // Para std::unique_ptr
#include <cstdlib> // Para std::stoi (convierte string a int)

int main(int argc, char** argv) {
    // Verifica si el archivo de entrada fue proporcionado como argumento
    if (argc < 2) {
        std::cerr << "Error: No input file provided." << std::endl;
        return 1;
    }

    // Nombre del archivo de entrada
    std::string file_path = argv[1];

    // Nombre del archivo de salida (opcional, con valor predeterminado)
    std::string output_file = (argc >= 3) ? argv[2] : "output_file.off";

    // Número de iteraciones para el suavizado (opcional, sin valor predeterminado)
    int max_iterations = (argc >= 4) ? std::stoi(argv[3]) : -1; // -1 indica que no se aplicará suavizado

    // Flag para preservar las crestas
    bool preserve_ridges = true; // Esto también podría ser un parámetro opcional si se desea

    // Llamar a gamer::readOFF para leer el archivo y obtener un SurfaceMesh
    auto mesh = gamer::readOFF(file_path);
    if (!mesh) {
        std::cerr << "Error: Failed to read the mesh from file." << std::endl;
        return 1;
    }

    // Calcular el volumen de la malla
    double volume = gamer::getVolume(*mesh);
    std::cout << "Volumen de la malla: " << volume << std::endl;

    // Verificar si el volumen es negativo
    if (volume < 0) {
        std::cout << "Volumen negativo. Invirtiendo las normales..." << std::endl;
        // Invertir las normales si el volumen es negativo
        gamer::flipNormals(*mesh);

        // Verificar el volumen nuevamente
        double corrected_volume = gamer::getVolume(*mesh);
        std::cout << "Volumen corregido: " << corrected_volume << std::endl;
    }

    // Aplicar suavizado si se especificó un número de iteraciones
    if (max_iterations > 0) {
        std::cout << "Suavizando la malla con " << max_iterations << " iteraciones..." << std::endl;
        gamer::smoothMesh(*mesh, max_iterations, preserve_ridges);
    } else {
        std::cout << "No se especificó número de iteraciones. Saltando el suavizado." << std::endl;
    }

    // Escribir la malla (suavizada o no) al archivo de salida
    gamer::writeOFF(output_file, *mesh);
    std::cout << "Archivo de salida generado: " << output_file << std::endl;

    return 0;
}
