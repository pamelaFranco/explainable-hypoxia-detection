*Pasos para obtener una imagen etiquetada

Para obtener una imagen etiquetada, es decir, que cada vaso capilar (segmento) posea un nivel de gris correspondiente a su nombre.

1. Copiar todos los códigos de fuentes que constituyen a VascuSynth, en una nueva 
   carpeta que se llamó ‘VascuSynthEtiquetado’
2. Modificar los códigos: 

   TreeDrawer.cpp — Código donde se dibuja la red vascular (matriz) en la imagen, 
                    declarando la variable en la función ‘valueAtVoxel’.
                    int segment (j); 
                    Que corresponde al nombre de cada segmento.
                    Por tanto se cambia value += 32 a value =segment;
                    Y se quita la condición value —; luego del primer for, pues de 
                    esta manera no se resta un valor de gris a la imagen. 

  Además, ahora se necesita guardar mayor cantidad de valores de grises, debido a que   
  el número de segmentos es mayor a 255, por tanto se cambian todas las condiciones 
  del tamaño de pixel, para almacenar mayor cantidad de datos. Por tanto se decide  
  cambiar de una variable char a short, modificando los archivos.
  TreeDrawer.cpp, TreeDrawer.h y VascuSynth.cpp
  Tal como se comenta en cada uno de ellos.

  Y al igual que en ‘VascuSynthBinario’, se modifica:

  VascuSynth.cpp - Archivo principal, donde se agrupan los parámetros, generando la
                    red vascular y dibujando el árbol realizado en ‘TreeDrawer’ en 
                    una imagen volumétrica, en una serie de imágenes axiales (2D).
                    Cambiando en ‘drawImage’, la extensión de la imagen de .jpg 
                    a .tiff, debido a que este último no ocupa un archivo de 
                    comprensión, lo que mejoraría la 
                    resolución de la imagen.


3. Generar VascuSynthBinario de la misma manera que VascuSynth.
