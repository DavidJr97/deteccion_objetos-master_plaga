# Detección de Objetos con TensorFlow

Detección de objetos: Algoritmo el cual se encarga de detectar varios elementos dentro de una imagen y clasificarlos. Por ejemplo, le damos una imagen que contiene una parada de tránsito y nos identifica que en la imagen hay un semáforo, autos, señales de tránsito, árboles etc.

# Requerimientos

Para poder llevar a cabo este tutorial en tu totalidad es importante tener lo siguiente en nuestra computadora.

 

- Python 3.6
- Tensorflow 1.14
- Numpy 1.16.4
- Pandas
- Matplotlib
- Tarjeta de gráficos (Recomendada para poder hacer un entrenamiento de manera rápida, aunque es posible hacerlo sin GPU, puede llegar a tardar horas o      días sin una tarjeta de gráficos NVIDIA)

# Pasos a seguir

Estos son los pasos que seguiremos en este tutorial, no te preocupes si algo no queda claro, más adelante lo veremos a detalle.

- Preparación de la data
- Preparar nuestros datos para entrenamiento, es decir marcar en un set de entrenamiento las coordenadas donde están los objetos que querremos que           detecte. - Usaremos un programa que nos ayuda a hacer esto de manera fácil y nos da un archivo XML con la información que requerimos.
- Convertir los datos de XML a TFRecord. TFrecord es el formato de imagen que necesita nuestro algoritmo para poder entrenarse.

# Entrenamiento

- Elegir el modelo que entrenaremos.
- Preparar los archivos para entrenamiento
- Configuración de modelo
- Etiquetas de entrenamiento
- Gráfica computacional del modelo
- Entrenar el modelo

# Preparación de la data
Antes de empezar necesitamos preparar los datos con los que entrenaremos a nuestro programa. Para esto necesitaremos tener un monton de imagenes, como mínimo recomendaria tener 200 imágenes por cada uno de los objetos que queremos detectar (si en una imagen tenemos 3 de los objetos que queremos aprender a detectar, esto podria contar como tres imágenes).

Al etiquetar nuestras imágenes le diremos a nuestro programa en que coordenadas de nuestras imágenes puede encontrar cada uno de los objetos que queremos que nuestro programa pueda detectar, esta puede ser una tarea algo tediosa, pero usaremos una herramienta que nos facilitara hacerlo y aparte la recompensa al final será grande.

Es importante que tengamos un set de datos variado, es decir que tengamos los objetos que queremos detectar desde varios ángulos, tipos de iluminación, posiciones etc. Al igual, también es importante que nuestras imágenes no sean de gran tamaño ya que pueden llegar a ser mucho para nuestra computadora, por lo cual recomiendo que se haga un a modificación en tamaño para que una imagen no pese más de 0.5 MB

 

IMPORTANTE:

Sobre la misma imagen podemos seleccionar distintos elementos que queremos que nuestro programa detecte, por ejemplo, en la misma imagen marcar autos, semáforos, pasos peatonales, camionetas, letreros etc.

 

# Etiquetado con labelImg
 
Estaremos usando un programa llamado labelImg el cual nos facilitara el etiquetado de nuestras imágenes, para descargarlo e instalarlo les recomiento entrar a su Github (https://github.com/tzutalin/labelImg)  tambien pueden descargarlos para Windows o Linux desde esta liga (https://tzutalin.github.io/labelImg)

 Básicamente lo que hacemos con este programa es abrir una imagen, seleccionar un recuadro para marcar el objeto que queremos que nuestro programa aprenda a detectar y salvar un XML con la información de las coordenadas. Para ver un poco como funciona pueden ver en este video 

 Estas imágenes (Junto con el archivo XML) que se genera las debemos de guardar en la carpeta de ‘images’ dentro del proyecto.

 

# Conversion de las imagenes a TFRecord
 

Ya que tenemos todas la imágenes con sus respectivos XML marcando sus coordenadas, tendremos que convertirlas a un formato llamado TFRecord, este tipo de archivo es especial para que nuestra red neuronal en Tensor Flow pueda ser entrenada, el TFRecord contendra la informacion de todas las imágenes y las coordenadas que marcamos en un solo archivo. Para poderlos llevar a TFRecord, primero convertiremos TODOS los XMLs en un solo archivo tipo CSV, después ya convertiremos estos CSVs al formato final.

Antes de empezar, vamos a duplicar nuestras imágenes, haremos dos carpetas llamadas ‘img_test’ e ‘img_entrenamiento’ en la primera pondremos alrededor del 10% de nuestras imagenes con sus respectivos XMLs y en la segunda el 90% restante. (DUPLICAREMOS LAS IMÁGENES, ES DECIR QUE EN LA CARPETA DE IMÁGENES SEGUIREMOS TENIENDO EL 100% DE LAS IMÁGENES)

Ya que tenemos las imágenes en esta estructura ahora en nuestra terminal (cmd en windows o bash en ubuntu) nos posicionamos en la carpeta ‘deteccion-de-objetos. Y ejecutaremos un comando a la vez. 

# python setup.py build
# python setup.py install

Nota: para ir a la carpeta usa el comando "cd", en mi caso fue:
# cd C:\Users\Julio Ríos\Downloads\deteccion-de-objetos

Luego ir a la carpeta slim con el comando cd (estando en la deteccion-de-objetos) y ejecutar el siguiente comando:

# pip install -e .

Volver (desde el command prompt) a la carpeta principal, en mi caso la llame deteccion-de-objetos.
Nota: esto lo haces ejecutando el siguiente comando: cd ../
 


Después nos regresamos a la carpeta raíz del repositorio que clonamos de GitHub (deteccion-de-objetos) y ejecutamos los siguientes cuatros comandos.

 # python xml_a_csv.py --inputs=img_test --output=test

# python xml_a_csv.py --inputs=img_entrenamiento --output=entrenamiento

# python csv_a_tf.py --csv_input=CSV/test.csv --output_path=TFRecords/test.record --images=images

# python csv_a_tf.py --csv_input=CSV/entrenamiento.csv --output_path=TFRecords/entrenamiento.record --images=images

Suponiendo que los scripts corrieron sin problema debemos tener ahora una carpeta llamada TFRecords en la cual tendremos dos archivos, entrenamiento.record y test.record  Estos dos archivos ya contienen la información de todas las imágenes y de las coordenadas de los objetos que marcamos.

Ya con esto listo pasaremos a preparar los archivos necesarios para nuestro entrenamiento y al entrenamiento del modelo que deseemos.

 

# Entrenamiento

Elegir modelo a entrenar
Antes de empezar, debemos decidir que modelo es el que querremos entrenar, algunos nos ofrecen detecciones más veloces, sacrificando certeza o viceversa. Para ver todos los modelos podemos ingresar a esta liga. En este tutorial usaremos el modelo faster_rcnn_resnet101_coco (dar clic para descargar) el cual nos brinda predicciones más veloces.  A su vez tambien descargaremos un archivo tipo config que coincida con el modelo que vamos a entrenar (faster_rcnn_resnet101_coco.config), desde esta liga: (https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

Al descargarlo encontraremos varios archivos, los que nos interesan para entrenar un modelo desde cero son:

faster_rcnn_resnet101_coco.config (configuración sobre cómo entrenaremos el modelo)
Model.ckpt.index
Model.ckpt.meta
Model.ckpt.data-000000-of-00001
 

Tomemos los archivos con el formato ‘xxxx.ckpt.yyyy’ al igual que el XXXX.config y pasarlos a la carpeta llamada ‘modelo’ dentro de nuestro proyecto.

 
Si no queremos entrenar un programa nosotros y queremos usar un algoritmo ya pre-entrenado podemos usarlo con los archivos mencionados anteriormente y aparte con el ‘frozen_inference_graph.pb’ y ‘saved_model.pb’ el cual tiene toda la información sobre un modelo ya pre-entrenado

 

# Preparar archivos para entrenamiento
 

Ya que elegimos qué modelo vamos a entrenar, ahora es momento de preparar unos archivos de configuración, estos archivos le dirán a nuestro script de entrenamiento, donde encontrar el modelo que encontramos, donde encontrar las imagenes para entrenar, cuales son las etiquetas que usaremos (los objetos que queremos entrenar) entre otros parámetros más.

Estos archivos los podemos en la carpeta llamada ‘configuracion’

# Etiquetas (label_map.pbtxt)
En este archivo (configuracion/label_map.pbtxt) le dirá a nuestro algoritmo cuales son las etiquetas sobre el cual lo entrenaremos. El nombre que pongamos en las etiquetas debe ser el mismo que usamos en la herramienta labelImg (incluyendo mayúsculas y espacios). Básicamente este archivo tiene una serie de elementos ‘item’ con su respectivo identificador ‘id’ y nombre de clase ‘name’.

 
He aquí un ejemplo, esto cambia segun el numero de elementos que quieras aprender a detectar.

 
 item {
  id: 1
  name: 'Auto'
}
item {
  id: 2
  name: ‘Semaforo’
}

item {
  id: 3
  name: 'Paso Peatonal'
}
 

# Labels.txt
Este archivo es similar al pasado pero mucho mas sencillo, solo es una lista de los elementos que queremos detectar, siendo el primer elemento (en el primer renglon) siempre el valor null. He aqui un ejemplo.

null
Auto
Semaforo
Paseo Peatonal
 

# Configuración de entrenamiento (faster_rcnn_resnet101_coco.config)
Todos los archivos que hemos editado tienen un grado de importancia, pero si fuera a elegir uno como favorito seria este. Si hemos seguido el tutorial este es el archivo que debemos tener en la carpeta de ‘modelo’

Este archivo es el que nuestro script para entrenamiento va a leer para saber parámetros sumamente importantes, tales como:

Donde obtener los tfrecords
Donde obtener el archivos de etiquetas label_map.pbtxt
Donde encontrar los archivos requeridos de nuestro modelo (los checkpoints que aparecen como xxxxx.ckpt.yyyyy)
El número de pasos a entrenar
Batch_size (Número de imágenes que entrenaremos en cada iteración, podemos empezar con un número bajo como 1 e irlo subiendo si vemos que nuestra computadora lo soporta)
 

Para cambiar estos parámetros, debemos de abrir el archivo pipeline.config y cambiar todos los campos que digan ‘PATH_TO_BE_CONFIGURED’.  Los cambios que tenemos que hacer son los siguientes:

 

model {
  faster_rcnn {
    num_classes: 13 (Aqui ponemos el número de objetos a detectar)
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }

...




train_config: {
  batch_size: 1
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "modelo/model.ckpt"
  from_detection_checkpoint: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "TFRecords/entrenamiento.record"
  }
  label_map_path: "configuracion/label_map.pbtxt"
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "TFRecords/test.record"
  }
  label_map_path: "configuracion/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  num_epochs: 1
}
 

# Entrenar
Excelente, ya estamos ahora en la parte más interesante, vamos a empezar a entrenar nuestro algoritmo, teniendo ya todo esto preparado el entrenamiento será sencillo, solo debemos correr el siguiente comando en nuestra terminal.

 

# python object_detection/train.py --logtostderr --train_dir=train --pipeline_config_path=modelo/faster_rcnn_resnet101_coco.config
 

Si todo ha salido bien veremos en nuestra terminal algo similar a esto:

 

INFO:tensorflow:global step 11: loss = 0.6935 (0.648 sec/step)
INFO:tensorflow:global step 12: loss = 0.7426 (0.885 sec/step)
INFO:tensorflow:global step 13: loss = 0.7700 (3.551 sec/step)
INFO:tensorflow:global step 14: loss = 0.8026 (0.664 sec/step)
INFO:tensorflow:global step 15: loss = 0.9608 (0.646 sec/step)
 

Lo que estamos buscando es llegar a un ‘loss’ muy bajo, como mínimo que esté por debajo de 0.9, ya que llegue a este número, podemos terminar el entrenamiento tecleando CTRL -C desde terminal/

# Congelar el modelo entrenado
Ahora que hemos terminado nuestro entrenamiento, tendremos una carpeta llamada ‘train’ en la cual tendremos varios checkpoints (los cuales nos sirven en el futuro por si queremos re-entrenar sobre lo que ya hemos hecho) y un graph.pbtxt , estos archivos son los que contienen la información necesaria para poder hacer predicciones en el futuro, pero antes de esto debemos de ‘congelar’ nuestro modelo, es decir, vamos a convertir nuestros ckeckpoints a un modelo final.

 

Para esto, solo debemos correr un comando, la parte STEP_NUMBER debemos de cambiarla por el ultimo checkpoint que tengamos generado, es decir el de valor mas alto.

 

# python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path modelo/faster_rcnn_resnet101_coco.config  --trained_checkpoint_prefix train/model.ckpt-684 --output_directory modelo_congelado
Después de haber corrido esto con éxito tendremos un archivo en una carpeta llamada ‘modelo_congelado’, este ya es nuestro archivo listo para generar predicciones.

# Prediccion
 

Listo, hemos llegado al final, espero que todos hayan llegado hasta aquí sin problemas. Ahora es momento de generar predicciones. Para esto solo tenemos que poner las imágenes en las que queremos generar detección de objetos en la carpeta llamada ‘img_pruebas’ y correremos el siguiente comando, el resultado lo obtendremos en una nueva carpeta llamada output.

 

# python object_detection/object_detection_runner.py
  

LISTO! Ya tenemos nuestras imágenes con predicciones.
 

# Teconolgias usadas

1. [Tensorflow](https://www.tensorflow.org/?hl=es)
2. [Anaconda](https://www.anaconda.com/products/individual) 
3. [LabelImg](https://tzutalin.github.io/labelImg)
4. [Visual Studio Code](https://code.visualstudio.com/)



# Versiones

A continuación se muestra en la Tabla I el detalle de cada versión especificando el commit y su descripción de la funcionalidad incluida.



# Detección de Plagas en Cultivo de Café con Modelo a entrenado.

1. ![brocacafe]( https: // user-images.githubusercontent.com/36302181/82742866-59d70700-9d20-11ea-8d3a-b10ec8b47b8c.jpeg)
2. ![ojodegallo]( https://user-images.githubusercontent.com/36302181/82742867-5b083400-9d20-11ea-8710-fb54e8cc2ddf.jpeg)
3. ![roya]( https://user-images.githubusercontent.com/36302181/82742868-5c396100-9d20-11ea-8211-7b9584650ff9.jpeg)
4. ![roya]( https://user-images.githubusercontent.com/36302181/82742869-5c396100-9d20-11ea-9dc2-f6a0e98e8bb2.jpeg)


