import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
import os
from tqdm.notebook import tqdm # Importamos la versión bonita para notebooks

def load_image_for_eval(image_path):
    """Carga y preprocesa una imagen individual para inferencia."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    # ¡IMPORTANTE!: Usar el mismo preprocesamiento que en el entrenamiento (ResNet)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, image_path

def evaluate(image_path, encoder, decoder, tokenizer, max_len, attention_features_shape=49):
    """
    Genera un caption para una imagen dada usando búsqueda Greedy.
    Devuelve la frase generada y los pesos de atención para visualizar.
    """
    attention_plot = np.zeros((max_len, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    # 1. Preprocesar imagen
    temp_input = tf.expand_dims(load_image_for_eval(image_path)[0], 0)
    
    # 2. Pasar por Encoder
    features = encoder(temp_input, training=False)

    # 3. Iniciar Decoder con <start>
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    
    result = []

    # 4. Bucle palabra a palabra
    for i in range(max_len):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        # Guardar pesos de atención para plotear luego
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        # Búsqueda Greedy: Tomar la palabra con mayor probabilidad
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        # Si es <end>, terminamos
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        result.append(tokenizer.index_word[predicted_id])

        # La predicción actual es la entrada del siguiente paso
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, attention_plot


def calculate_bleu_score(encoder, decoder, tokenizer, max_len, test_img_paths, all_captions_dict, sample_size=None):
    """
    Calcula los scores BLEU-1 a BLEU-4 para el conjunto de prueba.
    
    Args:
        encoder, decoder: Modelos entrenados.
        tokenizer: Tokenizador ajustado.
        max_len: Longitud máxima de secuencia.
        test_img_paths (list): Lista con las rutas de las imágenes de test.
        all_captions_dict (dict): Diccionario {nombre_imagen: [cap1, cap2...]} con las captions limpias.
        sample_size (int, optional): Si se define, solo evalúa esa cantidad de imágenes (para pruebas rápidas).
    """
    actual, predicted = [], []
    
    # Si queremos probar rápido, limitamos el número de imágenes
    eval_paths = test_img_paths[:sample_size] if sample_size else test_img_paths

    print(f"Calculando BLEU para {len(eval_paths)} imágenes...")

    for img_path in tqdm(eval_paths):
        # 1. Generar predicción del modelo
        # Nota: evaluate devuelve (result, attention_plot), solo queremos result
        pred_seq, _ = evaluate(img_path, encoder, decoder, tokenizer, max_len)
        predicted.append(pred_seq)
        
        # 2. Obtener referencias reales
        img_name = os.path.basename(img_path)
        
        # Obtenemos todas las captions reales para esa imagen
        # Las limpiamos y tokenizamos (split) para que coincidan con la predicción
        # Quitamos <start> y <end> si están en el diccionario para comparar solo el contenido
        raw_captions = all_captions_dict[img_name]
        references = []
        for c in raw_captions:
            # Asumiendo que c es "<start> un perro corre <end>"
            tokens = c.split()
            # Quitamos <start> (índice 0) y <end> (índice -1)
            # Ajusta esto según cómo hayas guardado tus captions limpias
            if tokens[0] == '<start>': tokens = tokens[1:]
            if tokens[-1] == '<end>': tokens = tokens[:-1]
            references.append(tokens)
            
        actual.append(references)

    # 3. Calcular BLEU Scores
    # BLEU-1: Coincidencia de palabras sueltas (unigramas)
    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    # BLEU-2: Coincidencia de pares de palabras (bigramas)
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    # BLEU-3: Trigramas
    b3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    # BLEU-4: Cuatrigramas (frases más largas)
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    print(f'\n--- Resultados BLEU ---')
    print(f'BLEU-1: {b1:.4f}')
    print(f'BLEU-2: {b2:.4f}')
    print(f'BLEU-3: {b3:.4f}')
    print(f'BLEU-4: {b4:.4f}')
    
    return b1, b2, b3, b4
