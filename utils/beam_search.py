import tensorflow as tf
import math
import numpy as np
from utils.eval import load_image_for_eval

def beam_search_evaluate(image_path, encoder, decoder, tokenizer, max_len, beam_width=3):
    """
    Genera un caption usando Beam Search.
    
    Args:
        beam_width (int): Cuántas secuencias candidatas mantener vivas (ej. 3, 5, 7).
                          Si es 1, equivale a la búsqueda Greedy.
    """
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    # 1. Procesar imagen y obtener características (solo se hace una vez)
    # Shape: (1, 224, 224, 3)
    img_tensor, _ = load_image_for_eval(image_path) 
    img_tensor = tf.expand_dims(img_tensor, 0)
    
    # Features shape: (1, 49, embedding_dim)
    features = encoder(img_tensor, training=False)

    # 2. Inicializar Decoder
    hidden = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([start_token], 0)

    # --- ESTRUCTURA DEL BEAM ---
    # Cada candidato es una tupla: (secuencia_actual, score_logaritmico, estado_hidden)
    # Score inicial 0.0 (log(1) = 0)
    sequences = [([start_token], 0.0, hidden)]

    # 3. Bucle paso a paso
    for i in range(max_len):
        all_candidates = []
        
        # Para cada secuencia candidata que tenemos viva...
        for seq, score, hidden_state in sequences:
            
            # Si la secuencia ya terminó con <end>, la guardamos tal cual sin expandir
            if seq[-1] == end_token:
                all_candidates.append((seq, score, hidden_state))
                continue
            
            # Preparamos la entrada para el decoder (última palabra de la secuencia)
            dec_input = tf.expand_dims([seq[-1]], 0)
            
            # Predecir siguiente palabra
            # Importante: Pasamos el hidden_state específico de ESTA secuencia candidata
            predictions, new_hidden, _ = decoder(dec_input, features, hidden_state)
            
            # predictions shape: (1, vocab_size) -> softmax probabilities
            predictions = tf.nn.softmax(predictions)
            
            # Obtenemos los top K candidatos para este paso
            # (Podríamos coger todo el vocabulario, pero top_k es más eficiente)
            top_k_probs, top_k_ids = tf.nn.top_k(predictions, k=beam_width)
            
            # Expandimos la secuencia actual con estas k posibilidades
            for k in range(beam_width):
                word_id = top_k_ids[0][k].numpy()
                prob = top_k_probs[0][k].numpy()
                
                # Sumamos logaritmos (porque multiplicar probabilidades pequeñas da underflow)
                # Log(prob) siempre es negativo, buscamos el valor más cercano a 0 (mayor probabilidad)
                new_score = score + math.log(prob + 1e-20) # 1e-20 para evitar log(0)
                
                new_seq = seq + [word_id]
                
                # Añadimos a la lista gigante de candidatos de este turno
                all_candidates.append((new_seq, new_score, new_hidden))
        
        # 4. Selección: Ordenamos todos los candidatos por score (de mayor a menor)
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        
        # Nos quedamos solo con los mejores 'beam_width' para el siguiente ciclo
        sequences = ordered[:beam_width]
        
        # (Opcional) Si el mejor candidato ya terminó, ¿podríamos parar? 
        # A veces se deja correr por si una frase más larga termina teniendo mejor score promedio.

    # 5. Resultado final: Tomamos la mejor secuencia (la primera de la lista ordenada)
    best_seq = sequences[0][0]
    
    # Convertir IDs a palabras
    result_caption = [tokenizer.index_word[i] for i in best_seq if i not in [start_token, end_token]]
    
    return result_caption

