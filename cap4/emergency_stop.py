#!/usr/bin/env python

"""
Este programa implementa un freno de emergencia para evitar accidentes en Duckietown.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2

def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    action = actions.get(key, np.array([0.0, 0.0]))
    return action

def det_duckie(obs):
    ### DETECTOR HECHO EN LA MISIÓN ANTERIOR
    # Parametros para el detector de patos
    # Se debe encontrar el rango apropiado
    lower_yellow = np.array([22., 60., 170.])
    upper_yellow = np.array([55., 255., 255.])
    min_area = 2500
    
    #Transformar imagen a espacio HSV
    img_outHSV=cv2.cvtColor(obs,cv2.COLOR_RGB2HSV)
    img_input=cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # Filtrar colores de la imagen en el rango utilizando
    mask=cv2.inRange(img_outHSV,lower_yellow,upper_yellow)
        

    # Bitwise-AND entre máscara (mask) y original (obs) para visualizar lo filtrado
    bitwise=cv2.bitwise_and(obs,obs,mask=mask)
        

    # Se define kernel para operaciones morfológicas
    kernel = np.ones((3,3),np.uint8)
     #Operacion morfologica erode
    erode=cv2.erode (bitwise,kernel,iterations=1)

    #Operacion morfologica dilate
    dilate=cv2.dilate (erode,kernel,iterations=1)
    gray=cv2.cvtColor(dilate,cv2.COLOR_RGB2GRAY) #necesario para el findcountours
    
    # Busca contornos de blobs
    contours, hierarchy=cv2.findContours(gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    dets = list()

    for cnt in contours:
        # Obtener rectangulo que bordea un contorno
        x,y,w,h=cv2.boundingRect(cnt)
        AREA=w*h

        if AREA > min_area:
            # En lugar de dibujar, se agrega a la lista
            dets.append((x,y,w,h))

    return dets

def draw_dets(obs, dets):
    for d in dets:
        x1, y1 = d[0], d[1]
        x2 = x1 + d[2]
        y2 = y1 + d[3]
        cv2.rectangle(obs, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 3)

    return obs

def red_alert(obs):
    red_img = np.zeros(obs.shape, dtype = np.uint8)
    red_img[:,:,0] = 255
    blend = cv2.addWeighted(obs, 0.5, red_img, 0.5, 0)

    return blend

if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='free.yaml')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            map_name = args.map_name,
            domain_rand = False,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    # Inicialmente no hay alerta
    alert = False

    # Posición del pato en el mapa (fija)
    duck_pos = np.array([2,0,2])

    # Constante que se debe calcular
    
    #mediciones
    #a continuación las mediciones tomadas para p y dr
    lista_p=np.array([55, 58, 60, 60, 63, 66, 70, 73, 77, 77, 81, 87, 94, 100, 109, 117, 129, 143, 160, 181, 210])
    lista_dr=np.array([1.0436371367866557, 1.003751241240131, 0.9638748027114095, 0.9638748027114095, 0.9238984224387755, 0.8839241792528123, 0.8439523768262387, 0.8039833792023049, 0.7640176265744507, 0.7640176265744507, 0.7240179873495745, 0.6840183903192955, 0.6440188433456523, 0.6040193563732403, 0.564019942167664, 0.524020617392185, 0.4840214042182003, 0.44402233280616654, 0.40402344526017525, 0.36402480218959465, 0.3240264941304109])
    
    #calculando f a partir de las mediciones
    lista_f=(lista_p*lista_dr)/0.08
    f= lista_f.mean() # f=755.7431330677339 px

   
    while True:


        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(0)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        # Se define la acción dada la tecla presionada
        action = mov_duckiebot(key)

        # Si hay alerta evitar que el Duckiebot avance
        if alert:
            action=np.array([-1,0])

        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)
        print()


        # Detección de patos, retorna lista de detecciones
        dets=det_duckie(obs)

        # Dibuja las detecciones
        draw_dets(obs, dets)

        # Obtener posición del duckiebot
        dbot_pos = env.cur_pos
        # Calcular distancia real entre posición del duckiebot y pato
        # esta distancia se utiliza para calcular la constante
        
        dist = np.sqrt ((dbot_pos[0]-duck_pos[0])**2+(dbot_pos[2]-duck_pos[2])**2)

        # La alerta se desactiva (opción por defecto)
        alert = False
        
        for d in dets:
            # Alto de la detección en pixeles
            p = dets[0][3]
            # La aproximación se calcula según la fórmula mostrada en la capacitación
            d_aprox = f*0.08/p


            # Muestra información relevante
            #print('Da:', d_aprox)
            print('p:', p)
            print('Dr:', dist)
            print('da',d_aprox)
            
            
            # Si la distancia es muy pequeña activa alerta (activa pantalla roja antes de detener el bot)
            if d_aprox < 0.6:
                # Muestra ventana en rojo
                obs=red_alert(obs)
            if d_aprox < 0.45:
                # Activar alarma
                alert=True


        # Se muestra en una ventana llamada "patos" la observación del simulador
        cv2.imshow('patos', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # Se cierra el environment y termina el programa
    env.close()
