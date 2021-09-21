#!/usr/bin/env python

"""
Este programa permite detectar patos dentro del simulador mediante el análisis
de color.
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

    return actions.get(key, np.array([0.0, 0.0]))


if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='loop_pedestrians')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            seed = args.seed,
            map_name = args.map_name,
            draw_curve = args.draw_curve,
            draw_bbox = args.draw_bbox,
            domain_rand = args.domain_rand,
            frame_skip = args.frame_skip,
            distortion = args.distortion,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    # Parametros para el detector de patos
    # Se debe encontrar el rango apropiado
    lower_yellow = np.array([22., 60., 170.])
    upper_yellow = np.array([55., 255., 255.])
    min_area = 2500

    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(30)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        action = mov_duckiebot(key)
        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)
        # obs consiste en un imagen RGB de 640 x 480 x 3

        # done significa que el Duckiebot chocó con un objeto o se salió del camino
        if done:
            print('done!')
            # En ese caso se reinicia el simulador
            env.reset()

        ### CÓDIGO DE DETECCIÓN POR COLOR ###

        #Transformar imagen a espacio HSV
        img_outHSV=cv2.cvtColor(obs,cv2.COLOR_RGB2HSV)
        img_input=cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # Filtrar colores de la imagen en el rango utilizando
        mask=cv2.inRange(img_outHSV,lower_yellow,upper_yellow)
        

        # Bitwise-AND entre máscara (mask) y original (obs) para visualizar lo filtrado
        bitwise=cv2.bitwise_and(obs,obs,mask=mask)
        

      # Se define kernel para operaciones morfológicas
        kernel = np.ones((3,3),np.uint8)

        # Aplicar operaciones morfológicas para eliminar ruido
        # Esto corresponde a hacer un Opening
        # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
       
        #Operacion morfologica erode
        erode=cv2.erode (bitwise,kernel,iterations=1)

        #Operacion morfologica dilate

        dilate=cv2.dilate (erode,kernel,iterations=1)
        gray=cv2.cvtColor(dilate,cv2.COLOR_RGB2GRAY) #necesario para el findcountours

        # Busca contornos de blobs
        # https://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html
        
        contours, hierarchy=cv2.findContours(gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
               
       # Iterar sobre contornos y dibujar bounding box de los patos
        for cnt in contours:
            # Obtener rectangulo que bordea un contorno
            x,y,w,h=cv2.boundingRect(cnt)
            AREA=w*h
            #Filtrar por area minima
            if AREA > min_area: # DEFINIR AREA
                #Dibujar rectangulo en el frame original
                cv2.rectangle(img_input,(x,y),(x+100,y+100),(0,255,0),2)

        # Se muestra en una ventana llamada "patos" la observación del simulador
        # con los bounding boxes dibujados

        cv2.imshow('patos',img_input )
        # Se muestra en una ventana llamada "filtrado" la imagen filtrada
        cv2.imshow('filtrado',cv2.bitwise_and(img_input,img_input,mask=gray) )

    #Se cierra el environment y termina el programa
env.close()
