from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import os
#import json

from flask import Flask, request, render_template, jsonify

# Création de l'objet Flask
app = Flask(__name__) 

# Lancement du Débogueur
app.config["DEBUG"] = True 


def predict(image_path):
    model = load_model('model2.h5')
    # Charger l'image à partir du disque
    image = cv2.imread(image_path)
    # Convertir l'image BGR en RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Utiliser l'image pour détecter le visage 
    detector = MTCNN()
    # Détecter les visages sur l'image
    faces = detector.detect_faces(image)
    face_images = []
    face_locations = []
    noms = {'Muriel_Blanche': 0, 'barack obama': 1, 'corine': 2, 'dadju': 3, 
        'davido': 4, 'maitre_gims': 5, 'mandela': 6, 'paul_biya': 7, 'rihanna': 8}
    noms = {v: k for k, v in noms.items()}
    names = []
    if faces:
        for face in faces:
            x, y, w, h = face['box']
            # Extraire la région du visage à partir de l'image originale
            face_boundary = image[y:y + h, x:x + w]
            # Redimensionner les pixels à la taille du modèle
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize((224, 224))
            face_array = np.array(face_image)
            # Prétraiter l'image
            face_array = face_array / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            # Prédire la classe de chaque visage à l'aide du modèle
            prediction = model.predict(face_array)
            classe = np.argmax(prediction)
            proba = prediction[0][classe]
            print(proba)
            if proba >= 0.65:
                return noms[classe]
                #print(f"la personne sur l'image est {noms[classe]} avec une probabilité de {proba}")
                # Ajouter le nom de la personne prédite à la liste des noms
                #names.append(noms[classe])
                # Ajouter l'emplacement du visage à la liste des emplacements des visages
                #face_locations.append((y, x + w, y + h, x))
            else:
                return {'prediction': 'Désolée mais je ne connais pas cette personne'}
    else:
        return {'prediction': 'Aucun visage détecté veuillez mettre une autre image pour voir le rendu'}
    # Retourner la liste des noms et les emplacements des visages
    return noms[classe]

@app.route('/predict', methods=["POST"])
def prediction():
    # Vérifiez si l'utilisateur a sélectionné une image
    if 'image' not in request.files:
        return jsonify({'error': 'Veuillez sélectionner une image avant de soumettre le formulaire'})
    # Récupérez les données de l'image envoyée par l'utilisateur
    image = request.files['image']
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    # Enregistrez l'image sur le serveur
    image_path = os.path.join('static/uploads', image.filename)
    image.save(image_path)
    # Utilisez l'image pour faire des prédictions avec votre modèle
    resultat = predict(image_path)
    if type(resultat) == str:
        resultat = resultat.capitalize()
    #print(resultat)
    return render_template('resultat.html', image_path=image_path, prediction_result=resultat)

@app.route('/')
def index():
    return render_template('index.html')

    
app.run()
