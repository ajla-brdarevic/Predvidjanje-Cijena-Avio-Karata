# Glavna aplikacija za pokretanje predviđanja cijene leta na osnovu korisničkih unosa
# Potrebne biblioteke: Flask za web aplikaciju, joblib za učitavanje modela, pandas za rad sa podacima, numpy za matematičke operacije
# Aplikacija je dostupna na adresi http://127.0.0.1:5000

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)  # Kreira instancu Flask aplikacije

# Učitavanje prethodno sačuvanog Random Forest modela za predviđanje sa diska
model = joblib.load('model_rf.pkl')

# Funkcija koja mapira kategorijske vrijednosti na numeričke
# Korištena za unos podataka koji se kasnije koriste u modelu
def map_features(value, feature_mapping):
    try:
        return feature_mapping.get(value, 0)  # Ako vrijednost nije pronađena u mapiranju, vraća 0
    except Exception as e:
        return 0  # U slučaju greške, također vraća 0

# Ruta koja renderira početnu stranicu aplikacije (HTML forma za unos podataka)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Vraća HTML stranicu sa formom za unos

# Ruta za predviđanje cijene leta na osnovu podataka unesenih kroz formu
# Ovdje dolazi do obrade podataka sa forme, njihove konverzije u numeričke vrijednosti
# i predviđanja cijene putem modela
@app.route('/predict', methods=['POST'])
def predict():
    # Dohvatanje podataka sa forme korištenjem POST metode
    airline = request.form.get('airline')  # Podaci o aviokompaniji
    source_city = request.form.get('source_city')  # Izvorni grad
    departure_time = request.form.get('departure_time')  # Vrijeme polaska
    stops = request.form.get('stops')  # Broj presjedanja
    arrival_time = request.form.get('arrival_time')  # Vrijeme dolaska
    destination_city = request.form.get('destination_city')  # Odredišni grad
    class_type = request.form.get('class')  # Klasa leta (Economy ili Business)
    departure_date = request.form.get('departure_date')  # Datum polaska

    # Definicije mapiranja za svaku od kategorijskih vrijednosti koje će se koristiti u modelu
    # Svaka vrijednost je mapirana na jedinstveni broj
    airline_mapping = {'AirAsia': 0, 'Indigo': 1, 'GO_FIRST': 2, 'SpiceJet': 3, 'Air_India': 4, 'Vistara': 5}
    source_city_mapping = {'Delhi': 0, 'Hyderabad': 1, 'Bangalore': 2, 'Mumbai': 3, 'Kolkata': 4, 'Chennai': 5}
    departure_time_mapping = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5}
    arrival_time_mapping = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5}
    stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
    class_mapping = {'Economy': 0, 'Business': 1}
    destination_city_mapping = {'Delhi': 0, 'Hyderabad': 1, 'Mumbai': 2, 'Bangalore': 3, 'Chennai': 4, 'Kolkata': 5}

    # Mapiranje unesenih podataka na numeričke vrijednosti korištenjem prethodno definisanih mapiranja
    features = [
        map_features(airline, airline_mapping),  # Aviokompanija
        map_features(source_city, source_city_mapping),  # Izvorni grad
        map_features(departure_time, departure_time_mapping),  # Vrijeme polaska
        map_features(stops, stops_mapping),  # Broj presjedanja
        map_features(arrival_time, arrival_time_mapping),  # Vrijeme dolaska
        map_features(destination_city, destination_city_mapping),  # Odredišni grad
        map_features(class_type, class_mapping)  # Klasa leta
    ]

    # Pretvaranje liste u DataFrame za dalju obradu
    # Ovdje se kreira DataFrame sa kolonama koje odgovaraju atributima modela
    features_df = pd.DataFrame([features], columns=['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'])
    
    # Provjera i ispravka podataka
    # 'apply(pd.to_numeric, errors="coerce")' pokušava konvertirati sve podatke u numeričke, a 'NaN' vrijednosti se pretvaraju u 0
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    features_df = features_df.fillna(0)  # Zamjena svih 'NaN' vrijednosti sa 0

    # Predviđanje na osnovu modela koji je učitan prethodno
    prediction = model.predict(features_df)
    
    # Konverzija predviđene cijene u EUR (pretpostavljeni kurs 1 INR = 0.011 EUR)
    conversion_rate = 0.011
    prediction_in_eur = np.round(prediction * conversion_rate, 2)
    
    # Vraćanje rezultata u HTML stranicu (predviđena cijena u EUR)
    return render_template('index.html', prediction=prediction_in_eur[0])

# Pokretanje aplikacije u debug modu, što omogućava praćenje grešaka u kodu tokom razvoja
if __name__ == '__main__':
    app.run(debug=True)
