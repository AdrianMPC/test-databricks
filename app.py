import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from uuid import uuid4

# Fetch variables ME ENCANTA NO USAR ENV 
USER = "postgres.jecjnxivfbwsvjzsrymi" #os.getenv("user")
PASSWORD = "pinado132"# os.getenv("password")
HOST = "aws-1-us-east-2.pooler.supabase.com" #os.getenv("host")
PORT = 6543 #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
        pool_mode= transaction
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    st.write(str(e))



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    st.write(result)
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
   # Botón de predicción
    if st.button("Predecir Especie"):
        try:
            # Preparar datos
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)

            # Predicción
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            target_names = model_info['target_names']
            predicted_species = target_names[prediction]

            # Mostrar resultado
            st.success(f"Especie predicha: **{predicted_species}**")
            st.write(f"Confianza: **{max(probabilities):.1%}**")
            st.write("Probabilidades:")
            for species, prob in zip(target_names, probabilities):
                st.write(f"- {species}: {prob:.1%}")

            # Guardar en DB
            conn = psycopg2.connect(
                user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
            )
            conn.autocommit = True  # evita tener que hacer conn.commit()
            with conn, conn.cursor() as cur:
                row_id = str(uuid4())
                cur.execute("""
                    INSERT INTO modelo_data
                    (id, sepal_length, sepal_width, petal_length, petal_width, predicted_species)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (row_id, sepal_length, sepal_width, petal_length, petal_width, predicted_species))
            conn.close()
            st.info(f"✅ Registro guardado con id: `{row_id}`")

        except Exception as e:
            st.error(f"Error en predicción/guardado: {e}")