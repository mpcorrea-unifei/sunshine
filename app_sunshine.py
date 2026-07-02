#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface web para estimativa de brilho solar com XGBoost.
Aceita entrada manual ou upload de arquivo CSV.
"""

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime
import io

# -------------------------------------------------------------------
# Funções astronômicas (mesmas do treinamento)
# -------------------------------------------------------------------
def solar_declination(doy):
    return 23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (284 + doy)))

def earth_sun_distance_correction(doy):
    return 1.0 + 0.033 * np.cos(np.deg2rad(360.0 * doy / 365.0))

def hourly_extraterrestrial_radiation(lat, doy, hour):
    G_sc = 1361.0
    lat_rad = np.deg2rad(lat)
    decl = np.deg2rad(solar_declination(doy))
    dr = earth_sun_distance_correction(doy)
    omega1 = np.deg2rad(15.0 * (hour - 0.5 - 12.0))
    omega2 = np.deg2rad(15.0 * (hour + 0.5 - 12.0))
    term = (np.sin(lat_rad) * np.sin(decl) * (omega2 - omega1) +
            np.cos(lat_rad) * np.cos(decl) * (np.sin(omega2) - np.sin(omega1)))
    return (12.0 / np.pi) * G_sc * dr * max(0.0, term)

def cos_zenith(lat, doy, hour):
    lat_rad = np.deg2rad(lat)
    decl = np.deg2rad(solar_declination(doy))
    omega = np.deg2rad(15.0 * (hour - 12.0))
    cz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(omega)
    return max(0.0, cz)

# -------------------------------------------------------------------
# Carregar modelo (cache para não recarregar)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model('modelo_xgb_inso.json')
    return model

# -------------------------------------------------------------------
# Função de previsão única
# -------------------------------------------------------------------
def predict_inso(model, lat, doy, hour, sw_top, lw_top, lw_bot,
                 t_air, humidity, pressure, wind, sw_bot, rain):
    G0h = hourly_extraterrestrial_radiation(lat, doy, hour)
    if G0h < 1.0:
        return 0.0
    Kt = (sw_top * 3600.0) / G0h
    cz = cos_zenith(lat, doy, hour)
    features = np.array([[Kt, cz, t_air, humidity, pressure, wind,
                          lw_top, lw_bot, sw_bot, rain]])
    inso = model.predict(features)[0]
    return float(np.clip(inso, 0.0, 1.0))

# -------------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------------
st.set_page_config(page_title='Estimador de Brilho Solar', layout='wide')
st.title('☀️ Estimador de Fração de Insolação Horária')
st.markdown('Modelo XGBoost treinado com dados de Itajubá-MG (inclui radiação de onda longa).')

model = load_model()

# Barra lateral para escolha do modo
modo = st.sidebar.radio('Modo de entrada', ['Entrada manual', 'Upload de arquivo CSV'])

if modo == 'Entrada manual':
    st.header('Insira os dados da hora desejada')
    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input('Latitude (graus)', value=-22.4269, help='Negativa para sul')
        data = st.date_input('Data', value=datetime(2023, 7, 1))
        hora = st.selectbox('Hora local', list(range(24)), index=12)
    with col2:
        sw_top = st.number_input('SWTop (W/m²)', value=800.0, help='Irradiância global descendente')
        sw_bot = st.number_input('SWBot (W/m²)', value=80.0)
        lw_top = st.number_input('LWTop (W/m²)', value=350.0)
        lw_bot = st.number_input('LWBot (W/m²)', value=410.0)
    with col3:
        t_air = st.number_input('Temperatura (°C)', value=25.0)
        humidity = st.number_input('Umidade (%)', value=65.0, min_value=0.0, max_value=100.0)
        pressure = st.number_input('Pressão (hPa)', value=1015.0)
        wind = st.number_input('Vento (m/s)', value=2.0)
        rain = st.number_input('Precipitação (mm)', value=0.0)

    doy = data.timetuple().tm_yday
    if st.button('Estimar'):
        inso = predict_inso(model, lat, doy, hora,
                            sw_top, lw_top, lw_bot,
                            t_air, humidity, pressure, wind,
                            sw_bot, rain)
        st.success(f'**Fração de brilho solar estimada:** {inso:.4f}')
        st.progress(float(inso))

else:
    st.header('Upload de arquivo CSV')
    st.markdown("""
    O arquivo deve conter as colunas (com ou sem cabeçalho):
    `lat, doy, hour, SWTop, LWTop, LWBot, T_air, Humidity, Pressure, Wind, SWBot, Rain`
    Se o arquivo tiver `date` (DD/MM/YYYY) em vez de `doy`, a conversão será automática.
    """)
    arquivo = st.file_uploader('Selecione o arquivo CSV', type=['csv', 'txt'])
    if arquivo:
        try:
            df = pd.read_csv(arquivo, sep=None, engine='python')
        except:
            df = pd.read_csv(arquivo, sep=';')
        st.write('Pré-visualização:', df.head())

        # Converter date para doy se necessário
        if 'date' in df.columns and 'doy' not in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            df['doy'] = df['date'].dt.dayofyear

        required = ['lat', 'doy', 'hour', 'SWTop', 'LWTop', 'LWBot',
                    'T_air', 'Humidity', 'Pressure', 'Wind', 'SWBot', 'Rain']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f'Colunas faltando: {missing}')
        else:
            df = df.fillna(0)
            previsoes = []
            for _, row in df.iterrows():
                inso = predict_inso(model,
                                    row['lat'], int(row['doy']), int(row['hour']),
                                    row['SWTop'], row['LWTop'], row['LWBot'],
                                    row['T_air'], row['Humidity'], row['Pressure'],
                                    row['Wind'], row['SWBot'], row['Rain'])
                previsoes.append(inso)
            df['inso_predito'] = previsoes
            st.success('Estimativas calculadas!')
            st.dataframe(df[['lat', 'doy', 'hour', 'SWTop', 'inso_predito']])

            # Download
            csv = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button('Baixar resultados', csv,
                               'resultado.csv', 'text/csv')