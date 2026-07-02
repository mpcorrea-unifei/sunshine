#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface web para estimativa de brilho solar.
Comparação entre XGBoost e Lasso, com e sem radiação de onda longa.
Resultado em minutos de sol por hora.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from xgboost import XGBRegressor
from datetime import datetime

# -------------------------------------------------------------------
# Funções astronômicas
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
# Carregamento dos modelos
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    
    # XGBoost com LW
    if os.path.exists('modelo_xgb_inso.json'):
        xgb_lw = XGBRegressor()
        xgb_lw.load_model('modelo_xgb_inso.json')
        models['XGBoost (com LW)'] = {'type': 'xgb', 'model': xgb_lw, 'lw': True}
    
    # XGBoost sem LW
    if os.path.exists('modelo_xgb_inso_semLW.json'):
        xgb_nlw = XGBRegressor()
        xgb_nlw.load_model('modelo_xgb_inso_semLW.json')
        models['XGBoost (sem LW)'] = {'type': 'xgb', 'model': xgb_nlw, 'lw': False}
    
    # Lasso com LW
    if os.path.exists('lasso_comLW.json'):
        with open('lasso_comLW.json') as f:
            lasso_lw = json.load(f)
        models['Lasso (com LW)'] = {'type': 'lasso', 'coef': lasso_lw, 'lw': True}
    
    # Lasso sem LW
    if os.path.exists('lasso_semLW.json'):
        with open('lasso_semLW.json') as f:
            lasso_nlw = json.load(f)
        models['Lasso (sem LW)'] = {'type': 'lasso', 'coef': lasso_nlw, 'lw': False}
    
    return models

# -------------------------------------------------------------------
# Funções de predição
# -------------------------------------------------------------------
def predict_xgb(model_obj, lat, doy, hour, sw_top, lw_top, lw_bot,
                t_air, humidity, pressure, wind, sw_bot, rain, use_lw):
    G0h = hourly_extraterrestrial_radiation(lat, doy, hour)
    if G0h < 1.0:
        return 0.0
    Kt = (sw_top * 3600.0) / G0h
    cz = cos_zenith(lat, doy, hour)
    if use_lw:
        features = [Kt, cz, t_air, humidity, pressure, wind,
                    lw_top, lw_bot, sw_bot, rain]
    else:
        features = [Kt, cz, t_air, humidity, pressure, wind, sw_bot, rain]
    inso = model_obj.predict(np.array([features]))[0]
    return float(np.clip(inso, 0.0, 1.0))

def predict_lasso(coef_data, lat, doy, hour, sw_top, lw_top, lw_bot,
                  t_air, humidity, pressure, wind, sw_bot, rain, use_lw):
    G0h = hourly_extraterrestrial_radiation(lat, doy, hour)
    if G0h < 1.0:
        return 0.0
    Kt = (sw_top * 3600.0) / G0h
    cz = cos_zenith(lat, doy, hour)
    # Dicionário com todas as variáveis possíveis
    vars_all = {
        'Kt': Kt, 'cos_zen': cz, 'T_air': t_air, 'Humidity': humidity,
        'Pressure': pressure, 'Wind': wind,
        'LWTop': lw_top, 'LWBot': lw_bot, 'SWBot': sw_bot, 'Rain': rain
    }
    intercept = coef_data['intercept']
    coef = coef_data['coef']
    inso = intercept
    for var, val in coef.items():
        if var in vars_all:
            inso += val * vars_all[var]
    return float(np.clip(inso, 0.0, 1.0))

def predict_all(models, lat, doy, hour, sw_top, lw_top, lw_bot,
                t_air, humidity, pressure, wind, sw_bot, rain):
    results = {}
    for name, info in models.items():
        if info['type'] == 'xgb':
            pred = predict_xgb(info['model'], lat, doy, hour,
                               sw_top, lw_top, lw_bot,
                               t_air, humidity, pressure, wind,
                               sw_bot, rain, info['lw'])
        else:  # lasso
            pred = predict_lasso(info['coef'], lat, doy, hour,
                                 sw_top, lw_top, lw_bot,
                                 t_air, humidity, pressure, wind,
                                 sw_bot, rain, info['lw'])
        # Converter para minutos de sol na hora
        results[name] = round(pred * 60, 2)
    return results

# -------------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------------
st.set_page_config(page_title='Estimador de Brilho Solar', layout='wide')
st.title('☀️ Estimador de Fração de Insolação Horária')
st.markdown('Comparação entre modelos XGBoost e Lasso (com e sem radiação de onda longa).')

models = load_models()
if not models:
    st.error('Nenhum modelo encontrado. Verifique os arquivos na pasta do aplicativo.')
    st.stop()

# Barra lateral: seleção de modelos
st.sidebar.header('Modelos a utilizar')
model_names = list(models.keys())
selected_models = st.sidebar.multiselect(
    'Selecione um ou mais modelos:',
    model_names,
    default=model_names[:1]  # primeiro disponível como padrão
)

# Modo de entrada
modo = st.sidebar.radio('Modo de entrada', ['Entrada manual', 'Upload de arquivo CSV'])

if modo == 'Entrada manual':
    st.header('Insira os dados da hora desejada')
    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input('Latitude (graus)', value=-22.4269)
        data = st.date_input('Data', value=datetime(2023, 7, 1))
        hora = st.selectbox('Hora local', list(range(24)), index=12)
    with col2:
        sw_top = st.number_input('SWTop (W/m²)', value=800.0)
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
        if not selected_models:
            st.warning('Selecione pelo menos um modelo na barra lateral.')
        else:
            res = predict_all({k: models[k] for k in selected_models},
                              lat, doy, hora,
                              sw_top, lw_top, lw_bot,
                              t_air, humidity, pressure, wind,
                              sw_bot, rain)
            st.subheader('Resultados (minutos de sol na hora)')
            df_res = pd.DataFrame(res.items(), columns=['Modelo', 'Minutos de sol'])
            st.dataframe(df_res)
            # Mostrar também um gráfico de barras comparativo
            st.bar_chart(df_res.set_index('Modelo'))

else:  # Upload de arquivo
    st.header('Upload de arquivo CSV')
    st.markdown("""
    O arquivo deve conter as colunas (com ou sem cabeçalho):
    `lat, doy, hour, SWTop, SWBot, LWTop, LWBot, T_air, Humidity, Pressure, Wind, Rain`
    Se houver `date` (DD/MM/YYYY) em vez de `doy`, a conversão será automática.
    """)
    arquivo = st.file_uploader('Selecione o arquivo CSV', type=['csv', 'txt'])
    if arquivo and selected_models:
        try:
            df = pd.read_csv(arquivo, sep=None, engine='python')
        except:
            df = pd.read_csv(arquivo, sep=';')
        st.write('Pré-visualização:', df.head())

        if 'date' in df.columns and 'doy' not in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            df['doy'] = df['date'].dt.dayofyear

        required = ['lat', 'doy', 'hour', 'SWTop', 'SWBot', 'LWTop', 'LWBot',
                    'T_air', 'Humidity', 'Pressure', 'Wind', 'Rain']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f'Colunas faltando: {missing}')
        else:
            df = df.fillna(0)
            progress = st.progress(0)
            total = len(df)
            for i, (_, row) in enumerate(df.iterrows()):
                res = predict_all({k: models[k] for k in selected_models},
                                  row['lat'], int(row['doy']), int(row['hour']),
                                  row['SWTop'], row['LWTop'], row['LWBot'],
                                  row['T_air'], row['Humidity'], row['Pressure'],
                                  row['Wind'], row['SWBot'], row['Rain'])
                for model_name, minutos in res.items():
                    df.at[i, f'min_sol_{model_name}'] = minutos
                progress.progress((i+1)/total)
            st.success('Estimativas calculadas!')
            colunas_exibir = ['lat', 'doy', 'hour', 'SWTop']
            colunas_exibir += [f'min_sol_{m}' for m in selected_models]
            st.dataframe(df[colunas_exibir])

            csv = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button('Baixar resultados', csv, 'resultado.csv', 'text/csv')