# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:49:35 2019

@author: crist
"""
# Librerías de siempre
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

from statsmodels.formula.api import ols
from scipy import stats

def formatear_base(db):
    df = db.copy()
    if 'fecha' in df.columns:
        df = df.sort_values(by=['Torneo','Round'], ascending = [True,True]).reset_index(drop=True)
    equipos = df['Local'].unique()
    equipos = pd.DataFrame(equipos, columns=['equipo'])
    equipos['i'] = equipos.index
    df = pd.merge(df, equipos, left_on='Local', right_on='equipo', how='left')
    df = df.rename(columns = {'i': 'i_local'}).drop('equipo', 1)
    df = pd.merge(df, equipos, left_on='Visita', right_on='equipo', how='left')
    df = df.rename(columns = {'i': 'i_visita'}).drop('equipo', 1)
    df = df.replace({'. Round' : ''}, regex = True)
    df['Round'] = df['Round'].astype(int)
    df['goles L'] = df['goles L'].astype(int)
    df['goles V'] = df['goles V'].astype(int)
    return df

def tabla_final_torneo(db):
    df = db.copy()
    if 'i_local' not in df.columns:
        df = formatear_base(df)
    tabla = df[['Local','i_local']].drop_duplicates()
    tabla = tabla.set_index(['i_local'])
    tabla.columns = ['equipo']
    conditions = [
            (df['goles L'] > df['goles V']),
            (df['goles L'] < df['goles V'])]
    choices = ['local', 'visita']
    df = df.join(pd.get_dummies(np.select(conditions, choices, default = 'empate')))
    ghome = df.groupby('i_local')
    gaway = df.groupby('i_visita')
    df_home = pd.DataFrame({'wins_h': ghome['local'].sum(),
                            'draws_h': ghome['empate'].sum(),
                            'losses_h': ghome['visita'].sum(),
                            'gf_h': ghome['goles L'].sum(),
                            'ga_h': ghome['goles V'].sum(),
                            'gd_h': ghome['goles L'].sum() - ghome['goles V'].sum()})
    df_away = pd.DataFrame({'wins_a': gaway['visita'].sum(),
                            'draws_a': gaway['empate'].sum(),
                            'losses_a': gaway['local'].sum(),
                            'gf_a': gaway['goles V'].sum(),
                            'ga_a': gaway['goles L'].sum(),
                            'gd_a': gaway['goles V'].sum() - gaway['goles L'].sum()})
    tabla = tabla.join(df_home, how='left').join(df_away,how = 'left').fillna(0)
    tabla['wins'] = tabla.wins_h + tabla.wins_a
    tabla['draws'] = tabla.draws_h + tabla.draws_a
    tabla['losses'] = tabla.losses_h + tabla.losses_a
    tabla['gf'] = tabla.gf_h +tabla.gf_a
    tabla['ga'] = tabla.ga_h +tabla.ga_a
    tabla['gd'] = tabla.gd_h +tabla.gd_a
    tabla['points'] = (tabla['wins']*3 + tabla['draws']).astype(int)
    tabla = tabla.sort_values(by=['points','gd'], ascending = False).reset_index(drop=True)
    tabla['position'] = (tabla.index + 1).astype(int)
    return tabla[['equipo','wins','draws','losses','gf','ga','gd','points','position']]

def tabla_final_local(db):
    df = db.copy()
    if 'i_local' not in df.columns:
        df = formatear_base(df)
    tabla = df[['Local','i_local']].drop_duplicates()
    tabla = tabla.set_index(['i_local'])
    tabla.columns = ['equipo']
    conditions = [
            (df['goles L'] > df['goles V']),
            (df['goles L'] < df['goles V'])]
    choices = ['local', 'visita']
    df = df.join(pd.get_dummies(np.select(conditions, choices, default = 'empate')))
    ghome = df.groupby('i_local')
    df_home = pd.DataFrame({'wins': ghome['local'].sum(),
                            'draws': ghome['empate'].sum(),
                            'losses': ghome['visita'].sum(),
                            'gf': ghome['goles L'].sum(),
                            'ga': ghome['goles V'].sum(),
                            'gd': ghome['goles L'].sum() - ghome['goles V'].sum()})
    tabla = tabla.join(df_home, how='left').fillna(0)
    tabla['points'] = (tabla['wins']*3 + tabla['draws']).astype(int)
    #Normalizar puntos según cantidad de partidos jugados de local
    tabla['norm_points'] = tabla['points']/(tabla['wins'] + tabla['draws'] + tabla['losses']) 
    tabla = tabla.sort_values(by=['norm_points','points','gd'], ascending = False).reset_index(drop=True)
    tabla['position'] = (tabla.index + 1).astype(int)
    return tabla[['equipo','wins','draws','losses','gf','ga','gd','points','position']]

def tabla_final_visita(db):
    df = db.copy()
    if 'i_visita' not in df.columns:
        df = formatear_base(df)
    tabla = df[['Visita','i_visita']].drop_duplicates()
    tabla = tabla.set_index(['i_visita'])
    tabla.columns = ['equipo']
    conditions = [
            (df['goles L'] > df['goles V']),
            (df['goles L'] < df['goles V'])]
    choices = ['local', 'visita']
    df = df.join(pd.get_dummies(np.select(conditions, choices, default = 'empate')))
    gaway = df.groupby('i_visita')
    df_away = pd.DataFrame({'wins': gaway['visita'].sum(),
                            'draws': gaway['empate'].sum(),
                            'losses': gaway['local'].sum(),
                            'gf': gaway['goles V'].sum(),
                            'ga': gaway['goles L'].sum(),
                            'gd': gaway['goles V'].sum() - gaway['goles L'].sum()})
    tabla = tabla.join(df_away, how='left').fillna(0)
    tabla['points'] = (tabla['wins']*3 + tabla['draws']).astype(int)
    #Normalizar puntos según cantidad de partidos jugados de local
    tabla['norm_points'] = tabla['points']/(tabla['wins'] + tabla['draws'] + tabla['losses']) 
    tabla = tabla.sort_values(by=['norm_points','points','gd'], ascending = False).reset_index(drop=True)
    tabla['position'] = (tabla.index + 1).astype(int)
    return tabla[['equipo','wins','draws','losses','gf','ga','gd','points','position']]


def bases_pre_indicadores(df,torneo_actual,torneo_anterior, pto_corte = 5, replace_releg = False):
    """
    Crea las columnas que se agregarán para definir los indicadores de interés
    """
    t_ant = tabla_final_torneo(df[df['Torneo'] == torneo_anterior])
    t_ant_local = tabla_final_local(df[df['Torneo'] == torneo_anterior])
    t_ant_visita = tabla_final_visita(df[df['Torneo'] == torneo_anterior])
    df_actual = formatear_base(df[df['Torneo'] == torneo_actual])[['Local','Visita','goles L','goles V','Torneo','Round']].sort_values(by=['Round'], ascending = True).reset_index(drop=True)
    pto_corte = max(2,pto_corte)
    eq_act = df_actual['Local'].drop_duplicates().tolist()
    eq_ant_l =  t_ant_local['equipo'].drop_duplicates().tolist()
    eq_ant_v =  t_ant_visita['equipo'].drop_duplicates().tolist()
    eq_ant = t_ant['equipo'].drop_duplicates().tolist() #esta ya está ordenada por posición
    eq_correc = [i for i in eq_ant if i in eq_act] + [j for j in eq_act if j not in eq_ant]
    eq_correc_l = [i for i in eq_ant_l if i in eq_act] + [j for j in eq_act if j not in eq_ant_l]
    eq_correc_v = [i for i in eq_ant_v if i in eq_act] + [j for j in eq_act if j not in eq_ant_v]
    # reemplazar o eliminar equipos descendidos
    eq_act = df_actual['Local'].drop_duplicates().tolist()
    eq_ant_l =  t_ant_local['equipo'].drop_duplicates().tolist()
    eq_ant_v =  t_ant_visita['equipo'].drop_duplicates().tolist()
    eq_ant = t_ant['equipo'].drop_duplicates().tolist() #esta ya está ordenada por posición
    eq_correc = [i for i in eq_ant if i in eq_act] + [j for j in eq_act if j not in eq_ant]
    eq_correc_l = [i for i in eq_ant_l if i in eq_act] + [j for j in eq_act if j not in eq_ant_l]
    eq_correc_v = [i for i in eq_ant_v if i in eq_act] + [j for j in eq_act if j not in eq_ant_v]
    if replace_releg:
        t_ant = pd.DataFrame({'equipo': eq_correc,
                              'position': [i+1 for i in range(len(eq_correc))]})
        t_ant_local = pd.DataFrame({'equipo': eq_correc_l,
                              'position': [i+1 for i in range(len(eq_correc_l))]})
        t_ant_visita = pd.DataFrame({'equipo': eq_correc_v,
                              'position': [i+1 for i in range(len(eq_correc_v))]})
    else:    
        t_ant_local = t_ant_local[t_ant_local['equipo'].isin(df[(df['Torneo'] == torneo_actual)]['Local'].drop_duplicates().tolist())].reset_index(drop=True)
        t_ant_local['position'] = t_ant_local.index + 1
        t_ant_visita = t_ant_visita[t_ant_visita['equipo'].isin(df[(df['Torneo'] == torneo_actual)]['Local'].drop_duplicates().tolist())].reset_index(drop=True)
        t_ant_visita['position'] = t_ant_visita.index + 1
        posicion_nuevos = t_ant_local['position'].max() + 1
        position = [min(i+1,posicion_nuevos) for i in range(len(eq_correc))]
        t_ant = pd.DataFrame({'equipo': eq_correc,
                          'position': position})
        t_ant_local = pd.DataFrame({'equipo': eq_correc_l,
                              'position': position})
        t_ant_visita = pd.DataFrame({'equipo': eq_correc_v,
                              'position': position})
#    if replace_releg:
#        # En este caso considera que los ascendidos ocupan las ultimas posiciones según orden alfabético
#        position = [i+1 for i in range(len(eq_correc))]
#    else:
#        # En este caso considera que los ascendidos ocupan la misma posicion
#        t_ant_local = t_ant_local[t_ant_local['equipo'].isin(df[(df['Torneo'] == torneo_actual)]['Local'].drop_duplicates().tolist())].reset_index(drop=True)
#        t_ant_local['position'] = t_ant_local.index + 1
#        posicion_nuevos = t_ant_local['position'].max() + 1
#        position = [max(i+1,posicion_nuevos) for i in range(len(eq_correc))]
#    t_ant = pd.DataFrame({'equipo': eq_correc,
#                      'position': position})
#    t_ant_local = pd.DataFrame({'equipo': eq_correc_l,
#                          'position': position})
#    t_ant_visita = pd.DataFrame({'equipo': eq_correc_v,
#                          'position': position})
    t_ant_ultimo = t_ant['position'].astype(int).max()
    pto_corte = min(max(2,pto_corte),t_ant_ultimo)
    equipos_5p = t_ant[t_ant['position'] <= pto_corte]['equipo'].values.tolist()
    local_5p = t_ant_local[t_ant_local['position'] <= pto_corte]['equipo'].values.tolist()
    visita_5p = t_ant_visita[t_ant_visita['position'] <= pto_corte]['equipo'].values.tolist()
    equipos_5u = t_ant[t_ant['position'] >= t_ant_ultimo - pto_corte + 1]['equipo'].values.tolist()
    local_5u = t_ant_local[t_ant_local['position'] >= t_ant_ultimo - pto_corte + 1]['equipo'].values.tolist() 
    visita_5u = t_ant_visita[t_ant_visita['position'] >= t_ant_ultimo - pto_corte + 1]['equipo'].values.tolist()
    
#    df_p5 = df_actual[df_actual['Round'] <= ronda_corte].sort_values(by=['Round'], ascending = True).reset_index(drop=True)
#    df_u5 = df_actual[df_actual['Round'] >= t_act_maxround - ronda_corte + 1].sort_values(by=['Round'], ascending = True).reset_index(drop=True)
    
    dict_eq_pos_gral = dict(zip(t_ant['equipo'].values, t_ant['position'].values))
    dict_eq_pos_local = dict(zip(t_ant_local['equipo'].values, t_ant_local['position'].values))
    dict_eq_pos_visita = dict(zip(t_ant_visita['equipo'].values, t_ant_visita['position'].values))

    
    """
    SIGNIFICADO VARIABLES
    
    
    f_gral_visita: 1 si el equipo visitante es fácil según la tabla general
    del torneo anterior, 0 si no
    d_gral_visita: 1 si el equipo visitante es difícil según la tabla general
    del torneo anterior, 0 si no
    f_gral_local: 1 si el equipo local es fácil según la tabla general
    del torneo anterior, 0 si no
    d_gral_local: 1 si el equipo local es difícil según la tabla general
    del torneo anterior, 0 si no
    
    f_visita: 1 si el equipo visitante es fácil jugando de visita según
    la tabla de posiciones de visita, 0 si no
    d_visita: 1 si el equipo visitante es difícil jugando de visita según
    la tabla de posiciones de visita, 0 si no
    f_local: 1 si equipo local es fácil jugando de local según
    la tabla de posiciones de local, 0 si no
    d_local: 1 si el equipo local es difícil jugando de local según
    la tabla de posiciones de local, 0 si no
    
    f_..._corr es lo mismo, salvo que se hacre la corrección de pertenecer
    al mismo grupo
    
    posicion_gral_visita: posición del equipo visitante según la tabla general del
    torneo anterior
    posicion_gral_local: posición del equipo local según la tabla general del
    torneo anterior
    posicion_visita: posición del equipo visitante según la tabla de visita del
    torneo anterior
    posicion_local: posición del equipo local según la tabla de local del
    torneo anterior
    
    ronda_f_gral_visita: si el equipo visitante es fácil según la tabla general,
    toma valor igual a la ronda. Si no lo es, toma np.nan
    ronda_d_gral_visita: si el equipo visitante es difícil según la tabla general,
    toma valor igual a la ronda. Si no lo es, toma np.nan    
    ronda_f_gral_local: si el equipo local es fácil según la tabla general,
    toma valor igual a la ronda. Si no lo es, toma np.nan
    ronda_d_gral_local: si el equipo local es difícil según la tabla general,
    toma valor igual a la ronda. Si no lo es, toma np.nan
    
    ronda_f_visita: si el equipo visitante es fácil según la tabla de visita,
    toma valor igual a la ronda. Si no lo es, toma np.nan
    ronda_d_visita: si el equipo visitante es difícil según la tabla de visita,
    toma valor igual a la ronda. Si no lo es, toma np.nan    
    ronda_f_local: si el equipo local es fácil según la tabla de local,
    toma valor igual a la ronda. Si no lo es, toma np.nan
    ronda_d_local: si el equipo local es difícil según la tabla de local,
    toma valor igual a la ronda. Si no lo es, toma np.nan
    

    """
    
    df_actual['f_gral_visita'] = np.where(df_actual['Visita'].isin(equipos_5u), 1, 0)
    df_actual['d_gral_visita'] = np.where(df_actual['Visita'].isin(equipos_5p), 1, 0)
    df_actual['f_gral_local'] = np.where(df_actual['Local'].isin(equipos_5u), 1, 0)
    df_actual['d_gral_local'] = np.where(df_actual['Local'].isin(equipos_5p), 1, 0)    
    
    df_actual['f_visita'] = np.where(df_actual['Visita'].isin(visita_5u), 1, 0)
    df_actual['d_visita'] = np.where(df_actual['Visita'].isin(visita_5p), 1, 0)
    df_actual['f_local'] = np.where(df_actual['Local'].isin(local_5u), 1, 0)
    df_actual['d_local'] = np.where(df_actual['Local'].isin(visita_5p), 1, 0)
    
    df_actual['f_gral_visita_corr'] =  df_actual['f_gral_visita']*np.where(df_actual['Local'].isin(equipos_5u),pto_corte/(pto_corte - 1) , 1)
    df_actual['d_gral_visita_corr'] =  df_actual['d_gral_visita']*np.where(df_actual['Local'].isin(equipos_5p),pto_corte/(pto_corte - 1) , 1)
    df_actual['f_gral_local_corr'] =  df_actual['f_gral_local']*np.where(df_actual['Visita'].isin(equipos_5u),pto_corte/(pto_corte - 1) , 1)
    df_actual['d_gral_local_corr'] =  df_actual['d_gral_local']*np.where(df_actual['Visita'].isin(equipos_5p),pto_corte/(pto_corte - 1) , 1)
    
    df_actual['f_visita_corr'] =  df_actual['f_visita']*np.where(df_actual['Local'].isin(visita_5u),pto_corte/(pto_corte - 1) , 1)
    df_actual['d_visita_corr'] =  df_actual['d_visita']*np.where(df_actual['Local'].isin(visita_5p),pto_corte/(pto_corte - 1) , 1)
    df_actual['f_local_corr'] = df_actual['f_local']*np.where(df_actual['Visita'].isin(local_5u),pto_corte/(pto_corte - 1) , 1)
    df_actual['d_local_corr'] =  df_actual['d_local']*np.where(df_actual['Visita'].isin(local_5p),pto_corte/(pto_corte - 1) , 1)
    
    df_actual['posicion_gral_visita'] = df_actual['Visita'].map(dict_eq_pos_gral)
    df_actual['posicion_gral_local'] = df_actual['Local'].map(dict_eq_pos_gral)
    df_actual['posicion_visita'] = df_actual['Visita'].map(dict_eq_pos_visita)
    df_actual['posicion_local'] = df_actual['Local'].map(dict_eq_pos_local)
    
    
    df_actual['ronda_f_gral_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(equipos_5u), 1, np.nan)
    df_actual['ronda_d_gral_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(equipos_5p), 1, np.nan)
    df_actual['ronda_f_gral_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(equipos_5u), 1, np.nan)
    df_actual['ronda_d_gral_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(equipos_5p), 1, np.nan)
    
    df_actual['ronda_f_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(visita_5u), 1, np.nan)
    df_actual['ronda_d_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(visita_5p), 1, np.nan)
    df_actual['ronda_f_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(local_5u), 1, np.nan)
    df_actual['ronda_d_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(local_5p), 1, np.nan)
    
    df_actual['ronda_f_gral_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(equipos_5u), 1, np.nan)
    df_actual['ronda_d_gral_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(equipos_5p), 1, np.nan)
    df_actual['ronda_f_gral_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(equipos_5u), 1, np.nan)
    df_actual['ronda_d_gral_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(equipos_5p), 1, np.nan)
    
    df_actual['ronda_f_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(visita_5u), 1, np.nan)
    df_actual['ronda_d_visita'] = df_actual['Round']*np.where(df_actual['Visita'].isin(visita_5p), 1, np.nan)
    df_actual['ronda_f_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(local_5u), 1, np.nan)
    df_actual['ronda_d_local'] = df_actual['Round']*np.where(df_actual['Local'].isin(local_5p), 1, np.nan)
    
    return df_actual


def variables_efecto_secuencial(df_actual, ronda_corte = 5):
    torneo = df_actual['Torneo'].drop_duplicates().values[0]
    df_test_pX = df_actual[df_actual['Round'] <= ronda_corte]
    cols_local = ['Visita'] + [i for i in df_actual.columns if '_' in i and 'local' not in i and 'ronda' not in i]
    func_local = ['count'] + ['mean' if i.startswith('posicion') else 'sum' for i in cols_local[1:]]
    cols_visita = ['Local'] + [i for i in df_actual.columns if '_' in i and 'visita' not in i and 'ronda' not in i]
    func_visita = ['count'] + ['mean' if i.startswith('posicion') else 'sum' for i in cols_visita[1:]]
    dict_df_local = dict(zip(cols_local,func_local))
    dict_df_visita = dict(zip(cols_visita,func_visita))
    g_test_l = df_test_pX.groupby('Local').agg(dict_df_local)
    g_test_l.columns = ['partidos_local'] + cols_local[1:]
    g_test_v = df_test_pX.groupby('Visita').agg(dict_df_visita)
    g_test_v.columns = ['partidos_visita'] + cols_visita[1:]
    cols_ronda_local = [i for i in df_actual.columns if 'ronda' in i and 'local' not in i]
    func_ronda = ['min' for i in cols_ronda_local]
    g_test_ronda_local = df_actual.groupby('Local').agg(dict(zip(cols_ronda_local,func_ronda)))
    cols_ronda_visita = [i for i in df_actual.columns if 'ronda' in i and 'visita' not in i]
    g_test_ronda_visita = df_actual.groupby('Visita').agg(dict(zip(cols_ronda_visita,func_ronda)))
    g_test_ronda = g_test_ronda_local.join(g_test_ronda_visita)
    g_test_lv = g_test_l.join(g_test_v)
    """
    SIGNIFICADO VARIABLES
    
    total_partidos: total de partidos disputados en las primeras X rondas
    
    facil_general: cantidad de partidos fáciles considerando la tabla general del
    torneo anterior
    perc_facil_general: porcentaje partidos fáciles considerando la tabla general
    del torneo anterior
    dificil_general: cantidad de partidos difíciles considerando la tabla general
    del torneo anterior
    perc_dificil_general: porcentaje partidos difíciles considerando la tabla general
    del torneo anterior
    
    facil_lv: cantidad de partidos fáciles considerando las tablas de local y visita
    del torneo anterior
    perc_facil_lv: porcentaje partidos fáciles considerando las tablas de local y visita
    del torneo anterior
    dificil_lv: cantidad de partidos difíciles considerando las tablas de local y visita
    del torneo anterior
    perc_dificil_lv: porcentaje partidos difíciles considerando las tablas de local y visita
    del torneo anterior
    
    prom_posicion_general: promedio de posición de rivales considerando tabla general
    del torneo anterior
    prom_posicion_lv: promedio de posición de rivales considerando las tablas de local y visita
    del torneo anterior
    
    ronda_primer_facil_general: ronda en que juega contra el primer equipo fácil según la tabla general
    del torneo anterior
    ronda_primer_dificil_general: ronda en que juega contra el primer equipo difícil según la tabla general
    del torneo anterior
    ronda_primer_facil: ronda en que juega contra el primer equipo fácil según las tablas de local y visita
    del torneo anterior
    ronda_primer_dificil: ronda en que juega contra el primer equipo difícil según las tablas de local y visita
    del torneo anterior
    """
    g_test_lv['total_partidos'] = g_test_lv['partidos_local'] + g_test_lv['partidos_visita']
    
    g_test_lv['facil_general'] = g_test_lv['f_gral_local'] + g_test_lv['f_gral_visita']
    g_test_lv['perc_facil_general'] = g_test_lv['facil_general']/g_test_lv['total_partidos']
    g_test_lv['dificil_general'] = g_test_lv['d_gral_local'] + g_test_lv['d_gral_visita']
    g_test_lv['perc_dificil_general'] = g_test_lv['dificil_general']/g_test_lv['total_partidos']
    g_test_lv['perc_regular_general'] = 1 - g_test_lv['perc_facil_general'] - g_test_lv['perc_dificil_general']
    
    g_test_lv['facil_lv'] = g_test_lv['f_local'] + g_test_lv['f_visita']
    g_test_lv['perc_facil_lv'] = g_test_lv['facil_lv']/g_test_lv['total_partidos']
    g_test_lv['dificil_lv'] = g_test_lv['d_local'] + g_test_lv['d_visita']
    g_test_lv['perc_dificil_lv'] = g_test_lv['dificil_lv']/g_test_lv['total_partidos']
    g_test_lv['perc_regular_lv'] = 1 - g_test_lv['perc_facil_lv'] - g_test_lv['perc_dificil_lv']
    
    g_test_lv['facil_general_corr'] = g_test_lv['f_gral_local_corr'] + g_test_lv['f_gral_visita_corr']
    g_test_lv['perc_facil_general_corr'] = np.minimum(g_test_lv['facil_general_corr']/g_test_lv['total_partidos'],1)
    g_test_lv['dificil_general_corr'] = g_test_lv['d_gral_local_corr'] + g_test_lv['d_gral_visita_corr']
    g_test_lv['perc_dificil_general_corr'] = np.minimum(g_test_lv['dificil_general_corr']/g_test_lv['total_partidos'],1)
    g_test_lv['perc_regular_general_corr'] = 1 - g_test_lv['perc_facil_general_corr'] - g_test_lv['perc_dificil_general_corr']
    
    g_test_lv['facil_lv_corr'] = g_test_lv['f_local_corr'] + g_test_lv['f_visita_corr']
    g_test_lv['perc_facil_lv_corr'] = np.minimum(g_test_lv['facil_lv_corr']/g_test_lv['total_partidos'],1)
    g_test_lv['dificil_lv_corr'] = g_test_lv['d_local_corr'] + g_test_lv['d_visita_corr']
    g_test_lv['perc_dificil_lv_corr'] = np.minimum(g_test_lv['dificil_lv_corr']/g_test_lv['total_partidos'],1)
    g_test_lv['perc_regular_lv_corr'] = 1 - g_test_lv['perc_facil_lv_corr'] - g_test_lv['perc_dificil_lv_corr']
    
    g_test_lv['prom_posicion_general'] = (g_test_lv['posicion_gral_local']*g_test_lv['partidos_visita'] + g_test_lv['posicion_gral_visita']*g_test_lv['partidos_local'])/g_test_lv['total_partidos']
    g_test_lv['prom_posicion_lv'] = (g_test_lv['posicion_local']*g_test_lv['partidos_visita'] + g_test_lv['posicion_visita']*g_test_lv['partidos_local'])/g_test_lv['total_partidos']
    
    g_test_ronda['ronda_primer_facil_general'] = g_test_ronda[['ronda_f_gral_visita','ronda_f_gral_local']].min(axis=1)
    g_test_ronda['ronda_primer_dificil_general'] = g_test_ronda[['ronda_d_gral_visita','ronda_d_gral_local']].min(axis=1)
    g_test_ronda['ronda_primer_facil'] = g_test_ronda[['ronda_f_visita','ronda_f_local']].min(axis=1)
    g_test_ronda['ronda_primer_dificil'] = g_test_ronda[['ronda_d_visita','ronda_d_local']].min(axis=1)
    
    g_test_lv = g_test_lv[[i for i in g_test_lv if i not in cols_local and i not in cols_visita and 'partidos' not in i]]
    g_test_lv.columns = [i + '_p' + str(ronda_corte) for i in g_test_lv.columns]
    g_test_ronda = g_test_ronda[[i for i in g_test_ronda if i not in cols_ronda_local and i not in cols_ronda_visita]]
    t_act = tabla_final_torneo(df_actual)[['equipo','position']]
    t_act['Torneo'] = torneo
    return pd.merge(t_act, g_test_lv.join(g_test_ronda), left_on = 'equipo', right_index = True)

def base_efecto_secuencial(df, filtro_ronda = False, ronda_corte = 5, pto_corte = 5, replace_releg = False):
    Torneos = df['Torneo'].drop_duplicates().values.tolist()
    dbs = []
    max_ronda = []
    for i in range(1,len(Torneos)):
        df_actual = bases_pre_indicadores(df,Torneos[i],Torneos[i-1], pto_corte = pto_corte, replace_releg = replace_releg)
        dbs.append(variables_efecto_secuencial(df_actual, ronda_corte = ronda_corte))
#        dbs.append(bases_pre_indicadores(df,Torneos[i],Torneos[i-1]))
        max_ronda.append(formatear_base(df[df['Torneo'] == Torneos[i]])['Round'].max())
    db = pd.concat(dbs, ignore_index=True)
    if filtro_ronda:
        return db[db['position'] <= min(max_ronda)].reset_index(drop=True)
    else:
        return db
    

def grabar_bd(datadir,nombre_base, ronda_corte = 5, pto_corte = 5, replace_releg = False):
    dfs = []
    sheets = []
    for subdir, dirs, files in os.walk(datadir):
        for file in files:
            filepath = subdir  + file
            if filepath.endswith(".xlsx"):
#                print("\rLeyendo: ", filepath)
                df = pd.read_excel(filepath)
                dfs.append(base_efecto_secuencial(df, ronda_corte = ronda_corte,
                                                  pto_corte = pto_corte,
                                                  replace_releg = replace_releg).round(3))
                sheets.append(file[:-5])
                
    if ".xlsx" not in nombre_base:
        nombre_base = nombre_base + ".xlsx"
    writer = pd.ExcelWriter(nombre_base,engine='xlsxwriter')   
    for dataframe, sheet in zip(dfs, sheets):
        dataframe.to_excel(writer, sheet_name=sheet, index = False)   
    writer.save()
    return dfs, sheets

#datadir = "C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Ligas importantes/"    
#bases, sheets = grabar_bd(datadir,'C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Bases Construidas/Base Efecto V2.xlsx')

def grabar_groupby(bases, sheets, nombre_base):
    dfs = []
    for i in range(len(bases)):
        df_test = bases[i]
        if i < 1:
            col_interes = ['Torneo'] + [i for i in df_test.columns if '_' in i and 'position' not in i]
            aggregates = ['count'] + [['mean','std'] for i in range(1,len(col_interes))]
            dict_df = dict(zip(col_interes,aggregates))
        g_test = df_test.groupby('equipo').agg(dict_df)
        g_test.columns = ["_".join(pair) for pair in g_test.columns]
        dfs.append(g_test.round(3))
    writer = pd.ExcelWriter(nombre_base,engine='xlsxwriter')   
    for dataframe, sheet in zip(dfs, sheets):
        dataframe.to_excel(writer, sheet_name=sheet, index = True)   
    writer.save()
    return dfs





#test#



def test_dif_medias(g_base,columna):
    g = g_base[g_base['Torneo_count'] >= g_base['Torneo_count'].mean()]
    col_mean = columna + '_mean'
    col_std = columna + '_std'
    mu1 = np.array([g[col_mean].values,]*len(g[col_mean].values))
    s1 = np.array([g[col_std].values,]*len(g[col_std].values))
    n1 = np.array([g['Torneo_count'].values,]*len(g['Torneo_count'].values))
    mu2 = np.array([g[col_mean].values,]*len(g[col_mean].values)).transpose()
    s2 = np.array([g[col_std].values,]*len(g[col_std].values)).transpose()
    n2 = np.array([g['Torneo_count'].values,]*len(g['Torneo_count'].values)).transpose()
    num = np.abs(mu1 - mu2)
    denom = np.sqrt(((n1 - 1)*s1**2 + (n2-1)*s2**2)/(n1 + n2 - 2))*np.sqrt((1/n1) + (1/n2))
    T = num/denom
    p_val = stats.t.sf(np.abs(T), n1 + n2 - 1)*2
    df_pval = pd.DataFrame(p_val, index = g.index, columns = g.index)
    return df_pval

def excel_test_medias(bases,bases_groupby, sheets, nombre_base):
    np.seterr(divide='ignore', invalid='ignore')

    for i in range(len(bases_groupby)):
        dfs = []
        df = bases_groupby[i]
        if i < 1:
            cols_interes = [i for i in bases[i].columns if '_' in i and 'pos' not in i]
        nb = 'C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Bases Construidas/' + nombre_base + ' ' + sheets[i] + '.xlsx'
        nb = " ".join(nb.split())
        writer = pd.ExcelWriter(nb,engine='xlsxwriter') 
        for j in cols_interes:
            dfs.append(test_dif_medias(df,j))
        for dataframe, sheet in zip(dfs, cols_interes):
            dataframe.to_excel(writer, sheet_name=sheet, index = True)
        writer.save()





def test_f(bases, bases_groupby, sheets, nombre_base, p_corte = 0.05, regulares = False):
    nb = 'C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Bases Construidas/' + nombre_base + '.xlsx'
    dfs = []
    F_lists = []
    pval_lists = []
    for i in range(len(bases)):
        df_b = bases[i]
        F_list = []
        pval_list = []
        df_g = bases_groupby[i].reset_index()
        if regulares:
            equipos = df_g[df_g['Torneo_count'] == df_g['Torneo_count'].max()]['equipo'].values.tolist()
            df_b = df_b[df_b['equipo'].isin(equipos)].reset_index(drop=True)
            df_g = df_g[df_g['equipo'].isin(equipos)].reset_index(drop=True)
        if i < 1:
            columnas_interes = [i for i in df_b.columns if '_' in i and 'position' not in i and 'p_' not in i]
            columnas_g = [i + '_mean' for i in columnas_interes]
        df_g = df_g[['equipo'] + columnas_g]
        for j in columnas_interes:
            formula = j + " ~ C(equipo)"
            model = ols(formula,
                        data=df_b).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            F_list.append(aov_table['F'][0])
            pval_list.append(aov_table['PR(>F)'][0])
        df_g = df_g.append(pd.Series(dict(zip(['equipo'] + columnas_g, ['F'] + F_list))), ignore_index = True)
        df_g = df_g.append(pd.Series(dict(zip(['equipo'] + columnas_g, ['p-val'] + pval_list))), ignore_index = True)
        dfs.append(df_g.round(3))
        F_lists.append(F_list)
        pval_lists.append(pval_list)
    writer = pd.ExcelWriter(nb,engine='xlsxwriter') 
    for dataframe, sheet in zip(dfs, sheets):
        dataframe.to_excel(writer, sheet_name=sheet, index = False)
    writer.save()
    return dfs, F_lists, pval_lists
            




def test_f_diferentes(rondas_corte = [3,4,5,6,7], ptos_corte = [3,4,5], replace_relegs = [True,False]):
    dfs = []
    datadir = "C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Ligas importantes/"
    outputdir = "C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Bases Construidas/"
    flag_listas = True
    for ronda_corte in rondas_corte:
        for pto_corte in ptos_corte:
            for replace_releg in replace_relegs:
                print('Ejecutando: PC' + str(pto_corte) +  ' RC' + str(ronda_corte) + ' RR ' + str(replace_releg) + '               ', end='\t\r')
                nombre_efecto = outputdir + 'Bases efecto/Base Efecto PC' + str(pto_corte) +  ' RC' + str(ronda_corte) + ' RR ' + str(replace_releg) +   '.xlsx'
                nombre_groupby = outputdir + 'Group by/Base Group By Efecto PC' + str(pto_corte) + ' RC' + str(ronda_corte) + ' RR ' + str(replace_releg) +   '.xlsx'
                nombre_testf = 'Test F/Test F PC' + str(pto_corte) + ' RC' + str(ronda_corte) + ' RR ' + str(replace_releg)
                bases, sheets = grabar_bd(datadir,
                                          nombre_efecto,
                                          ronda_corte = ronda_corte,
                                          pto_corte = pto_corte,
                                          replace_releg = replace_releg)

                bases_groupby = grabar_groupby(bases, sheets, nombre_groupby)
                df_test_sig, F_lists, pval_lists= test_f(bases,bases_groupby,sheets,nombre_testf)
                if flag_listas:
                    listas_dfs_F = [[] for j in range(len(sheets))]
                    listas_dfs_p = [[] for j in range(len(sheets))]
                    cols = ['Ronda corte', 'Punto corte', 'Reemplazos'] + [j.replace(str(ronda_corte),'X') for j in df_test_sig[0].columns if 'equipo' not in j]
                    flag_listas = False
                for i in range(len(F_lists)):
                    listas_dfs_F[i].append([ronda_corte, pto_corte, str(replace_releg)] + F_lists[i])
                    listas_dfs_p[i].append([ronda_corte, pto_corte, str(replace_releg)] + pval_lists[i])
                dfs.append(df_test_sig)
    dfs_resumen_F = []
    dfs_resumen_p = []
    for i in range(len(listas_dfs_F)):
        dfs_resumen_F.append(pd.DataFrame(listas_dfs_F[i], columns = cols).round(3))
        dfs_resumen_p.append(pd.DataFrame(listas_dfs_p[i], columns = cols).round(3))
    writer = pd.ExcelWriter('Resumen Tests F.xlsx',engine='xlsxwriter') 
    for dataframe, sheet in zip(dfs_resumen_F, sheets):
        dataframe.to_excel(writer, sheet_name=sheet, index = False)
    writer.save()
    writer = pd.ExcelWriter('Resumen Tests F p-val.xlsx',engine='xlsxwriter') 
    for dataframe, sheet in zip(dfs_resumen_p, sheets):
        dataframe.to_excel(writer, sheet_name=sheet, index = False)
    writer.save()
    return dfs
    
df_test_f = test_f_diferentes(rondas_corte = [4,5,6,7], ptos_corte = [3,4,5], replace_relegs = [False])

#outputdir = "C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Bases Construidas/"
#df_test_reg = pd.read_excel('C:/Users/crist/Google Drive/Cursos/Tesis/Datos y scripts/Bases Construidas/Base Efecto PC5 RC5 RR False.xlsx',
#                            sheet_name = 'Inglaterra')
#
#result = ols(formula="position ~ ronda_primer_facil", data=df_test_reg).fit()
#print(result.summary())

#"""
#PRUEBAS
#
#"""
#
