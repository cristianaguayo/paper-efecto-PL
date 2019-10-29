#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:12:35 2019

@author: cristian
"""

import numpy as np
import pandas as pd
import os

def ObtenerFechasTorneosAnteriores(df_full, fin_mes = True):
    df = df_full[df_full['Date'] > '1950-05-31'].reset_index(drop=True)
    Torneos = df['Torneo'].drop_duplicates().tolist()
    last_dates = []
    for i in range(len(Torneos[1:])):
        torneo = Torneos[i]
        last_dates.append(df[df['Torneo'] == torneo]['Date'].max())
    df_torneos = pd.DataFrame(data = {'Fecha' : last_dates, 'Torneo': Torneos[1:]})
    if fin_mes:
        df_torneos['Fin anterior'] = pd.to_datetime(df_torneos['Fecha'])
        df_torneos['Fecha'] = df_torneos['Fin anterior'] + pd.offsets.MonthEnd(0)
        df_torneos['Fecha'] = df_torneos['Fecha'].astype(str)
        df_torneos = df_torneos[['Fecha','Torneo']]
    return df_torneos.values.tolist()

def ObtenerRankingELO(fecha, torneo, country = 'ENG'):
    df_elo = pd.read_csv('http://api.clubelo.com/%s' % fecha, usecols = ['Club','Country','Elo'])
    if country == 'GER':
        df_elo['Country'] = [r.replace('FRG','GER').replace('GDR','GER') for r in df_elo['Country'].tolist()]
    df_elo = df_elo[df_elo['Country'] == country].reset_index(drop=True)
    df_elo['Date'] = fecha
    return df_elo[['Date','Club', 'Elo']]

def ObtenerRankingELOLigaTemporadas(df, country = 'ENG'):
    prereplaces_country = {'ENG' : {'Man ': 'Manchester ',
                                    'QPR': 'Queens Park Rangers',
                                    'Middlesboro': 'Middlesbrough',
                                    'Wolves': 'Wolverhampton Wanderers',
                                    'Sheffield Weds': 'Sheffield Wednesday'},
                           'GER': {'Nuernberg' : 'Nürnberg',
                                   'Leverkusen' : 'Bayer Leverkusen',
                                   'Gladbach' : 'Mönchengladbach',
                                   'Muenchen 60' : 'TSV 1860 München' ,
                                   'Lautern' : 'Kaiserslautern',
                                   'Koeln' : 'Köln',
                                   'St Pauli' : 'St. Pauli',
                                   'Fuerth' : 'Fürth',
                                   'Duesseldorf' : 'Düsseldorf',
                                   'Blau-Weiss 90' : 'Blau-Weiß 90 Berlin',
                                   'St Kickers' : 'Stuttgarter Kickers',
                                   'Fortuna 1. FC Köln' : 'Fortuna Köln',
                                   'Saarbruecken' : '1. FC Saarbrücken',
                                   'Lok Leipzig' : 'VfB Leipzig'},
                           'ESP' : {'Barcelona' : 'FC Barcelona',
                                    'Espanyol' : 'Espanyol Barcelona',
                                    'Atletico' : 'Atlético',
                                    'Alaves' : 'Alavés',
                                    'Cadiz' : 'Cádiz',
                                    'Tarragona' : 'Gimnàstic',
                                    'Malaga' : 'Málaga',
                                    'Almeria' : 'UD Almería',
                                    'Gijon' : 'Sporting Gijón',
                                    'Hercules' : 'Hércules CF',
                                    'Cordoba' : 'Córdoba CF',
                                    'Leganes' : 'CD Leganés',
                                    'Tetuan' : 'Atlético Tetuán',
                                    'Jaen' : 'Real Jaén',
                                    'Castellon' : 'CD Castellón',
                                    'Logrones' : 'CD Logroñés',
                                    'Merida' : 'CP Mérida'},
                           'FRA' : {'Paris SG' : 'Paris Saint-Germain',
                                    'Saint-Etienne' : 'AS Saint-Étienne',
                                    'Evian TG' : 'Évian Thonon Gaillard',
                                    'Ajaccio' : 'AC Ajaccio',
                                    'Gazelec' : 'GFC Ajaccio',
                                    'Sete' : 'FC Sète',
                                    'RC Paris' : 'Racing Club de France',
                                    'FC FC Nancy' : 'AS Nancy',
                                    'FC Nancy' : 'AS Nancy',
                                    'Stade Francais' : 'Stade Français FC',
                                    'Arles-Association Sportive Avignonaise' : 'AC Arles-Avignon',
                                    'Nimes' : 'Nîmes Olympique',
                                    'Ales' : 'Olympique Alès',
                                    'Beziers' : 'AS Béziers',
                                    'Angouleme' : 'AS Angoulême',
                                    'Montpellier La Paillade SC' : 'Montpellier HSC',
                                    'UA Sedan-Torcy' : 'CS Sedan',
                                    'Chateauroux' : 'LB Châteauroux',
                                    'AS Troyes-Savinienne' : 'ESTAC Troyes'},
                           'ITA' : {'Roma' : 'AS Roma',
                                    'Lazio' : 'Lazio Roma',
                                    'Verona' : 'Hellas Verona',
                                    'Chievo' : 'Chievo Verona',
                                    'Spal' : 'SPAL 2013 Ferrara'}}
    dfs_elo = []
    fechas_torneos = ObtenerFechasTorneosAnteriores(df, fin_mes = True)
    torneos = []
    for i in range(len(fechas_torneos)):
        fecha, torneo = fechas_torneos[i][0], fechas_torneos[i][1]
        torneos.append(torneo)
        print('Obteniendo puntaje para %s' % torneo, end = '\t\r')
        df_elo = ObtenerRankingELO(fecha, torneo, country = country)
        df_elo['Torneo'] = torneo
        dfs_elo.append(df_elo)
    df_elo = pd.concat(dfs_elo, ignore_index = True)
    df_elo = df_elo[['Torneo','Club','Elo']]
    try:
        prereplaces = prereplaces_country[country]
        df_elo = df_elo.replace(prereplaces, regex = True)
    except:
        pass
    equipos_elo = df_elo['Club'].drop_duplicates().tolist()
    equipos_cal = df[df['Torneo'].isin(torneos)]['Local'].drop_duplicates().tolist()
    dictreplaces = {}
    for eq in equipos_elo:
        eq_cal = [i for i in equipos_cal if eq in i]
        if eq_cal:
            dictreplaces[eq] = eq_cal[0]
    df_elo = df_elo.replace(dictreplaces, regex = True)
    equipos_elo = df_elo['Club'].drop_duplicates().tolist()
    eq_int = [e for e in equipos_elo if e in equipos_cal]
    if len(eq_int) != len(equipos_cal):
        print("\n Equipos BD: ", [e for e in equipos_cal if e not in eq_int], "\n Equipos ELO: ", [e for e in equipos_elo if e not in eq_int], "\n")
    return df_elo

def LeerAgregarELO(datadir, str_liga = 'Inglaterra'):
    dictcountries = {'Inglaterra' : 'ENG',
                     'Alemania' : 'GER',
                     'Francia' : 'FRA',
                     'Italia' : 'ITA',
                     'Espana' : 'ESP'}
    country = dictcountries[str_liga]
    df = pd.read_excel(os.path.join(datadir, '%s.xlsx' % str_liga))
    df_elo = ObtenerRankingELOLigaTemporadas(df, country = country)
    return df_elo


def ParcharELO(elodir, liga):
    df_elo = pd.read_excel(os.path.join(elodir, '%s-hist.xlsx' % liga))
    dictliga = {'Alemania' : 'GER',
                'Espana' : 'ESP', 
                'Francia' : 'FRA',
                'Italia' : 'ITA'}
    prereplaces_country = {'ENG' : {'Man ': 'Manchester ',
                                    'QPR': 'Queens Park Rangers',
                                    'Middlesboro': 'Middlesbrough',
                                    'Wolves': 'Wolverhampton Wanderers',
                                    'Sheffield Weds': 'Sheffield Wednesday'},
                           'GER': {'Nuernberg' : 'Nürnberg',
                                   'Meidericher SV' : 'MSV Duisburg',
                                   'Frankfurter SG Eintracht' : 'Eintracht Frankfurt',
                                   'SV Bayer 04 Bayer Leverkusen' : 'Bayer Leverkusen',
                                   'Gladbach' : 'Mönchengladbach',
                                   'Muenchen 60' : 'TSV 1860 München' ,
                                   'Lautern' : 'Kaiserslautern',
                                   'Koeln' : 'Köln',
                                   'St Pauli' : 'St. Pauli',
                                   'Fuerth' : 'Fürth',
                                   'Duesseldorf' : 'Düsseldorf',
                                   'Blau-Weiss 90' : 'Blau-Weiß 90 Berlin',
                                   'St Kickers' : 'Stuttgarter Kickers',
                                   'Fortuna 1. FC Köln' : 'Fortuna Köln',
                                   'Saarbruecken' : '1. FC Saarbrücken',
                                   'Lok Leipzig' : 'VfB Leipzig'},
                           'ESP' : {'CD Málaga' : 'Málaga CF',
                                    'AD Almería' : 'UD Almería',
                                    'Real Burgos' : 'Burgos CF', 
                                    'Barcelona' : 'FC Barcelona',
                                    'Espanyol' : 'Espanyol Barcelona',
                                    'Atletico' : 'Atlético',
                                    'Alaves' : 'Alavés',
                                    'Cadiz' : 'Cádiz',
                                    'Tarragona' : 'Gimnàstic',
                                    'Malaga' : 'Málaga',
                                    'Almeria' : 'UD Almería',
                                    'Gijon' : 'Sporting Gijón',
                                    'Hercules' : 'Hércules CF',
                                    'Cordoba' : 'Córdoba CF',
                                    'Leganes' : 'CD Leganés',
                                    'Tetuan' : 'Atlético Tetuán',
                                    'Jaen' : 'Real Jaén',
                                    'Castellon' : 'CD Castellón',
                                    'Logrones' : 'CD Logroñés',
                                    'Merida' : 'CP Mérida'},
                           'FRA' : {'Stade Olympique Montpelliérain' : 'Montpellier HSC',
                                    'Brest Armorique FC' : 'Stade Brest',
                                    'Matra Racing' : 'Racing Club de France',
                                    'ATAC Troyes' : 'ESTAC Troyes',
                                    'US Valenciennes-Anzin' : 'Valenciennes FC', 
                                    'Paris SG' : 'Paris Saint-Germain',
                                    'Saint-Etienne' : 'AS Saint-Étienne',
                                    'Evian TG' : 'Évian Thonon Gaillard',
                                    'Ajaccio' : 'AC Ajaccio',
                                    'Gazelec' : 'GFC Ajaccio',
                                    'Sete' : 'FC Sète',
                                    'RC Paris' : 'Racing Club de France',
                                    'FC FC Nancy' : 'AS Nancy',
                                    'FC Nancy' : 'AS Nancy',
                                    'Stade Francais' : 'Stade Français FC',
                                    'Arles-Association Sportive Avignonaise' : 'AC Arles-Avignon',
                                    'Nimes' : 'Nîmes Olympique',
                                    'Ales' : 'Olympique Alès',
                                    'Beziers' : 'AS Béziers',
                                    'Angouleme' : 'AS Angoulême',
                                    'Montpellier La Paillade SC' : 'Montpellier HSC',
                                    'UA Sedan-Torcy' : 'CS Sedan',
                                    'Chateauroux' : 'LB Châteauroux',
                                    'AS Troyes-Savinienne' : 'ESTAC Troyes',
                                    'GFC AC AC AC AC AC AC AC Ajaccio' : 'GFC Ajaccio',
                                    'AC AC AC AC AC AC AC AC Ajaccio' : 'AC Ajaccio'},
                           'ITA' : {'Roma' : 'AS Roma',
                                    'Lazio' : 'Lazio Roma',
                                    'Verona' : 'Hellas Verona',
                                    'Chievo' : 'Chievo Verona',
                                    'Spal' : 'SPAL 2013 Ferrara'}}
    country = dictliga[liga]
    print(country)
    prereplaces = prereplaces_country[country]
    club = df_elo['Club'].tolist()
    for key, value in prereplaces.items():
        club = [c.replace(key,value) for c in club]
    df_elo['Club'] = club
    df_elo.to_excel(os.path.join(elodir, '%s-hist.xlsx' % liga), index = False)

datadir = os.path.join(os.path.pardir,'datos','ligas-hist')
outputdir = os.path.join(os.path.pardir,'datos','elo')
for liga in ['Espana']:
    ParcharELO(outputdir, liga)
#    df_elo = LeerAgregarELO(datadir, str_liga = liga)
#    df_elo.to_excel(os.path.join(outputdir, '%s-hist.xlsx' % liga), index = False)
    