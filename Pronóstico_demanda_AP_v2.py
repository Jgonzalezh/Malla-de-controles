# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:14:13 2020

@author: jgonzalezh

Modelos juntos

"""

""" NO OLVIDAR COMPLETAR POR 0 SI EXISTEN FECHAS EN LOS CUALES NO SE TENGA REGISTRO"""

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('seaborn')
from datetime import date
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
import math
from pandas.tseries.offsets import MonthEnd
import pyodbc
from sqlalchemy import create_engine
from six.moves import urllib

#Tiene que ser un csv con centro, fecha de agrupación de demanda, demanda variable, demanda fija
class DATA:
    def __init__(self,data_csv,dmd_col=2,long_test_month=4): 
        #leer la base
        # encoding='utf-8'problemas con SQL(?)
        self.data=data_csv #'Base de datos con demanda historica'# 
        df=data(data_csv)
        #dia en el que se corre el código
        hoy= pd.to_datetime('today').replace(hour=0,minute=0,second=0,microsecond=0,nanosecond=0)
        #inicio de este mes
        hoy_inicio_mes=hoy.replace(day=1)
        #se agrupan en listas una fecha definida para cada mes que coincida con los group by por mes para hacer los MAAPE de test y tratamiento
        self.ini=[0 for r in range(4)]
        self.prev=[0 for r in range(4)] 
        self.ini[0]= hoy_inicio_mes + MonthEnd(1)
        self.ini[1]= hoy_inicio_mes + relativedelta(months=1) + MonthEnd(1)
        self.ini[2]= hoy_inicio_mes + relativedelta(months=2) + MonthEnd(1)
        self.ini[3]= hoy_inicio_mes + relativedelta(months=3)+ MonthEnd(1)
        self.prev[0]= hoy_inicio_mes + relativedelta(months=-4) + MonthEnd(1)
        self.prev[1]= hoy_inicio_mes + relativedelta(months=-3) + MonthEnd(1)
        self.prev[2]= hoy_inicio_mes + relativedelta(months=-2) + MonthEnd(1)
        self.prev[3]= hoy_inicio_mes + relativedelta(months=-1)+ MonthEnd(1)
        #se le asigna un nombre estandar a las variables, el orden tiene que ser centro, fecha, demanda 1 y demanda 2
        df.columns=['Centro','Month','D var','D fix']
        df['Month']=pd.to_datetime(df['Month'])
        df.sort_values(by=['Month'], inplace=True)
        self.dt=df
        C=df.groupby(['Centro']).mean()
        self.centro=C.index.tolist()
        self.n_centros=len(C.index)
        P=df.groupby(['Month']).mean()
        self.n_periodos=len(P.index)
        iter_periodos=P.index.tolist()
        self.periodos=pd.to_datetime(pd.Series(iter_periodos))
        # se hacen listas para agrupar la demanda en su totalidad, en test y en entrenamiento. Se busca agrupar por meses completos
        self.ts1=[0 for t in range(self.n_centros)]
        self.ts1_train=[0 for t in range(self.n_centros)]
        self.ts1_test=[0 for t in range(self.n_centros)]
        self.ts2=[0 for t in range(self.n_centros)]
        self.ts2_train=[0 for t in range(self.n_centros)]
        self.ts2_test=[0 for t in range(self.n_centros)]
        self.Mts1=[0 for t in range(self.n_centros)]
        self.Mts1_train=[0 for t in range(self.n_centros)]
        self.Mts1_test=[0 for t in range(self.n_centros)]
        self.Mts2=[0 for t in range(self.n_centros)]
        self.Mts2_train=[0 for t in range(self.n_centros)]
        self.Mts2_test=[0 for t in range(self.n_centros)]
        self.data_mes=pd.DataFrame(columns=['Centro','Month','D var','D fix'])
        #Se agrega para cada centro su demanda, hay muchas funciones de fechas que sirven para que no hayan problema con años y meses de diferentes duraciones
        for c in range(self.n_centros):
            df_x_centro=df[df.Centro.isin([self.centro[c]])]
            ts_temp=df_x_centro[['Month','D var','D fix']]
            ts_temp2=df_x_centro[['Month','D var','D fix']]
            ts_temp2.set_index('Month', inplace=True)
            prev_data=ts_temp2.groupby(pd.Grouper(freq='M')).sum()
            prev_data=prev_data[:pd.to_datetime(prev_data.index.max()+relativedelta(months=-1)+ MonthEnd(1))]
            prev_data.reset_index(inplace=True)
            prev_data['Centro']=self.centro[c]
            prev_data=prev_data[['Centro','Month','D var','D fix']]
            self.data_mes=pd.concat([self.data_mes, prev_data], ignore_index=True)
            self.prev_d=prev_data
            ts_temp.set_index('Month', inplace=True)
            delta = ts_temp.index.max().replace(day=1) + relativedelta(months=-long_test_month)
            delta2 = ts_temp.index.max().replace(day=1) + relativedelta(months=-long_test_month*3)
            self.train=pd.to_datetime(delta)
            self.begin_train=pd.to_datetime(delta2)
            self.forecast_days=(pd.to_datetime(ts_temp.index.max()+relativedelta(months=4)+ MonthEnd(1))-pd.to_datetime(ts_temp.index.max() +relativedelta(months=-1)+ MonthEnd(1))).days
            self.test_todays=(pd.to_datetime(ts_temp.index.max().replace(day=1))-self.train).days
            self.test_tomonth=4#(pd.to_datetime(ts_temp.index.max().replace(day=1))-self.train).month
            self.train_todays=(pd.to_datetime(ts_temp.index.max().replace(day=1))-self.begin_train).days
            self.end_test=pd.to_datetime(ts_temp.index.max() +relativedelta(months=-1)+ MonthEnd(1))
            self.trains_2= pd.to_datetime(ts_temp.index.max().replace(day=1) + relativedelta(months=+long_test_month))
            self.ts1[c]=ts_temp['D var']
            self.ts2[c]=ts_temp['D fix']
            self.ts1[c]= self.ts1[c][:pd.to_datetime(ts_temp.index.max() +relativedelta(months=-1)+ MonthEnd(1))]
            self.ts2[c]=self.ts2[c][:pd.to_datetime(ts_temp.index.max() +relativedelta(months=-1)+ MonthEnd(1))]
            self.ts1_train[c]=self.ts1[c][:self.train+relativedelta(days=-1)]
            self.ts1_test[c]=self.ts1[c][self.train:pd.to_datetime(ts_temp.index.max() +relativedelta(months=-1)+ MonthEnd(1))]
            self.ts2_train[c]=self.ts2[c][:self.train+relativedelta(days=-1)]
            self.ts2_test[c]=self.ts2[c][self.train:pd.to_datetime(ts_temp.index.max() +relativedelta(months=-1)+ MonthEnd(1))] 
            self.Mts1[c]=self.ts1[c].groupby(pd.Grouper(freq='M')).sum()
            self.Mts2[c]=self.ts2[c].groupby(pd.Grouper(freq='M')).sum()
            self.Mts1_train[c]=self.ts1_train[c].groupby(pd.Grouper(freq='M')).sum()
            self.Mts2_train[c]=self.ts2_train[c].groupby(pd.Grouper(freq='M')).sum()
            self.Mts1_test[c]=self.ts1_test[c].groupby(pd.Grouper(freq='M')).sum()
            self.Mts2_test[c]=self.ts2_test[c].groupby(pd.Grouper(freq='M')).sum()

    def SARIMA_param(self,c):
        #ajustar parametros sarima diario, Resultados con data test (evaluar) resultados (mismos parametros) para data hasta actualidad
        #demanda de controles
        p = d = q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        resultado=999999999
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.ts2_train[c], #exog=self.ts1_train[c],
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
        
                    results = mod.fit()
        
                    #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                    if results.aic<resultado:
                        resultado=results.aic
                        param1=param
                        param2=param_seasonal
                except:
                    continue
        mod = sm.tsa.statespace.SARIMAX(self.ts2_train[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        mod2 = sm.tsa.statespace.SARIMAX(self.ts2[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        
        results2 = mod.fit()
        results3 = mod2.fit()
        return [results2, results3]
    def SARIMA_Mparam(self,c):
        #idem pero para datos mensuales
        p = d = q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        resultado=999999999
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.Mts2_train[c], #exog=self.ts1_train[c],
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
        
                    results = mod.fit()
        
                    #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                    if results.aic<resultado:
                        resultado=results.aic
                        param1=param
                        param2=param_seasonal
                except:
                    continue
        mod = sm.tsa.statespace.SARIMAX(self.Mts2_train[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        mod2 = sm.tsa.statespace.SARIMAX(self.Mts2[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        
        results2 = mod.fit()
        results3 = mod2.fit()
        return [results2, results3]
    def SARIMA_param2(self,c):
        #idem para demanda variable
        p = d = q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        resultado=999999999
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.ts1_train[c], #exog=self.ts2_train[c],
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
        
                    results = mod.fit()
        
                    #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                    if results.aic<resultado:
                        resultado=results.aic
                        param1=param
                        param2=param_seasonal
                except:
                    continue
        mod = sm.tsa.statespace.SARIMAX(self.ts1_train[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        mod2 = sm.tsa.statespace.SARIMAX(self.ts1[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        
        results2 = mod.fit()
        results3 = mod2.fit()
        return [results2, results3]
    def SARIMA_Mparam2(self,c):
        p = d = q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        resultado=999999999
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.Mts1_train[c], #exog=self.ts2_train[c],
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
        
                    results = mod.fit()
        
                    #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                    if results.aic<resultado:
                        resultado=results.aic
                        param1=param
                        param2=param_seasonal
                except:
                    continue
        mod = sm.tsa.statespace.SARIMAX(self.Mts1_train[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        mod2 = sm.tsa.statespace.SARIMAX(self.Mts1[c], 
                                        order=param1,
                                        seasonal_order=param2,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        
        results2 = mod.fit()
        results3 = mod2.fit()
        return [results2, results3]
        
    def SARIMA_forecast(self,c,params,frec='M'):
        #forecast diario
        results=params
        pred_uc = results.get_forecast(steps=self.test_todays)#index=self.ts2_test[c])
        pred_ci = pred_uc.conf_int()
        forecast=pred_uc.predicted_mean
        result=forecast.groupby(pd.Grouper(freq=frec)).sum()
        return result.to_frame()
    
    def SARIMA_forecast_M(self,c,params,frec='M'):
        #forecast mensual
        results=params
        pred_uc = results.get_forecast(steps=self.test_tomonth)#index=self.ts2_test[c])
        pred_ci = pred_uc.conf_int()
        forecast=pred_uc.predicted_mean
        result=forecast.groupby(pd.Grouper(freq=frec)).sum()
        return result.to_frame()

    def Test_Sarima(self,c, params,frec='D'):
        #Calcular MAAPE para SARIMA
        results=params
        pred = results.get_prediction(start=pd.to_datetime(self.begin_train), dynamic=False)
        pred_uc = results.get_forecast(steps=self.test_todays)
        train=pred.predicted_mean
        train.rename('forecast_train', inplace=True)
        test=pred_uc.predicted_mean
        test.rename('forecast_test', inplace=True)
        #both=train.add(test,fill_value=0)
        test_obs=self.ts2_test[c]
        train_obs=self.ts2_train[c][self.begin_train:]
        result_test = pd.merge(test,
                 test_obs,how='left',left_index=True, right_index=True)
        result_train= pd.merge(train,
                 train_obs,how='left',left_index=True, right_index=True)
        result_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        result_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        MAAPE_train=0
        MAAPE_test=0
        z=0
        for i in range(result_train.shape[0]):
            if np.isnan(math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,1]))==False:
                MAAPE_train+=math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,0])
                z+=1       
        if z!=0:
            MAAPE_train=MAAPE_train/(z)
        else: 
            MAAPE_train=999999999
        z=0
        for j in range(result_test.shape[0]):
            if np.isnan(math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1]))==False:
                MAAPE_test+=math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1])
                z+=1
        if z!=0:
            MAAPE_test=MAAPE_test/(z)
        else: 
            MAAPE_test=999999999
        #print('MAAPE train',MAAPE_train, ' \nMAAPE test', MAAPE_test)
        return  [ MAAPE_train,MAAPE_test]
    def plot_SARIMA_M(self,c,params,frec='M',date_init='2019-01-01'):# Freq puede ser D W Y etc, dia semana, año
        #Plot SARIMA MENSUAL
        results=params[0]
        pred = results.get_prediction(start=pd.to_datetime(self.begin_train+MonthEnd(1)), dynamic=False)
        pred_uc = results.get_forecast(steps=self.test_tomonth)
        train=pred.predicted_mean
        train.rename('forecast_train', inplace=True)
        test=pred_uc.predicted_mean
        test.rename('forecast_test', inplace=True)
        #both=train.add(test,fill_value=0)
        test_obs=self.Mts2_test[c]
        train_obs=self.Mts2_train[c][self.begin_train+MonthEnd(1):]
        result_test = pd.merge(test,
                 test_obs,how='left',left_index=True, right_index=True)
        result_train= pd.merge(train,
                 train_obs,how='left',left_index=True, right_index=True)
        out_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        out_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        
        ax=out_train.plot(kind='line',y='D fix', color='g',label='Real demand', title= 'SARIMA forecast', grid=True)
        out_train.plot(kind='line',y='forecast_train', color='b',ax=ax,label='Forecast Train')

        out_test.plot(kind='line',y='forecast_test', color='r', ax=ax,label='Forecast test')
        out_test.plot(kind='line',y='D fix', color='g', ax=ax, legend=False)
        return[out_test,out_train]
  
    def Test_Sarima_M(self,c, params,frec='M'):
        #calculare MAAPE para sarima mensual
        results=params
        pred = results.get_prediction(start=pd.to_datetime(self.begin_train+MonthEnd(1)), dynamic=False)
        pred_uc = results.get_forecast(steps=self.test_tomonth)
        train=pred.predicted_mean
        train.rename('forecast_train', inplace=True)
        test=pred_uc.predicted_mean
        test.rename('forecast_test', inplace=True)
        #both=train.add(test,fill_value=0)
        test_obs=self.Mts2_test[c]
        train_obs=self.Mts2_train[c][self.begin_train+MonthEnd(1):]
        result_test = pd.merge(test,
                 test_obs,how='left',left_index=True, right_index=True)
        result_train= pd.merge(train,
                 train_obs,how='left',left_index=True, right_index=True)
        result_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        result_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        MAAPE_train=0
        MAAPE_test=0
        z=0
        for i in range(result_train.shape[0]):
            if np.isnan(math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,1]))==False:
                MAAPE_train+=math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,0])
                z+=1       
        if z!=0:
            MAAPE_train=MAAPE_train/(z)
        else: 
            MAAPE_train=999999999
        z=0
        for j in range(result_test.shape[0]):
            if np.isnan(math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1]))==False:
                MAAPE_test+=math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1])
                z+=1
        if z!=0:
            MAAPE_test=MAAPE_test/(z)
        else: 
            MAAPE_test=999999999
        #print('MAAPE train',MAAPE_train, ' \nMAAPE test', MAAPE_test)
        return  [ MAAPE_train,MAAPE_test]
    def Test_Sarima_M2(self,c, params,frec='M'):
        #Calcula MAAPE mensual para variable
        results=params
        pred = results.get_prediction(start=pd.to_datetime(self.begin_train+MonthEnd(1)), dynamic=False)
        pred_uc = results.get_forecast(steps=self.test_tomonth)
        train=pred.predicted_mean
        train.rename('forecast_train', inplace=True)
        test=pred_uc.predicted_mean
        test.rename('forecast_test', inplace=True)
        #both=train.add(test,fill_value=0)
        test_obs=self.Mts1_test[c]
        train_obs=self.Mts1_train[c][self.begin_train+MonthEnd(1):]
        result_test = pd.merge(test,
                 test_obs,how='left',left_index=True, right_index=True)
        result_train= pd.merge(train,
                 train_obs,how='left',left_index=True, right_index=True)
        result_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        result_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        MAAPE_train=0
        MAAPE_test=0
        z=0
        for i in range(result_train.shape[0]):
            if np.isnan(math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,1]))==False:
                MAAPE_train+=math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,0])
                z+=1       
        if z!=0:
            MAAPE_train=MAAPE_train/(z)
        else: 
            MAAPE_train=999999999
        z=0
        for j in range(result_test.shape[0]):
            if np.isnan(math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1]))==False:
                MAAPE_test+=math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1])
                z+=1
        if z!=0:
            MAAPE_test=MAAPE_test/(z)
        else: 
            MAAPE_test=999999999
        #print('MAAPE train',MAAPE_train, ' \nMAAPE test', MAAPE_test)
        return  [ MAAPE_train,MAAPE_test]
    def Test_Sarima2(self,c, params,frec='D'):
        #Calcular MAAPE para variable
        results=params
        pred = results.get_prediction(start=pd.to_datetime(self.begin_train), dynamic=False)
        pred_uc = results.get_forecast(steps=self.test_todays)
        train=pred.predicted_mean
        train.rename('forecast_train', inplace=True)
        test=pred_uc.predicted_mean
        test.rename('forecast_test', inplace=True)
        #both=train.add(test,fill_value=0)
        test_obs=self.ts1_test[c]
        train_obs=self.ts1_train[c][self.begin_train:]
        result_test = pd.merge(test,
                 test_obs,how='left',left_index=True, right_index=True)
        result_train= pd.merge(train,
                 train_obs,how='left',left_index=True, right_index=True)
        result_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        result_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        MAAPE_train=0
        MAAPE_test=0
        z=0
        for i in range(result_train.shape[0]):
            if np.isnan(math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,1]))==False:
                MAAPE_train+=math.atan(math.fabs(result_train.iat[i,1]-result_train.iat[i,0])/result_train.iat[i,0])
                z+=1       
        if z!=0:
            MAAPE_train=MAAPE_train/(z)
        else: 
            MAAPE_train=999999999
        z=0
        for j in range(result_test.shape[0]):
            if np.isnan(math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1]))==False:
                MAAPE_test+=math.atan(math.fabs(result_test.iat[j,1]-result_test.iat[j,0])/result_test.iat[j,1])
                z+=1
        if z!=0:
            MAAPE_test=MAAPE_test/(z)
        else: 
            MAAPE_test=999999999
        #print('MAAPE train',MAAPE_train, ' \nMAAPE test', MAAPE_test)
        return  [ MAAPE_train,MAAPE_test]

    def prophet_forecast(self,c,frec='M'):
        #forecast de prophet variable
        df_x_centro=self.dt[self.dt.Centro.isin([self.centro[c]])]
        ts_temp=df_x_centro[['Month','D var','D fix']]
        ts_temp.columns=['ds','D var','y']
        ts_temp['ds']=pd.to_datetime(ts_temp['ds'])
        mask= (ts_temp['ds']<=self.end_test) 
        ts_temp.loc[mask]
        m = Prophet( yearly_seasonality=True)
        m.fit(ts_temp[['ds','y']])
        future = m.make_future_dataframe(periods=self.test_todays)
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        forecast['ds']=pd.to_datetime(forecast['ds'])
        forecast.set_index('ds', inplace=True)
        result=forecast[['yhat']].groupby(pd.Grouper(freq=frec)).sum()   
        return result
    
    def prophet_forecast2(self,c,frec='M'):
        #forecast de prophet contorles
        df_x_centro=self.dt[self.dt.Centro.isin([self.centro[c]])]
        ts_temp=df_x_centro[['Month','D var','D fix']]
        ts_temp.columns=['ds','y','D fix']
        ts_temp['ds']=pd.to_datetime(ts_temp['ds'])
        mask= (ts_temp['ds']<=self.end_test) 
        ts_temp.loc[mask]
        m = Prophet( yearly_seasonality=True)
        m.fit(ts_temp[['ds','y']])
        future = m.make_future_dataframe(periods=self.test_todays)
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        forecast['ds']=pd.to_datetime(forecast['ds'])
        forecast.set_index('ds', inplace=True)
        result=forecast[['yhat']].groupby(pd.Grouper(freq=frec)).sum()   
        return result

    def Test_prophet(self,c,frec):
        #Calcular MAAPE para prophet de manda variable
        df_x_centro=self.dt[self.dt.Centro.isin([self.centro[c]])]
        ts_temp=df_x_centro[['Month','D var','D fix']]
        ts_temp.columns=['ds','D var','y']
        ts_temp['ds']=pd.to_datetime(ts_temp['ds'])
        mask=(ts_temp['ds']<self.train)
        ts_temp_train=ts_temp.loc[mask]
        mask2=(ts_temp['ds']>=self.train)
        ts_temp_test=ts_temp.loc[mask2]
        m = Prophet( yearly_seasonality=True)
        m.fit(ts_temp_train[['ds','y']])
        future = m.make_future_dataframe(periods=self.test_todays)
        forecast = m.predict(future)
        forecast[['ds','yhat']]
        resultados=pd.merge(ts_temp[['ds','y']],
                 forecast[['ds','yhat']], how='left',left_on='ds', right_on='ds')

        mask3=(resultados['ds']<self.train)
        mask4=(resultados['ds']>=self.train) & (resultados['ds']<=self.end_test) 
        result_train=resultados.loc[mask3]
        result_test=resultados.loc[mask4]
        result_test[['y','yhat']].fillna(0, inplace=True)
        result_train.set_index('ds', inplace=True)
        result_test.set_index('ds', inplace=True)
        result_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        result_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        MAAPE_train=0
        MAAPE_test=0
        z=0
        for i in range(result_train.shape[0]):
            if np.isnan(math.atan(math.fabs(result_train.iat[i,0]-result_train.iat[i,1])/result_train.iat[i,0]))==False:
                MAAPE_train+=math.atan(math.fabs(result_train.iat[i,0]-result_train.iat[i,1])/result_train.iat[i,0])
                z+=1
        if z!=0:
            MAAPE_train=MAAPE_train/(z)
        else:
            MAAPE_train=999999999
        z=0
        for j in range(result_test.shape[0]):
            if np.isnan(math.atan(math.fabs(result_test.iat[j,0]-result_test.iat[j,1])/result_test.iat[j,0]))==False:
                MAAPE_test+=math.atan(math.fabs(result_test.iat[j,0]-result_test.iat[j,1])/result_test.iat[j,0])
                z+=1
        if z!=0:
            MAAPE_test=MAAPE_test/(z)
        else:
            MAAPE_test=999999999

        
        return [MAAPE_train,MAAPE_test]
    def Test_prophet2(self,c,frec):
        #calcular MAAPE prophet demanda controles
        df_x_centro=self.dt[self.dt.Centro.isin([self.centro[c]])]
        ts_temp=df_x_centro[['Month','D var','D fix']]
        ts_temp.columns=['ds','y','D fix']
        ts_temp['ds']=pd.to_datetime(ts_temp['ds'])
        mask=(ts_temp['ds']<self.train)
        ts_temp_train=ts_temp.loc[mask]
        mask2=(ts_temp['ds']>=self.train)
        ts_temp_test=ts_temp.loc[mask2]
        m = Prophet( yearly_seasonality=True)
        m.fit(ts_temp_train[['ds','y']])
        future = m.make_future_dataframe(periods=self.test_todays)
        forecast = m.predict(future)
        forecast[['ds','yhat']]
        resultados=pd.merge(ts_temp[['ds','y']],
                 forecast[['ds','yhat']], how='left',left_on='ds', right_on='ds')

        mask3=(resultados['ds']<self.train)
        mask4=(resultados['ds']>=self.train) & (resultados['ds']<=self.end_test)  
        result_train=resultados.loc[mask3]
        result_test=resultados.loc[mask4]
        result_test[['y','yhat']].fillna(0, inplace=True)
        result_train.set_index('ds', inplace=True)
        result_test.set_index('ds', inplace=True)
        result_test=result_test.groupby(pd.Grouper(freq=frec)).sum()
        result_train=result_train.groupby(pd.Grouper(freq=frec)).sum()
        MAAPE_train=0
        MAAPE_test=0
        z=0
        for i in range(result_train.shape[0]):
            if np.isnan(math.atan(math.fabs(result_train.iat[i,0]-result_train.iat[i,1])/result_train.iat[i,0]))==False:
                MAAPE_train+=math.atan(math.fabs(result_train.iat[i,0]-result_train.iat[i,1])/result_train.iat[i,0])
                z+=1
        if z!=0:
            MAAPE_train=MAAPE_train/(z)
        else:
            MAAPE_train=999999999
        z=0
        for j in range(result_test.shape[0]):
            if np.isnan(math.atan(math.fabs(result_test.iat[j,0]-result_test.iat[j,1])/result_test.iat[j,0]))==False:
                MAAPE_test+=math.atan(math.fabs(result_test.iat[j,0]-result_test.iat[j,1])/result_test.iat[j,0])
                z+=1
        if z!=0:
            MAAPE_test=MAAPE_test/(z)
        else:
            MAAPE_test=999999999
        
        return [MAAPE_train,MAAPE_test]

def evaluacion(indicadores1,indicadores2,indicadores3):
    #evaluar que maape es mejor
    score1=math.fabs(indicadores1[0]-indicadores1[1])*2+ indicadores1[0]+indicadores1[1]
    score2=math.fabs(indicadores2[0]-indicadores2[1])*2+ indicadores2[0]+indicadores2[1]
    score3=math.fabs(indicadores3[0]-indicadores3[1])*2+ indicadores3[0]+indicadores3[1]
    if indicadores1[0]==999999999 and indicadores1[1]==999999999 and indicadores2[0]==999999999 and indicadores2[1]==999999999:
        a=3
    elif score1<=score2 and score1<=score3 :
        a=0
    elif score2<=score1 and score2<=score3 :
        a=1
    else:
        a=2
    return a

def forecast(c,H):
    #dar el mejor forecast para un tipo de demanda
    p=H.SARIMA_param(c)
    params=p[0]
    params2=p[1]
    p_m=H.SARIMA_Mparam(c)
    sarima=H.Test_Sarima(c, params,frec='M')
    sarima2=H.Test_Sarima_M(c, p_m[0],frec='M')
    prophet=H.Test_prophet(c,frec='M')
    evaluada= evaluacion(sarima,prophet,sarima2)
    if evaluada==0:
        x=H.SARIMA_forecast(c,params2)
        print(H.centro[c],'Sarima')
    elif evaluada==1:    
        x=H.prophet_forecast(c)
        print(H.centro[c],'Prophet')
    elif evaluada==2:   
        x=H.SARIMA_forecast_M(c,p_m[1])
        print(H.centro[c],'Sarima mensual')
    else:
        x=pd.DataFrame()
        print(H.centro[c],'Null')
    return x
def forecast2(c,H):
    #dar mejor forecast para el otro tipo de demanda
    p=H.SARIMA_param2(c)
    params=p[0]
    params2=p[1]
    p_m=H.SARIMA_Mparam2(c)
    sarima=H.Test_Sarima2(c, params,frec='M')
    sarima2=H.Test_Sarima_M2(c, p_m[0],frec='M')
    prophet=H.Test_prophet2(c,frec='M')
    evaluada= evaluacion(sarima,prophet,sarima2)
    if evaluada==0:
        x=H.SARIMA_forecast(c,params2)
        print(H.centro[c],'Sarima')
    elif evaluada==1:    
        x=H.prophet_forecast2(c)
        print(H.centro[c],'Prophet')
    elif evaluada==2:   
        x=H.SARIMA_forecast_M(c,p_m[1])
        print(H.centro[c],'Sarima mensual')
    else:
        x=pd.DataFrame()
        print(H.centro[c],'Null')
    return x
        
def data(tabla):
    #Función para leer la data, pregunta a nuestro servidor de capacity por el nombre de la tabla que se solicite
    conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=ACHS-ANALYTICAZ.achs.cl;"
                      "Database=az-capacity;"
                      "Trusted_Connection=yes;")
    sql= "Select * from " + tabla
    #la consulta se hace a través de un select
    data=pd.read_sql(sql, conn)
    #guarda el archivo como una variable de python
    return data #devuelve la variable    

for i in range(200):
    plt.close()  


#leer la data
H=DATA('CP_DemandaHistoricaAP_JGH')


#Periodos del forecast +1; 4, es decir 3 meses a futuro tomando el primero el mes presente como uno pero no se
#listas y variables para ordenar los resultados
T_forecast=4
A=[t for t in range(H.n_centros)]
Forecast=[[0 for r in range(len(A))]for t in range(3)]

E=[[0 for i in range(T_forecast*len(A))]for j in range(4)]

#fechas de mes (se toma y agrupa por el ultimo diua del mes)
hoy= pd.to_datetime('today').replace(hour=0,minute=0,second=0,microsecond=0,nanosecond=0)

hoy_inicio_mes=hoy.replace(day=1)
ini=[0 for r in range(T_forecast)]
ini[0]= hoy_inicio_mes + MonthEnd(1)
ini[1]= hoy_inicio_mes + relativedelta(months=1) + MonthEnd(1)
ini[2]= hoy_inicio_mes + relativedelta(months=2) + MonthEnd(1)
ini[3]= hoy_inicio_mes + relativedelta(months=3)+ MonthEnd(1)



#se hace el forecast para todos los centros, comparando las series de tiempo ytomando la mejor para cada caso
for i in range(len(A)):
    Forecast[0][i]=forecast(i,H)
    Forecast[2][i]=forecast2(i,H)
    Forecast[1][i]=H.centro[i]

    for r in range(T_forecast):
        E[0][i*T_forecast+r]= Forecast[1][i]
        if not Forecast[0][i].empty:
            for d in range(len(Forecast[0][i])):
                if Forecast[0][i].index.values[d]==ini[r]:
                    E[1][i*T_forecast+r]=Forecast[0][i].index.values[d]
                    E[2][i*T_forecast+r]=Forecast[0][i].iat[d,0]
            for d2 in range(len(Forecast[2][i])):
                if Forecast[2][i].index.values[d2]==ini[r]:
                    #E[3][i*3+r]=Forecast[2][i].index.values[d2]
                    E[3][i*T_forecast+r]=Forecast[2][i].iat[d2,0]


#se agregan los archivos a una base de datos

db=[]
for c in range(T_forecast*len(A)):
    if E[1][c]!=0:
        if E[3][c]!=0:    
            db.append((E[0][c],E[1][c],E[2][c],E[3][c],(E[1][c]-pd.to_datetime(E[1][c]).replace(day=1)).days/7))
        else:
            db.append((E[0][c],E[1][c],E[2][c],None,(E[1][c]-pd.to_datetime(E[1][c]).replace(day=1)).days/7))#,E[4][c])) 
    else:
        if E[3][c]!=0:
            db.append((E[0][c],None,E[2][c],E[3][c],None))#,E[4][c]))   

#se unen con la data historica          
df= pd.DataFrame(db, columns=['Centro', 'Mes', 'Controles', 'Variable','Semanas_mes']) 
dat=H.data_mes
dat.columns=['Centro','Mes','D var','D fix']  
mask=dat['Mes']<pd.to_datetime(  dat['Mes'].max()+relativedelta(months=-1)+ MonthEnd(1))
dtf=dat#[mask]


#Se guarda en un pandas listo para subir al servidor
Export=pd.concat([df,dtf], ignore_index=True)


#Para exportar datos a servidor sacar de comentarios lo siguiente, no utilizar para probar
"""
import pyodbc
import pandas as pd
from sqlalchemy import create_engine
from six.moves import urllib

tablaDestino="CP_Pronóstico_AP_JGH"
siExiste="replace"     #  "append"  o "fail"
params = urllib.parse.quote_plus('Driver={SQL Server};'                                
                            "Server=ACHS-ANALYTICAZ.achs.cl;"
                            "Database=az-capacity;"
                            "Trusted_Connection=yes;")
engine_az = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
Export.to_sql(name=tablaDestino,con=engine_az,method='multi',index=False,if_exists=siExiste, chunksize=123)

"""