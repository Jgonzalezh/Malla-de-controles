# -*- coding: utf-8 -*-


"""
Created on Tue Dec 31 09:24:35 2019

@author: jgonzalezh
"""

import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from pulp import *
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import random 
from pandas.tseries.offsets import MonthEnd
from datetime import date
from dateutil.relativedelta import relativedelta

start=time.process_time()

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

def data2():
    #Función para leer la data, pregunta a nuestro servidor de capacity por el nombre de la tabla que se solicite
    conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=ACHS-ANALYTICAZ.achs.cl;"
                      "Database=az-capacity;"
                      "Trusted_Connection=yes;")
    sql= """with base as (
            select Centro,fechadisponible, fechacalculo, ([HorizontedeCitas(dias)]) 
            from CP_HorizonteDeCitas_Hist
            where fechacalculo in (select max(FechaCalculo) from CP_HorizonteDeCitas_Hist)
            ), 
            promedio as(
            select Centro, (avg(([HorizontedeCitas(dias)]))) as PromedioHorizonteCitas
            from CP_HorizonteDeCitas_Hist
            where fechacalculo >= dateadd(week,-4,getdate())group by Centro
            )
            , prev as (
            select Centro, (avg(([HorizontedeCitas(dias)]))) as PromedioHorizonteCitas1
            from CP_HorizonteDeCitas_Hist
            where fechacalculo >= dateadd(week,-13,getdate()) and fechacalculo <= dateadd(week,-10,getdate())group by Centro
            )
            , post as (
            select Centro, (avg(([HorizontedeCitas(dias)]))) as PromedioHorizonteCitas2
            from CP_HorizonteDeCitas_Hist
            where fechacalculo >= dateadd(week,-3,getdate())group by Centro
            )
            
            select ba.*, PromedioHorizonteCitas,PromedioHorizonteCitas1,PromedioHorizonteCitas2 from base ba 
            left join promedio pro on ba.centro = pro.centro
            left join prev pv on ba.centro = pv.centro
            left join post pt on ba.centro = pt.centro"""
    #la consulta se hace a través de un select
    data=pd.read_sql(sql, conn)
    #guarda el archivo como una variable de python
    return data #devuelve la variable


#if data_dmd.iat[][]
class Herramienta:
    def __init__(self,data_dmd, data_turnos):# tabla_turnos):
        #lecturas tablas servidor
        self.data=data_dmd 
        #Variables auxiliares
        self.Centros=data_dmd.Centro.unique()
        fil=len(self.data.index) 
        self.N_CE=len(data_dmd.Centro.unique())
        ini=meses_3()
        t=3
        n=7
        g=2
        m=48
        dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'] 
        #Tipo=['Espontaneo', 'Controles']
        horas=['00:00:00','00:30:00','01:00:00','01:30:00','02:00:00','02:30:00','03:00:00','03:30:00',
            '04:00:00','04:30:00','05:00:00','05:30:00','06:00:00','06:30:00','07:00:00','07:30:00',
            '08:00:00','08:30:00','09:00:00','09:30:00','10:00:00','10:30:00','11:00:00','11:30:00',
            '12:00:00','12:30:00','13:00:00','13:30:00','14:00:00','14:30:00','15:00:00','15:30:00',
            '16:00:00','16:30:00','17:00:00','17:30:00','18:00:00','18:30:00','19:00:00','19:30:00',
            '20:00:00','20:30:00','21:00:00','21:30:00','22:00:00','22:30:00','23:00:00','23:30:00']
        #Variables de demanda variables y programada, D[centro][0=espontaneo;1=controles][dia][bloque]
        self.D=[[[[[0 for m in range(m)] for i in range(n)]for e in range(g) ]for t in range(t)] for c in range(self.N_CE)]
        #self.Q=[[[[0 for t in range(m)] for i in range(n)]for e in range(g) ]for c in range(self.N_CE)]
        self.Desv=[[[[[0 for m in range(m)] for i in range(n)]for e in range(g) ]for t in range(t)]for c in range(self.N_CE)]
        #llenar variables con la demanda

        for i in range(len(data_dmd.index)):
            for t in range(3):
                if ini[t]==data_dmd.iat[i,3]:
                    for c in range(self.N_CE):
                        if data_dmd.iat[i,0]==self.Centros[c]:
                            for d in range(7):
                                if data_dmd.iat[i,1]==dias[d]:
                                    for h in range(48):
                                        if data_dmd.iat[i,2]==horas[h]:
                                            #self.Q[c][0][d][h]+=data_dmd.iat[i,5]
                                            #self.Q[c][1][d][h]+=data_dmd.iat[i,6]
                                            self.D[c][t][0][d][h]+=data_dmd.iat[i,4]/30
                                            self.D[c][t][1][d][h]+=data_dmd.iat[i,5]/30
                                            self.Desv[c][t][0][d][h]+=data_dmd.iat[i,6]
                                            self.Desv[c][t][1][d][h]+=data_dmd.iat[i,7]
        #llenar las variables con los turnos
        FTE=1
        self.turno=data_turnos 
        turnos=self.turno
        fill=len(turnos.index)
        turnos['Hora Inicio']=turnos['Hora inicio Turno'].dt.hour
        turnos['Minutos Inicio']=turnos['Hora inicio Turno'].dt.minute
        turnos['Hora Fin']=turnos['Hora Fin Turno'].dt.hour
        turnos['Minutos Fin']=turnos['Hora Fin Turno'].dt.minute
        turnos['Hora Colacion In']=turnos['Hora Inicio Colación'].dt.hour
        turnos['Minutos colacion In']=turnos['Hora Inicio Colación'].dt.minute
        turnos['Hora Colacion fin']=turnos['Hora Fin Colación'].dt.hour
        turnos['Minutos colacion fin']=turnos['Hora Fin Colación'].dt.minute
        self.T=[[[0 for i in range(48)]for j in range(7)]for c in range(self.N_CE)]
        
        for g in range(fill):
            for c in  range(self.N_CE):
                if turnos.iat[g,0]==self.Centros[c]:
                    for h in range(7):
                        if (turnos.iat[g,11]+turnos.iat[g,12]/60)!=(turnos.iat[g,13]+ turnos.iat[g,14]/60):                        
                            if turnos.iat[g,2]==dias[h]:
                                for j in range(int(round(2*turnos.iat[g,9]+turnos.iat[g,10]/30))):
                                #for j in range(int(2*turnos.iat[g,11]+turnos.iat[g,12]/30)):
                                    if j>=int(round(2*turnos.iat[g,7]+turnos.iat[g,8]/30)) and j<int(round(2*turnos.iat[g,11]+turnos.iat[g,13]/30)): #revisa si tiene que ponerle -1
                                        self.T[c][h][j]+=FTE
                                    elif j>=int(round(2*turnos.iat[g,13]+turnos.iat[g,14]/30)) and j<=int(round(2*turnos.iat[g,9]+(turnos.iat[g,10]/30))):
                                        self.T[c][h][j]+=FTE
                        else:
                            if turnos.iat[g,2]==dias[h]:
                                for j in range(int(round(2*turnos.iat[g,9]+turnos.iat[g,10]/30))):
                                    if j>=int(round(2*turnos.iat[g,7]+turnos.iat[g,8]/30)) and j<=int(round(2*turnos.iat[g,9]+(turnos.iat[g,10]/30))): #revisa si tiene que ponerle -1
                                        self.T[c][h][j]+=FTE
        n=7
        self.hor_inicio=[[0 for i in range(self.N_CE)]for e in range(n)]
        self.hor_fin=[[47 for i in range(self.N_CE)]for e in range(n)]
        for c in range(self.N_CE):
            for h in range(7):
                for i in range(4,35):
                     if self.T[c][h][i]>0 and self.T[c][h][i-1]==0 and self.T[c][h][i-2]==0 and self.T[c][h][i-3]==0:
                         self.hor_inicio[h][c]=i
                         #Hora de iniciio de atención del centro según turnos
                for i in range(25,45):
                    if self.T[c][h][i+3]==0 and self.T[c][h][i+2]==0 and self.T[c][h][i+1]==0 and self.T[c][h][i]>0:
                         self.hor_fin[h][c]=i   
    def plot(self,c,m):
        # Graficar
        witdh=0.7
        #largo de las barras
        ## Posible utilidad posterior: label2=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','45','46','47','48']
        label=['00:00',' ','01:00','','02:00',' ','03:00',' ',
        '04:00',' ','05:00',' ','06:00',' ','07:00',' ',
        '08:00',' ','09:00',' ','10:00',' ','11:00',' ',
        '12:00',' ','13:00',' ','14:00',' ','15:00',' ',
        '16:00',' ','17:00',' ','18:00',' ','19:00',' ',
        '20:00',' ','21:00',' ','22:00',' ','23:00',' ']
        plt.style.use('default')
        #se agregan el label del eje x
        index = np.arange(len(label)) # array([0,1,2,...,47])
        #lista del largo del label
        #plt.close()  
        Z=plt.figure() 
        Z.set_facecolor('lightgoldenrodyellow')
        #se abre el plot antes (al pnerlo ah{i me arreglo un problema de ploteo})
        dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        #Días de la semana para títulos de los diferentes graficos
        for i in range(7):
            #Iteración para gráficar lso 7 días de la semana
            #x=2*i
            #=2*i+1#,n in enumerate(7):
            Esp=self.D[c][m][0][i][:] #Demanda espontanea del d{ia de la semana i}
            Cit1=self.D[c][m][1][i][:] #Demanda programada de día i 
            Desv=self.Desv[c][m][0][i][:]
            plt.tight_layout()
            plt.subplot(421+i) #se grafica en un marco de 2 (horizontal) x 4 vertical, posición 1+i (tiene 8 posiciones)
            p1=plt.bar(index, np.array(Esp), witdh,  yerr=Desv, color='darkkhaki',capsize=2, ecolor='darkslateblue')#'mediumaquamarine') #'seagreen')#Gráfico de barras de espontaneos
            p2=plt.bar(index, np.array(Cit1), witdh, bottom=np.array(Esp), color='seagreen')
            #p3=plt.bar(index, np.array(Cit2), witdh, bottom=np.array(Cit1)+np.array(Esp),color='lightskyblue')## yerr=DEsp,color='lightskyblue',capsize=2 , ecolor='darkslateblue')
            #'lawngreen')#'mediumseagreen')
            #p2=plt.bar(index, Cit, witdh, bottom=Esp, yerr=DEsp,color='mediumseagreen' ,capsize=2 , ecolor='darkslateblue') #grafico de barra, agregar citados sobre espontaneos
            turnos=self.T[c][i][:]
            T2=[[0 for t in range(48)]for d in range(7)]
            for j in range(48):
               T2[i][j]=self.T[c][i][j]*0.83
            #Turnos por trabajar
            p5=plt.plot(index, turnos,'mediumaquamarine')#'salmon') #se gráfican 
            p6=plt.plot(index, T2[i][:] ,linestyle=':',color='mediumaquamarine')#'salmon') #se gráfican 
            #print('Grafico') #mostrar el minuto en que se graficaban todos
            plt.title(dias[i])
            #Se agrega el título por gráfico
            plt.subplots_adjust(hspace=0.4)
            #Se define el espacio entre gráficos
            plt.ylabel('Demanda', fontsize=12)
            #Label eje y
            plt.xticks(index, label, fontsize=6, rotation=90)
            #se agregan las horas del día para el eje x, fuente tamaño 5 y rotación 90 grados
            plt.suptitle(self.Centros[c], fontsize=14)
            plt.legend((p1[0],p2[0],p5[0],p6[0]), ('Espontaneo','Citados','Turnos','Utilización meta') ,loc='upper left',prop={'size':8})
                            #Se agrega loa leyenda para cada una de las variables
            #posición de la leyenda
            #plt.grid()
        plt.show()
    def H_cita(self,c):
        HC=data2()
        #c='Alameda'
        Centros2=HC.Centro.unique()
        m=buscar_centro(self.Centros[c],Centros2)
        if m!=None:
            HC.columns
            horizonte_cita=HC.iat[m,4]
            HC_prev=HC.iat[m,5]
            HC_post=HC.iat[m,6]
            percent=1 #para ver cuanto le agreo semanal
            if HC_prev<HC_post:
                #cuanto crece horizonte cita
                percent+=((HC_post-HC_prev)/4)/3 #dividido en 4 para saber cuanto están haciendo aumentar su demanda del horizonte cita en el periodo
                #dividido en tres porque son bloques de tres semanas (promedio semana -4 -5 y -6 vs la -1 -2 y -3)(0 actual)
            if horizonte_cita>4:
                percent+=((horizonte_cita-4)/4)/13 #Arreglar el procentaje maYOR EN EL PERIODO DE 13 SEMANAS
            #print('HC',horizonte_cita,'HCprev',HC_prev,'HCpost',HC_post)
            return percent
    def H_cita2(self,c):
        HC=data2()
        #c='Alameda'
        Centros2=HC.Centro.unique()
        m=buscar_centro(self.Centros[c],Centros2)
        if m!=None:
            HC.columns
            horizonte_cita=HC.iat[m,4]
            return horizonte_cita
        
    def plot_opt(self,C,c,mes):
        # Graficar
        witdh=0.7
        #largo de las barras
        ## Posible utilidad posterior: label2=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','45','46','47','48']
        label=['00:00',' ','01:00',' ','02:00',' ','03:00',' ',
        '04:00',' ','05:00',' ','06:00',' ','07:00',' ',
        '08:00',' ','09:00',' ','10:00',' ','11:00',' ',
        '12:00',' ','13:00',' ','14:00',' ','15:00',' ',
        '16:00',' ','17:00',' ','18:00',' ','19:00',' ',
        '20:00',' ','21:00',' ','22:00',' ','23:00',' ']
        plt.style.use('default')
        #se agregan el label del eje x
        index = np.arange(len(label)) # array([0,1,2,...,47])
        #lista del largo del label
        #plt.close()  
        Z=plt.figure() 
        Z.set_facecolor('lightgoldenrodyellow')
        #se abre el plot antes (al pnerlo ah{i me arreglo un problema de ploteo})
        dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        #Días de la semana para títulos de los diferentes graficos
        for i in range(5):
            #Iteración para gráficar lso 7 días de la semana
            #x=2*i
            #=2*i+1#,n in enumerate(7):
            Esp=self.D[c][mes][0][i][:] #Demanda espontanea del dia de la semana i
            Cit1=self.D[c][mes][1][i][:] #Demanda programada de día i 
            Cit=[0 for t in range(48)]
            Desv=self.Desv[c][mes][0][i][:]
            for j in range(48):
                for m in range(10):
                     Cit[j]+=C[(i,j,m)].varValue/2
                    
                """"    
                if type(C[(i,j)].varValue)!=type(None):
                    Cit[j]=+C[(i,j)].varValue/2
                else:
                    Cit[j]=+0
                """
            plt.tight_layout()
            plt.subplot(421+i) #se grafica en un marco de 2 (horizontal) x 4 vertical, posición 1+i (tiene 8 posiciones)
            p1=plt.bar(index, np.array(Esp), witdh,  yerr=Desv, color='darkkhaki',capsize=2, ecolor='darkslateblue')
            #p1=plt.bar(index, np.array(Esp), witdh, color='darkkhaki')#'mediumaquamarine') #'seagreen')#Gráfico de barras de espontaneos
            p2=plt.bar(index, np.array(Cit), witdh, bottom=np.array(Esp), color='seagreen')
            #p3=plt.bar(index, np.array(Cit2), witdh, bottom=np.array(Cit1)+np.array(Esp),color='lightskyblue')## yerr=DEsp,color='lightskyblue',capsize=2 , ecolor='darkslateblue')
            #'lawngreen')#'mediumseagreen')
            #p2=plt.bar(index, Cit, witdh, bottom=Esp, yerr=DEsp,color='mediumseagreen' ,capsize=2 , ecolor='darkslateblue') #grafico de barra, agregar citados sobre espontaneos
            turnos=self.T[c][i][:]
            T2=[[0 for t in range(48)]for d in range(5)]
            for j in range(48):
               T2[i][j]=self.T[c][i][j]*0.83
            #Turnos por trabajar
            p5=plt.plot(index, turnos,'mediumaquamarine')#'salmon') #se gráfican 
            p6=plt.plot(index, T2[i][:] ,linestyle=':',color='mediumaquamarine')#'salmon') #se gráfican 
            #print('Grafico') #mostrar el minuto en que se graficaban todos
            plt.title(dias[i])
            #Se agrega el título por gráfico
            plt.subplots_adjust(hspace=0.4)
            #Se define el espacio entre gráficos
            plt.ylabel('Demanda', fontsize=12)
            #Label eje y
            plt.xticks(index, label, fontsize=6, rotation=90)
            #se agregan las horas del día para el eje x, fuente tamaño 5 y rotación 90 grados
            plt.suptitle(self.Centros[c]+" Optimización Malla", fontsize=14)
            plt.legend((p1[0],p2[0],p5[0],p6[0]), ('Espontaneo','Citados','Turnos','Utilización meta') ,loc='upper left',prop={'size':8})
                            #Se agrega loa leyenda para cada una de las variables
            #posición de la leyenda
            #plt.grid()
        """a=plt.subplot(428)
        clust_data = np.random.random((10,3)) 
        collabel=("col 1", "col 2", "col 3") 
        a.axis('tight')
        a.axis('off')
        table=plt.table(cellText=clust_data,colLabels=collabel,fontsize=32,loc='center') 
        #table.set_fontsize(24)
        table.scale(1,1)"""
        plt.show()       
def suma(M):
    x=[0,0,0,0,0,0,0]
    for j in range(7):
        for i in range(48): 
            x[j]+=M[1][j][i]
    return x

def sumaobjetivo(C,D,T,c,F,P,DESV,mes):
    x=pulp.lpSum([(pulp.lpSum([C[(d,i,m)]for m in range(10)]))*(100)+4*pulp.lpSum([m/(m+3)*C[(d,i,m)]for m in range(10)])+(DESV[c][mes][0][d][i])*pulp.lpSum([C[(d,i,m)]for m in range(10)]) for i in range(48) for d in range(5)])
    #x=pulp.lpSum([(-pulp.lpSum([C[(d,i,m)]for m in range(10)]))*(100)+(5*DESV[c][mes][0][d][i])*pulp.lpSum([C[(d,i,m)]for m in range(10)]) for i in range(48) for d in range(5)])#10*pulp.lpSum([m*C[(d,i,m)]for m in range(10)])
    return x
def sumaobjetivo2(C,D,T,c,F,P,DESV,mes):
    x=pulp.lpSum([(pulp.lpSum([-C[(d,i,m)]for m in range(10)]))*(100)+4*pulp.lpSum([m/(m+3)*C[(d,i,m)]for m in range(10)])+(DESV[c][mes][0][d][i])*pulp.lpSum([C[(d,i,m)]for m in range(10)]) for i in range(48) for d in range(5)])
    #x=pulp.lpSum([(-pulp.lpSum([C[(d,i,m)]for m in range(10)]))*(100)+(5*DESV[c][mes][0][d][i])*pulp.lpSum([C[(d,i,m)]for m in range(10)]) for i in range(48) for d in range(5)])#10*pulp.lpSum([m*C[(d,i,m)]for m in range(10)])
    return x
    
def optimizacion(clase_dmd,c, mes):
    Q_controles=[0,1,2,3,4,5,6,7,8,9] #se crea una dimensión cantidad de controles, máximo 9 (es la trampa para alisar sin fn cuadratica)
    dias=[0,1,2,3,4] # se hace el ppl de lunes a viernes
    index = np.arange(48) #bloques del día
    modelo=LpProblem('Controles y turnos', LpMinimize) # Se define el modelo
    D=clase_dmd.D # Demanda
    T=clase_dmd.T #Turnos
    #Q=clase_dmd.Q
    DESV=clase_dmd.Desv #desviación estándar del model 
    ini=[0 for t in range(5)] #principio primer turno 
    fin=[0 for t in range(5)] #fin último turno
    E=[[0 for i in range(48)]for j in range(7)]
    tipo=0
    resultado=['Óptimo diario', 'Óptimo semanal', 'Máxima malla posible', 'Sin turnos']
    for d in dias:
        ini[d]=clase_dmd.hor_inicio[d][c]
        fin[d]=clase_dmd.hor_fin[d][c]
    print(ini, 'inicio')
    print(fin, 'fin')
    
    if (ini[0]==0 and fin[0]==47) or (ini[1]==0 and fin[1]==47) or (ini[2]==0 and fin[2]==47) or (ini[3]==0 and fin[3]==47) or (ini[4]==0 and fin[4]==47):
        print ('sin turnos',clase_dmd.Centros[c])   
        tipo=3
    else:
        tipo=0
        if clase_dmd.H_cita(c)!=None:
            H_C=clase_dmd.H_cita(c)
        else:
            H_C=1
        print(H_C, 'H C')
        F=2 #Controles que caben en un bloque 
        P=0.85 #Productividad
        x=list(range(48))
        C=LpVariable.dicts('Controles propuestos', [(j,i,m)for i in index for j in dias for m in Q_controles],lowBound=0, upBound=None,cat='Binary')
        modelo+=sumaobjetivo(C,D,T,c,F,P,DESV,mes)
        for d in dias:

            modelo+=pulp.lpSum([F*D[c][mes][1][d][i]*H_C*1.16 for i in index])<=pulp.lpSum([C[(d,i,m)] for i in index for m in Q_controles])   
            modelo+=pulp.lpSum([C[(d,fin[d],m)]+C[(d,fin[d]-1,m)]for m in Q_controles])<=0
            for i in index:
                if T[c][d][i]>D[c][mes][0][d][i]:
                    #Siempre cuando no haya demanda espontanea sin capacidad de turno
                    modelo+=F*D[c][mes][0][d][i]+pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=F*T[c][d][i]*1.20 #no se puede superar la capacidad más de un 20%
                else:
                    modelo+=pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=0 
            if T[c][d][ini[d]]*P>D[c][mes][0][d][ini[d]]:
                modelo+=pulp.lpSum([C[(d,ini[d],m)]for m in Q_controles])<=F*T[c][d][ini[d]]*P-F*D[c][mes][0][d][ini[d]]
            else:
                modelo+=pulp.lpSum([C[(d,ini[d],m)]for m in Q_controles])<=0
            for i in index[ini[d]:fin[d]+1]:
                if D[c][mes][0][d][i]+D[c][mes][0][d][i-1]<=(T[c][d][i]+T[c][d][i-1])*P:
                    modelo+=pulp.lpSum([D[c][mes][0][d][j]*F+pulp.lpSum([C[(d,j,m)]for m in Q_controles]) for j in [i-1,i]])<=pulp.lpSum([T[c][d][j]*F*P for j in [i-1,i]])
                else: 
                    modelo+=pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=0 
            
    
        modelo.solve(pulp.PULP_CBC_CMD(maxSeconds=60, msg=1, fracGap=0))#pulp.GLPK()
        print(LpStatus[modelo.status])
        if LpStatus[modelo.status]=='Infeasible' or LpStatus[modelo.status]=='Not Solved' :
            tipo=1
            modelo2=LpProblem('Controles y turnos', LpMinimize)    
            C=LpVariable.dicts('Controles propuestos', [(j,i,m)for i in index for j in dias for m in Q_controles],lowBound=0, upBound=None,cat='Binary')
            modelo2+=sumaobjetivo(C,D,T,c,F,P,DESV,mes)
            modelo2+=pulp.lpSum([F*D[c][mes][1][d][i]*H_C*1.15 for i in index] for d in dias)<=pulp.lpSum([C[(d,i,m)] for i in index for m in Q_controles for d in dias])   
            for d in range(5):
                modelo2+=pulp.lpSum([C[(d,fin[d],m)]+C[(d,fin[d]-1,m)]for m in Q_controles])<=0
                for i in index:
                    if T[c][d][i]>D[c][mes][0][d][i]:
                        #Siempre cuando no haya demanda espontanea sin capacidad de turno
                        modelo2+=F*D[c][mes][0][d][i]+pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=F*T[c][d][i]*1.20 #no se puede superar la capacidad más de un 20%
                    else:
                        modelo2+=pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=0 
                if T[c][d][ini[d]]*P>D[c][mes][0][d][ini[d]]:
                    modelo2+=pulp.lpSum([C[(d,ini[d],m)]for m in Q_controles])<=F*T[c][d][ini[d]]*P-F*D[c][mes][0][d][ini[d]]
                else:
                    modelo2+=pulp.lpSum([C[(d,ini[d],m)]for m in Q_controles])<=0
                for i in index[ini[d]:fin[d]+1]:
                    if D[c][mes][0][d][i]+D[c][mes][0][d][i-1]<=(T[c][d][i]+T[c][d][i-1])*P:
                        modelo2+=pulp.lpSum([D[c][mes][0][d][j]*F+pulp.lpSum([C[(d,j,m)]for m in Q_controles]) for j in [i-1,i]])<=pulp.lpSum([T[c][d][j]*F*P for j in [i-1,i]])
                    else: 
                        modelo2+=pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=0 
            modelo2.solve(pulp.PULP_CBC_CMD(maxSeconds=60, msg=1, fracGap=0))
            print(LpStatus[modelo2.status])
            modelo=modelo2
            
            if LpStatus[modelo2.status]=='Infeasible' or LpStatus[modelo2.status]=='Not Solved' :
                tipo=2
                modelo3=LpProblem('Controles y turnos', LpMinimize)    
                C=LpVariable.dicts('Controles propuestos', [(j,i,m)for i in index for j in dias for m in Q_controles],lowBound=0, upBound=None,cat='Binary')
                modelo3+=sumaobjetivo2(C,D,T,c,F,P,DESV,mes)
                #modelo3+=pulp.lpSum([F*D[c][mes][1][d][i]*H_C for i in index] for d in dias)+5<=pulp.lpSum([C[(d,i,m)] for i in index for m in Q_controles for d in dias])   
                    
                for d in range(5):
                    modelo3+=pulp.lpSum([C[(d,fin[d],m)]+C[(d,fin[d]-1,m)]for m in Q_controles])<=0
                    for i in index:
                        if T[c][d][i]>D[c][mes][0][d][i]:
                            #Siempre cuando no haya demanda espontanea sin capacidad de turno
                            modelo3+=F*D[c][mes][0][d][i]+pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=F*T[c][d][i]*1.20 #no se puede superar la capacidad más de un 20%
                        else:
                            modelo3+=pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=0 
                    if T[c][d][ini[d]]*P>D[c][mes][0][d][ini[d]]:
                        modelo3+=pulp.lpSum([C[(d,ini[d],m)]for m in Q_controles])<=F*T[c][d][ini[d]]*P-F*D[c][mes][0][d][ini[d]]
                    else:
                        modelo3+=pulp.lpSum([C[(d,ini[d],m)]for m in Q_controles])<=0
                    for i in index[ini[d]:fin[d]+1]:
                        if D[c][mes][0][d][i]+D[c][mes][0][d][i-1]<=(T[c][d][i]+T[c][d][i-1])*P:
                            modelo3+=pulp.lpSum([D[c][mes][0][d][j]*F+pulp.lpSum([C[(d,j,m)]for m in Q_controles]) for j in [i-1,i]])<=pulp.lpSum([T[c][d][j]*F*P for j in [i-1,i]])
                        else: 
                            modelo3+=pulp.lpSum([C[(d,i,m)]for m in Q_controles])<=0                                     
                modelo3.solve()
                print(LpStatus[modelo3.status])
                modelo=modelo3
        
        #
        print('\n'+clase_dmd.Centros[c]," ",LpStatus[modelo.status],resultado[tipo],'mes',mes)
        DS=['Lunes','Martes','Miércoles','Jueves','Viernes']
        L=0
        M=0
        Mi=0
        J=0
        V=0
        L2=0
        M2=0
        Mi2=0
        J2=0
        V2=0
        TL=0
        TM=0
        TMi=0
        TJ=0
        TV=0
        L1=0
        M1=0
        Mi1=0
        J1=0
        V1=0
        for i in index:
            L+=D[c][mes][1][0][i]*2
            L1+=D[c][mes][0][0][i]
            TL+=T[c][0][i]
            for m in Q_controles:
                L2+=C[(0,i,m)].varValue
            M+=D[c][mes][1][1][i]*2
            TM+=T[c][1][i]
            M1+=D[c][mes][0][1][i]
            for m in Q_controles:
                M2+=C[(1,i,m)].varValue
            Mi+=D[c][mes][1][2][i]*2
            TMi+=T[c][2][i]
            Mi1+=D[c][mes][0][2][i]
            for m in Q_controles:
                Mi2+=C[(2,i,m)].varValue
            J+=D[c][mes][1][3][i]*2
            TJ+=T[c][3][i]
            J1+=D[c][mes][0][3][i]
            for m in Q_controles:
                J2+=C[(3,i,m)].varValue
            V+=D[c][mes][1][4][i]*2
            TV+=T[c][4][i]
            V1+=D[c][mes][0][4][i]
            for m in Q_controles:
                V2+=C[(4,i,m)].varValue
        print("Detalle de controles\nDía\t\t","Controles\t","Controles Propuestos\t","Utilización Actual/Propuesta") 
        print(DS[0],'\t\t',L,'\t\t',L2, '\t\t\t',round(((L/2)+L1)/TL,2),'-',round(((L2/2)+L1)/TL,2),'\n'+DS[1],'\t\t',M,'\t\t',M2, '\t\t\t',round(((M/2)+M1)/TM,2),'-',round(((M2/2)+M1)/TM,2))
        print(DS[2],'\t',Mi,'\t\t',Mi2, '\t\t\t',round(((Mi/2)+Mi1)/TMi,2),'-',round(((Mi2/2)+Mi1)/TMi,2),'\n'+DS[3],'\t\t',J,'\t\t',J2, '\t\t\t',round(((Mi/2)+Mi1)/TMi,2),'-',round(((J2/2)+J1)/TJ,2),'\n'+DS[4],'\t',V,'\t\t',V2, '\t\t\t',round(((V/2)+V1)/TV,2),'-',round(((V2/2)+V1)/TV,2))
        FTE=0
        DMD=0
        DMD2=0
        for d in dias:
            for i in index:
                FTE+=T[c][d][i]/2
                DMD+=(D[c][mes][0][d][i]+D[c][mes][1][d][i])/2
                DMD2+=(D[c][mes][0][d][i])/2
                for m in Q_controles:
                    DMD2+=(C[(d,i,m)].varValue/2)/2
        if FTE>0:
            print("\nFTE:\t\t",round(FTE/45,2),"\nDemanda semanal:", round(DMD/45,2),"\nUtilización\t", round(DMD/FTE,2),'%',"\nUtilización2\t", round(DMD2/FTE,2),'%','\nHorizonte Cita:\t',clase_dmd.H_cita2(c), 'días' ) # clase_dmd.H_cita(c),
        else:
            print("Sin turnos\nDemanda semanal:\t\t", round(DMD/45,2) )  
         
        for i in range(48):
            for d in range(5):
                for m in range(10):
                   E[d][i]+=C[(d,i,m)].varValue
    #clase_dmd.plot_opt(C,c,mes)
    return [E, mes, resultado[tipo]]
def meses_3():
    ini=[0 for r in range(3)]
    hoy= pd.to_datetime('today').replace(hour=0,minute=0,second=0,microsecond=0,nanosecond=0)
    hoy_inicio_mes=hoy.replace(day=1)
    ini[0]= hoy_inicio_mes + relativedelta(months=1) + MonthEnd(1)
    ini[1]= hoy_inicio_mes + relativedelta(months=2) + MonthEnd(1)
    ini[2]= hoy_inicio_mes + relativedelta(months=3)+ MonthEnd(1)
    return ini

def buscar_centro(Centro,centros):
    indice=0
    while indice<len(centros):
        if Centro==centros[indice]:
            break
        else:
            indice+=1 
            if indice==len(centros):
                indice=None
                break
    return indice
dataHC=data2()
data_dmd=data("CP_detalleDMD_pronostico_JGH")
data_turnos=data("[DI_turnosAP_10-19]")
H=Herramienta(data_dmd,data_turnos) 
dmd=H.D
Tur=H.T
desv=H.Desv
Centros=H.Centros
#print(*dmd)
#len(dmd)
lenghts=np.shape(dmd)
lenghts2=np.shape(Tur)
lenghts3=np.shape(desv)
dimen=len(lenghts)

#opti=optimizacion(H,13,0)
#opti2=optimizacion(H,13,1)


horas=['00:00:00','00:30:00','01:00:00','01:30:00','02:00:00','02:30:00','03:00:00','03:30:00',
            '04:00:00','04:30:00','05:00:00','05:30:00','06:00:00','06:30:00','07:00:00','07:30:00',
            '08:00:00','08:30:00','09:00:00','09:30:00','10:00:00','10:30:00','11:00:00','11:30:00',
            '12:00:00','12:30:00','13:00:00','13:30:00','14:00:00','14:30:00','15:00:00','15:30:00',
            '16:00:00','16:30:00','17:00:00','17:30:00','18:00:00','18:30:00','19:00:00','19:30:00',
            '20:00:00','20:30:00','21:00:00','21:30:00','22:00:00','22:30:00','23:00:00','23:30:00']
dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'] 


db=[]
for i in range(lenghts[0]):
    print(H.Centros[i],i)


for c in range(lenghts[0]):
    for mes in range(lenghts[1]):
        opti=optimizacion(H,c,mes)
        E=opti[0]
        mes=opti[1]
        tipo=opti[2]
        for d in range(lenghts[3]):
            for b in range(lenghts[4]):
                db.append((Centros[c],meses_3()[mes].month_name(locale='Spanish'),tipo,dias[d],horas[b],round(dmd[c][mes][0][d][b],2),round(dmd[c][mes][1][d][b],2),round(E[d][b]/2,2),round(Tur[c][d][b],2),round(desv[c][mes][0][d][b],2)))
                
df= pd.DataFrame(db, columns=['Centro','Mes','Tipo optimo','Dia','Hora','Demanda variable','Demanda controles','Controles propuestos','Turnos', 'Desviación variable'])    


import pyodbc
import pandas as pd
from sqlalchemy import create_engine
from six.moves import urllib

tablaDestino="CP_MallaControles_AP_JGH"
siExiste="replace"     #  "append"  o "fail"
params = urllib.parse.quote_plus('Driver={SQL Server};'                                
                            "Server=ACHS-ANALYTICAZ.achs.cl;"
                            "Database=az-capacity;"
                            "Trusted_Connection=yes;")
engine_az = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
df.to_sql(name=tablaDestino,con=engine_az,method='multi',index=False,if_exists=siExiste, chunksize=123)

"""

for i in range(len(H.Centros)):
    plt.close()

p=H.hor_inicio[0][0]
H.hor_fin[0][0]

Centro_c_t=[4,10,27,15,28,79]#[10,66,76,63,13]#11,43,53,19,

#for i in Centro_c_t:
for i in range(A):
    print('principio')
    E=optimizacion(H,i,mes=0)
    E=optimizacion(H,i,mes=1)
    E=optimizacion(H,i,mes=2)
    print('final')

    #print(*optimizacion(H,i))
"""
"""
a=H.data.to_numpy()
type(H.data)
type(H.data['Mes'])
type(H.data.iat[9,3])
ini=[0 for r in range(3)]
hoy= pd.to_datetime('today').replace(hour=0,minute=0,second=0,microsecond=0,nanosecond=0)
hoy_inicio_mes=hoy.replace(day=1)
ini[0]= hoy_inicio_mes + relativedelta(months=1) + MonthEnd(1)
ini[1]= hoy_inicio_mes + relativedelta(months=2) + MonthEnd(1)
ini[2]= hoy_inicio_mes + relativedelta(months=3)+ MonthEnd(1)


#H.data.iat[9,3]==ini[0]

end=time.process_time()
print(end-start)
buscar_centro('Chillán',H.Centros)
#buscar_centro('Viña del Mar',H.Centros)
#buscar_centro('Parque Las Américas',H.Centros)
#buscar_centro('La Serena',H.Centros)
#buscar_centro('Vespucio Oeste',H.Centros)
#buscar_centro('Puente Alto',H.Centros)
#buscar_centro('Puerto Montt',H.Centros)
#buscar_centro('Concepción',H.Centros)
#buscar_centro('Curicó',H.Centros)


for y in [11,81,43,53,19,71]:
    print(*H.Desv[y][0][:][:])

H.hor_inicio[:][6]
H.hor_fin[6]

HC=data2()
#c='Alameda'
Centros2=HC.Centro.unique()
len(Centros2)       
for c in H.Centros:
    print (c)
    #m=buscar_centro(c,Centros2)
    X=H.H_cita2(15)
    X=H.H_cita(15)
    print(X,Centros2[15])
    buscar_centro('Iquique',Centros)
for i in range(7):
    print(H.hor_inicio[1][13])
    print(H.hor_fin[1][13])
import pandas as pd
df=pd.DataFrame(data=[1,2,3,4], columns=['a'])

"""
