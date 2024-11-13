# -*- coding: utf-8 -*-
"""
@author: YM
"""
import os
os.chdir("c:/мкс222_09_01/")
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


pth_a = './data/bike_sharing_dataset_day.csv'
BKSH = pd.read_csv(pth_a)
#обрезать датасет до нужных колонок
BS = BKSH[['workingday', 'weathersit', 'temp', 'hum','windspeed','cnt']].copy()

#гистограмма и график плотности распределения для температуры, страница 5, рисунок 1
sns.distplot(BS[['temp']], hist=True, kde=True, 
             bins='fd', color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3},
             axlabel = 'Температура от -8°C до +39°C')
plt.savefig('./graphics/bikes_temphist.png', format='png')
plt.show()
BS[['temp']].describe()
BS[['temp']].skew()
BS[['temp']].kurtosis()
#гистограмма для влажности, страница 4, рисунок 2
sns.distplot(BS[['hum']], hist=True, kde=True, 
             bins='fd', color = 'darkorange', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3},
             axlabel = 'Влажность, доля от 0 до 1')
plt.savefig('./graphics/bikes_humhist.png', format='png')
plt.show()
BS[['hum']].describe()
BS[['hum']].skew()
BS[['hum']].kurtosis()
#гистограмма для скорости ветра, страница 6, рисунок 3
sns.distplot(BS[['windspeed']], hist=True, kde=True, 
             bins='fd', color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3},
             axlabel = 'Нормализованная скорость ветра, от 0 до 67 километров в час')
plt.savefig('./graphics/bikes_windspeedhist.png', format='png')
plt.show()
BS[['windspeed']].describe()
BS[['windspeed']].skew()
BS[['windspeed']].kurtosis()
#гистограмма для аренды велосипедов, страница 7, рисунок 4
sns.distplot(BS[['cnt']], hist=True, kde=True, 
             bins='fd', color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3},
             axlabel = 'Количество арендованных велосипедов по дням')
plt.savefig('./graphics/bikes_cnthist.png', format='png')
plt.show()
BS[['cnt']].describe()
BS[['cnt']].skew()
BS[['cnt']].kurtosis()

#качественные данные: переименование категорий 
BKS = BS.copy()
BKS['workingday'] = BKS['workingday'].apply(lambda x: 'раб' if x==1 else 'вых' )
def recode(x):
    if x == 1:
        return 'ясно'
    elif x == 2:
        return 'облачно'
    elif x == 3:
        return 'осадки'
    elif x ==4:
        return 'шторм'
    else:
        pass
BKS['weathersit'] = BKS['weathersit'].apply(recode)
BKS = BKS.astype({'workingday':'category', 'weathersit':'category', 'temp':np.float64, 'hum':np.float64, 'windspeed':np.float64, 'cnt':np.float64})
BKS.dtypes

#качественные данные: столбчатые диаграммы, страница 8, рисунок 5
dfn = BKS.select_dtypes(include=['O', "category"]) 
plt.figure(figsize=(15, 9)) 
plt.subplots_adjust(wspace=0.5, hspace=0.5)
nplt = 0 
nrow = dfn.shape[1] 
for s in dfn.columns:
    nplt += 1
    ax = plt.subplot(nrow, 1, nplt)
    ftb = pd.crosstab(dfn[s], s) 
    ftb.index.name = 'Категории'

    ftb.plot.bar(ax=ax, grid=True, legend=False, title=s, colormap='Pastel1')
plt.savefig('./graphics/workingday_categories.png', format='png')

#страница 8, таблица 6
BKS['workingday'].value_counts()
BKS['weathersit'].value_counts()
#по итогам разведочного анализа перекодируем некоторые данные     
BKS['weathersit'] = BKS['weathersit'].replace('осадки', 'облачно')
BKS['weathersit'].value_counts()
#для нового разделения на категории посмотрим столбчатую диаграмму, страница 9, рисунок 6
BKS['weathersit'].value_counts().plot(kind='bar')
plt.savefig('./graphics/weathersit_TWOcategories.png', format='png')

#анализ связи
#между качественными объясняющими и количественной целевой, страница 9, рисунок 7
dfn = BKS.copy()
cols = dfn.select_dtypes(include='category').columns
nrow = len(cols)
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 12)
nplt = -1
for s in cols:
    nplt += 1
# Доверительные интервалы строятся методом бутстрепа    
    dfn.boxplot(column='cnt', by=s, ax=ax_lst[nplt], grid=True, notch=True, 
                bootstrap=50, showmeans=True, color=None)
fig.subplots_adjust(wspace=0.5, hspace=1.0)
# Общая подпись к графикам
fig.suptitle('Категоризированные диагарммы Бокса-Вискера')
plt.savefig('./graphics/bikes_whisker.png', format='png')

#критерий Крускала-Уоллиса для строгой проверки предположений, страница 10
from scipy.stats import kruskal
# Качественная переменная - 'workingday'
# Создаем подвыборки
sel_yes = BKS['workingday']=='вых'
x_1 = BKS.loc[sel_yes, 'cnt']
sel_no = BKS['workingday']=='раб'
x_2 = BKS.loc[sel_no, 'cnt']
# Используем криетрий Крускала-Уоллиса
bike_sig = kruskal(x_1, x_2)
# Сохраняем текстовый отчет
with open('./output/bikes_workingday_STAT.txt', 'w') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'workingday\' и \'cnt\'',
          file=fln)
    print(bike_sig, file=fln)
    
sel_yes = BKS['weathersit']=='ясно'
x_1 = BKS.loc[sel_yes, 'cnt']
sel_no = BKS['weathersit']=='облачно'
x_2 = BKS.loc[sel_no, 'cnt']
# Используем криетрий Крускала-Уоллиса
bike_sig = kruskal(x_1, x_2)
# Сохраняем текстовый отчет
with open('./output/bikes_weathersit_STAT.txt', 'w') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'weathersit\' и \'cnt\'',
          file=fln)
    print(bike_sig, file=fln)


#между количественными объясняющими и количественной целевой, страница 10, рисунок 8
dfn = BKS.select_dtypes(include='float64')
nrow = dfn.shape[1] - 1 # Учитываем, что одна переменная целевая - ось 'Y'
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 9) 
nplt = -1
for s in dfn.columns[:-1]: # Последняя переменная - целевая ('Y')
    nplt += 1
    dfn.plot.scatter(s, 'cnt', ax=ax_lst[nplt])
    ax_lst[nplt].grid(visible=True)
    ax_lst[nplt].set_title(f'Связь количества арендованных велосипедов с {s}')
fig.subplots_adjust(wspace=0.5, hspace=1.0)
"""
Общая подпись к графикам
Используем форматированные 'f'-строки
{} - позиция для подстановки значения
"""
fig.suptitle(f'Связь количества арендованных велосипедов с {list(dfn.columns[:-1])}')
plt.savefig('./graphics/bikes_scat.png', format='png')

# Анализ корреляции между количественными переменными, страницы 10-11, таблицы 8-9
# Используем библиотеку scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr

BK = BKS.select_dtypes(include='float')
# Здесь будут значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=BK.columns, columns=BK.columns) 
# Здесь будут значения значимости оценок коэффициента корреляции Пирсона
P_P = pd.DataFrame([], index=BK.columns, columns=BK.columns)
# Здесь будут значения оценок коэффициента корреляции Спирмена
C_S = pd.DataFrame([], index=BK.columns, columns=BK.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Спирмена
P_S = pd.DataFrame([], index=BK.columns, columns=BK.columns)
for x in BK.columns:
    for y in BK.columns:
        C_P.loc[x,y], P_P.loc[x,y] = pearsonr(BK[x], BK[y])
        C_S.loc[x,y], P_S.loc[x,y] = spearmanr(BK[x], BK[y])

# Сохраняем текстовый отчет на разные листы Excel файла
with pd.ExcelWriter('./output/bikes_STAT.xlsx', engine="openpyxl") as wrt:
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pirson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pirson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость
#страница 11, рисунок 9
corr = BK.corr(method = 'spearman')    
sns.heatmap(corr, annot = True)
plt.savefig('./graphics/bikes_heatmap.png', format='png')

#таблица корреляции и хи-квадрат для качественных переменных, страница 12, таблица 10
ct_table = pd.crosstab(BKS['workingday'], BKS['weathersit'])

dt = np.array([[75, 156], [193, 307]])
x = sp.stats.chi2_contingency(dt)[0]
#calculate Cramer's V 
n = np.sum(dt)
minDim = min(dt.shape)-1
np.sqrt((x / n) / minDim)

#моделирование
import statsmodels.api as sm
B = BKS.copy()
# Разбиение данных на тренировочное и тестовое множество
# frac- доля данных в тренировочном множестве
# random_state - для повторного отбора тех же элементов
B_train = B.sample(frac=0.8, random_state=42) 
# Символ ~ обозначает отрицание (not)
B_test = B.loc[~B.index.isin(B_train.index)] 

# Будем накапливать данные о качестве постреонных моделей
# Используем  adjR^2 и AIC
mq = pd.DataFrame([], columns=['adjR^2', 'AIC']) # Данные о качестве

"""
Постреоние базовой модели - таблица 11
Базовая модель - линейная регрессия, которая включает в себя 
все количественные переменные и фиктивные переменные дял качественных 
переменных с учетом коллинеарности. Для каждого качетсвенного показателя
включаются все уровни за исключением одного - базового. 
"""
# Формируем целевую переменную
Y = B_train['cnt']
# Формируем фиктивные (dummy) переменные для всех качественных переменных
DUM = pd.get_dummies(B_train[['workingday', 'weathersit']])
# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. ВЛияние включенных уровней на зависимую 
# переменную отсчитывается от него
DUM = DUM[['workingday_раб', 'weathersit_облачно']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X = pd.concat([DUM, B_train[['temp', 'hum', 'windspeed']]], axis=1)
# Добавляем переменную равную единице для учета константы
X = sm.add_constant(X)
X = X.astype({'const':'uint8'}) # Сокращаем место джля хранения константы
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg00 = sm.OLS(Y,X)
# Оцениваем модель
fitmod00 = linreg00.fit()
# Сохраняем результаты оценки в файл
with open('./output/bikes_STAT_woWeather.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod00.summary(), file=fln)
# Проверяем степень мультиколлинеарности только базовой модели, таблица 12
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X.select_dtypes(include='float64')# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/bikes_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    vif.to_excel(wrt, sheet_name='vif')
# Проверяем гетероскедастичность базовой модели, таблица 13
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod00.resid
WHT = pd.DataFrame(het_white(e, X), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/bikes_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod00.rsquared_adj, fitmod00.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_00']).T
mq = pd.concat([mq, q])    
#модификация для гипотезы 2, статус дня к погодным условиям, таблица 14
# Вводим переменную вщаимодействия
X_1 = X.copy()
X_1['ww'] = X_1['workingday_раб']*X_1['weathersit_облачно']
linreg02 = sm.OLS(Y,X_1)
fitmod02 = linreg02.fit()
# Сохраняем результаты оценки в файл
with open('./output/bikes_stat_ww.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod02.summary(), file=fln)
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod02.rsquared_adj, fitmod02.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])    
#модификация для гипотезы 3, влажность к температуре, таблица 15
X_2 = X.copy()
X_2['ht'] = X_1['temp']*X_1['hum']
linreg03 = sm.OLS(Y,X_2)
fitmod03 = linreg03.fit()
# Сохраняем результаты оценки в файл
with open('./output/bikes_stat_ht.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod03.summary(), file=fln) 
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod03.rsquared_adj, fitmod03.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])  
#модицификация для гипотез 2 и 3, таблица 16
X_12 = X.copy()
X_12['ww'] = X_12['workingday_раб']*X_12['weathersit_облачно']
X_12['ht'] = X_12['temp']*X_12['hum']
linreg04 = sm.OLS(Y,X_12)
fitmod04 = linreg04.fit()
# Сохраняем результаты оценки в файл
with open('./output/bikes_stat_pair.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod04.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod04.rsquared_adj, fitmod04.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])