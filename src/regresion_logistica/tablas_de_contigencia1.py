import pandas as pd


data = pd.read_csv('../../python-ml-course/datasets/gender-purchase/Gender Purchase.csv')

print (data.head())
print ('Numero de datos:', data.shape)

contingency_table = pd.crosstab(data['Gender'], data['Purchase'])
print ('--Contigency table--')
print (contingency_table)

print ('-- axisc=1')
# suma horizontalmente(filas)
print (contingency_table.sum(axis=1))
print ('-- axisc=0')
# suma vertical(columnas)
print (contingency_table.sum(axis=0))

print ('--')
prob = contingency_table.astype('float').div(contingency_table.sum(axis=1), axis=0)
print (prob)



# Probabilidad condicional
## Ccal es la probabilidad de que un cliente compre un producto sabiendo que es hombre?
print (121 / 246)
## Cual es la probabilidad de que sabiendo que un cliente compra un producto sea mujer?
print (159 / 280)

# NOTA: al final de la prueba hay 4 probabilidades que queremos saber
# para ajaecutar unas accion, las cuales son:
# P(Purchase | Male)  121/246
# P(NO Purchase | Male) 125/246
# P(Purchase | Famale) 159/265
# P(NO Purchase | female) 106/265
# que en conclusion son los mismo resultados que arroja la variable "prob"


# Siguiente concepto: Ratio de probabilidades
## Cociente entre los casos de exito sobre los de fracaso en el suceso estudiado 
#y para cada grupo

ratio_m = 121/125
ratio_f = 159/106

print (ratio_m, ratio_f)

# Si el ratio es superior a 1, es mas probable el exito que el fracaso
# Si el ratio es igual a 1, exito y fracaso son equiprobables (p=0.5)

cociente_m = ratio_m/ratio_f 
cociente_f = 1/cociente_m

print (cociente_m, cociente_f)