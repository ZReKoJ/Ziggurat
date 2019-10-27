# Máster Universitario en Inteligencia Artificial
## Métodos de Simulación
## Enunciado 8

### 1. Generación de números y variables aleatorias
Describir el algoritmo	 de	
Ziggurat para	distribuciones	con	función	de	densidad	decreciente	y	compararlo	
con	otros	métodos	para	la	generación	de	valores	de	la	normal.
Utilizando	el	algoritmo	de	Ziggurat obtener	una	aproximación	de	la	tabla	de	la	
tabla	de	la	distribución	normal	estándar.

### 2. Simulación	de	sucesos	discretos
Llegan	petroleros	para	descargar	en	el	muelle	según	un	proceso	de	Poisson	no	
homogéneo	con	la	siguiente	tasa:
El	petrolero	llega	hasta	la	entrada	del	puerto,	y	espera	a	que	un	remolcador	
esté	 disponible	 y	 lo	 lleve	 hasta	 el	 muelle.	 Se	 disponen	 en	 el	 puerto	 de	 10	
remolcadores.
Los	remolcadores	también	realizan	la	labor	de	llevar	cada	petrolero	hasta	la	
entrada	del	puerto	tras	haber	descargado.	En el fichero “desplazamientos.txt”
se dispone de una muestra de las duraciones de los desplazamientos del
remolcador con el petrolero. Contrástese si la distribución de dichos tiempos es
normal (truncada), uniforme o exponencial y estímense los parámetros de la
distribución correspondiente.
Cuando	el	remolcador	va	de	vacío	(sin	remolcar)	la	distribución	es	también	
normal	pero	con	media	de	2	minutos	y	desviación	típica	1.	
Existe	 un	 número	 limitado	 de	 20 muelles	 donde	 pueden	 atracar	 los	
petroleros.	El	 tiempo	de	descarga	de	cada	petrolero	 tiene	una	distribución	
chi cuadrado con	2 grados	de	libertad,	expresada	en	horas.	
El	 remolcador	 da	 prioridad	 a	 los	 petroleros	 que	 llegan	 sobre	 los	 que	
abandonan	el	puerto.
A. Simule	el	comportamiento	del	puerto	para	estimar	el	tiempo	medio	
que	 tardan	 en	 atracar	 los	 barcos,	 el	 tiempo	 máximo	 en	 atracar,	 el	
número	medio	de	barcos	atracados	en	el	puerto	y	el	número	medio	y	
máximo	de	barcos	esperando	a	atracar.
B. Analice	 la	 posibilidad	 de	 disponer	 de	 3 nuevos remolcadores y	
realizar	 obras	 para	 disponer	 de	 5	 nuevos	 muelles	 ¿cuál	 de	 las	 dos	
opciones	es	mejor?

### 3. Aplicaciones	 de	 la	 simulación.	
Descripción	 de	 los	 métodos	 de	 remuestreo
(bootstrap	y	jackniffe).
