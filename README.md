<h1 align="left">GMM y RNN para el reconomiento de voz</h1>


En este documento exploramos dos métodos para la identificación de voces: el Modelo de Mezclas Gaussianas (GMM) y las redes neuronales profundas. En ambos casos usamos la información del Corpus de Lengua Oral del Español (CLOE), pero de manera diferente. Con el GMM comparamos dos grabaciones y obtenemos su Razón de Verosimilitud Logarítmica (LLR) para saber si se trata del mismo hablante. Con las redes neuronales, primero entrenamos un modelo utilizando los formantes de los individuos y luego calculamos el LLR de que un registro aleatorio esté presente en la base de datos. 
</p>


