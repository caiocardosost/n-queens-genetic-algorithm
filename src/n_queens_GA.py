# -*- coding: utf-8 -*-
"""
-----------Algoritmo Genético para resolver o problema das n rainhas-----------

- A ideia é gerar uma população inicial composta por indivíduos que representam
soluções candidatas. Cada individuo corresponde a uma permutação dos números de
0 a n-1, onde:
    - indice do vetor representa a coluna do tabuleiro
    - o valor nessa posição representa a linha onde a rainha está posicionada
    - n é o tamanho do tabuleiro
- Em seguida, utiliza-se uma função de fitness que que avalia o quão boa é uma 
solução candidata, verificando transgressões às restrições do problema. Essa função
Calcula o número de colisões entre rainhas, de modo que as soluções com menos
colisões são consideradas melhores. O objetivo do algoritmo é atingir a convergencia
da população para indivíduos com menor número de colisões.
- Não utilizamos um dataset nessa solução devido a natureza combinatória e a existencia
de diversas soluções possiveis para cada instancia de entrada, assim a avaliação da solução 
é realizada diretamente por meio da dunção de fitness, que é baseada na restição do problema

"""

#@authors: Caio Cardoso, Joel da Silva, Victor Ravazio

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import psutil

# obter metricas de consumo de memoria e cpu

process = psutil.Process(os.getpid()) 

process.cpu_percent(None) # inicializa

#-------------------------Calculo de fitness------------------------
""" Para o calculo do fitness consideramos o quão correta está a populção candidata
- No problema das n rainhas, uma permutação é considerada correta se satisfaz as seguintes condicoes:
    # 1 - Não existem duas rainhas na mesma linha.
    # 2 - Não existem duas rainhas na mesma coluna.
    # 3 - Não existem duas rainhas na mesma diagonal.

 Como cada individuo gerado na populção consiste em permutação sem repetição,
então automaticamente as condições 1 e 2 são satisfeitas, restando testar "3".
Assim, o fitness é considerado aqui como o numero de colisões (rainhas na mesma diagonal),
de forma que uma solução é otima se o numero de colisões diagonais é zero.
"""
def calc_fitness(solution):
    conflitos = 0
    n = len(solution)
    
    for i in range(n):
        for j in range(i+1, n):
            if abs(solution[i] - solution[j]) == abs(i - j): #verificando colisão
                conflitos += 1
    
    return conflitos  


# ---------------------------Algoritmo Genético -----------------------
    #função que realiza a execução do algoritmo genético
def execute_AG(fitness_hist, elapsed_times, nume_epochs_execute, cpu_consume, ram_consume, peak_ram, perc_cpu):
    
    # -------------------- Gerando a População -----------------------
    """Gerada de forma aleatória, onde cada linha consiste em um individuo da população
    e cada coluna desta linha corresponde a permutação inicial de posições das n rainhas"""
    popsize = 500
    population = np.zeros((popsize,n),dtype=int)
    for i in range(popsize):
        population[i,:] = np.random.permutation(n)

    # ------------------ Definindo Hiper Parametros--------------------
    survival_rate = 0.3 #definindo a taxa de sobreviventes na seleção
    mutation_prob = 0.1 #definindo a probabilidade de mutação de um descendente
    num_epochs = 1000 #definindo o numero de épocas

    #vetores para plot do fitness ao longo das epocas
    best_fitness = []
    avg_fitness = []

    start = time.time()  # início do contador de tempo
    start_cpu = process.cpu_times() # início de registro de CPU
    start_ram = process.memory_info().rss # início de registro de RAM
    
    #---------------Iniciando o ciclo evolutivo------------------------
    for epoch in range(num_epochs):
        
        # Calculando o fitness de cada individuo da população
        fitness = np.zeros(popsize,dtype=int)
        for i in range(popsize):
            fitness[i] = calc_fitness(population[i])
        
        # Guardando o historico do fitness     
        best_fit = np.min(fitness)
        avg_fit = np.mean(fitness)
        best_fitness.append(best_fit)
        avg_fitness.append(avg_fit)
        
        #parando caso uma solução foi encontrada
        if best_fit == 0:
            break
        
                
        # ------------------ Etapa de Seleção ------------------
            # Mecanismo: Torneio
                #Seleciona dois individuos aleatórios da população
                #Elimina o individuo de "pior" Fitness (neste caso, o que tiver mais colisões)
                
        while population.shape[0] > survival_rate * popsize:
            
            p1 = np.random.randint(0, population.shape[0])
            p2 = np.random.randint(0, population.shape[0])
            
            while p2 == p1:
                p2 = np.random.randint(0, population.shape[0])
            
            if fitness[p1] < fitness[p2]:
                population = np.delete(population, p2, axis=0)
                fitness = np.delete(fitness, p2)
            else:
                population = np.delete(population, p1, axis=0)
                fitness = np.delete(fitness, p1)
        
        # ------------------ Etapa de Reprodução ------------------
            #Mecanismo: Order Crossover
                #Seleciona um ponto de corte aleatorio
                #Descendente herda o segmento inicial do pai 1 até o ponto de corte
                #Os genes restantes são preenchidos na ordem em que aparecem no pai 2, ignorando os já herdados do pai 1
        while population.shape[0] < popsize:
            
            p1 = np.random.randint(0, population.shape[0])
            p2 = np.random.randint(0, population.shape[0])
            
            while p2 == p1:
                p2 = np.random.randint(0, population.shape[0])
            
            cut = np.random.randint(1, n-1)
            offspring = population[p1, :cut]
            
            for gene in population[p2]:
                if gene not in offspring:
                    offspring = np.append(offspring, gene)
            
            # ----------------- Etapa de Mutação-------------------
                #Mecanismo: Swap
                    #Sorteia dos indices aleatorios
                    #invertes os valores dos correspondentes indices sorteados
            if np.random.rand() < mutation_prob:
                i = np.random.randint(n)
                j = np.random.randint(n)
                offspring[i], offspring[j] = offspring[j], offspring[i]
            
            offspring = np.reshape(offspring, (1, n))
            population = np.append(population, offspring, axis=0)
            
        best_idx = np.argmin(fitness) # guardando a mposição do melhor fitness
        best_fit = fitness[best_idx] # obtendo o melhor fitness

    end = time.time()  # fim do contador de tempo
    end_cpu = process.cpu_times() # fim do registro de CPU
    end_ram = process.memory_info().rss # fim do registro de RAM

    # ------------------ RESULTADO ------------------

    #graficos:
    plt.figure()
    plt.plot(best_fitness, label="Melhor fitness")
    plt.plot(avg_fitness, label="Fitness médio")

    plt.xlabel("Épocas")
    plt.ylabel("Número de conflitos")
    plt.title("Evolução do fitness - Algoritmo Genético (N-Rainhas)")
    plt.legend()
    plt.grid()

    plt.show()

    #permutação
    best_idx = np.argmin(fitness) #buscando a elemento de melhor fitness
    fitness_hist.append(fitness[best_idx])
    print("Melhor solução GA:")
    print(population[best_idx])
    print("Conflitos:", fitness[best_idx])
    print(f"Solução encontrada na época {epoch}")
    print()

    #Tempo de execução
    elapsed = end - start
    #Tempo de uso de CPU
    cpu_time_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
    # Porcentagem de uso da CPU
    cpu_percent = process.cpu_percent(None)

    #print(f"Tempo total de execução: {elapsed:.4f} segundos")
    elapsed_times.append(elapsed)
    num_epochs_execute.append(epoch)
    #print(f"CPU total consumida pelo script: {cpu_time_used:.4f} segundos")
    cpu_consume.append(cpu_time_used)
    #print(f"RAM final do processo: {end_ram / 1024**2:.2f} MB")
    ram_consume.append((end_ram / 1024**2))
    #print(f"Pico aproximado de RAM observado: {max(start_ram, end_ram) / 1024**2:.2f} MB")
    peak_ram.append((max(start_ram, end_ram) / 1024**2))
    #print(f"CPU% do processo desde a última leitura: {cpu_percent:.2f}%")
    perc_cpu.append(cpu_percent)


# ------------------ Definindo o tamanho da instancia do problema ------------------

n = 10 # Tamanho do tabuleiro

fitness_hist = []
elapsed_times = []
num_epochs_execute = []
cpu_consume = []
ram_consume = []
peak_ram = []
perc_cpu = []

print("Iniciando Ciclo evolutivo")
print()
if n < 50:
    for i in range (10):
        print(f"Execução: {i}")
        execute_AG(fitness_hist, elapsed_times, num_epochs_execute, cpu_consume, ram_consume, peak_ram, perc_cpu)
elif (n < 70):
    for i in range (5):
        print(f"Execução: {i}")
        execute_AG(fitness_hist, elapsed_times, num_epochs_execute, cpu_consume, ram_consume, peak_ram, perc_cpu)
else:
    print("Execução:")
    execute_AG(fitness_hist, elapsed_times, num_epochs_execute, cpu_consume, ram_consume, peak_ram, perc_cpu)
        

num_sol_otima = fitness_hist.count(0)

print("------------------Resultados-----------------")
print(f"Tamanho do tabuleiro {n}x{n}:")
print(f"Numero de soluções otimas encontradas: {num_sol_otima}")
print(f"Numero medio de épocas executadas: {np.round(np.mean(num_epochs_execute))}")
print(f"Menor numero de épocas executadas: {np.min(num_epochs_execute)}")
print(f"Tempo medio de execução: {np.mean(elapsed_times):.5f} segundos")
print(f"menor tempo de execução: {np.min(elapsed_times):.5f} segundos")
print(f"CPU total medio consumida pelo script: {np.mean(cpu_consume):.5f} segundos")
print(f"Memorial RAM final médio do processo: {np.mean(ram_consume):.2f} MB")
print(f"Pico médio aproximado de RAM observado: {np.mean(peak_ram):.2f} MB")
print(f"CPU% médio do processo desde a última leitura: {np.mean(perc_cpu):.2f}%")

