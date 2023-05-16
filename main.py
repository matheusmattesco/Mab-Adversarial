from mab import MabAdversarial
import matplotlib.pyplot as plt
import math
import streamlit as st
import numpy as np
from PIL import Image
import time
import pandas as pd
import plotly.graph_objects as go

st.set_page_config("MAB-Adversarial", "🐙", layout="wide")
image = Image.open('img/mab.png')
logo = Image.open('img/logo-unifil.png')
col1, col2, col3 = st.columns([6, 18, 6])



texto_apresentacao = "  O código implementa o problema do Multi-Armed Bandit Adversarial em um ambiente simulado. O usuário pode definir o número de braços e o número de interações para o problema. O objetivo do agente é escolher o braço que maximiza a recompensa ao longo do tempo. Cada braço tem uma recompensa associada desconhecida. O algoritmo utilizado é o exp3, que leva em conta tanto a recompensa obtida quanto a exploração de braços menos conhecidos. A cada interação, o algoritmo atualiza as estimativas das recompensas e os pesos dos braços. O código exibe gráficos da evolução das recompensas médias, do número de vezes que cada braço foi usado e do arrependimento acumulado ao longo do tempo."
texto_explicativo = "O código implementa o problema do Multi-Armed Bandit Adversarial em um ambiente simulado. O usuário pode definir o número de braços e o número de interações para o problema. O objetivo do agente é escolher o braço que maximiza a recompensa ao longo do tempo. Cada braço tem uma recompensa associada desconhecida. O algoritmo utilizado é o exp3, que leva em conta tanto a recompensa obtida quanto a exploração de braços menos conhecidos. A cada interação, o algoritmo atualiza as estimativas das recompensas e os pesos dos braços. O código exibe gráficos da evolução das recompensas médias, do número de vezes que cada braço foi usado e do arrependimento acumulado ao longo do tempo."
with col2:
   st.image(image, width=None)
with col3:
   st.image(logo)
with col2:  
   st.markdown("<h1 style='text-align: center;'>Introdução</h1>", unsafe_allow_html=True)
   st.write(texto_explicativo)
   st.markdown("<h1 style='text-align: center;'>Defina as variáveis:</h1>", unsafe_allow_html=True)
   
  
co1, co2, co3, co4, co5 = st.columns([6, 8, 1, 8, 6])  
   
with co2:
    numArms = st.number_input("Número de Braços", value=10, min_value=1, max_value=10)
    button1 = st.button('Saber mais 🔎', key='button1') 
    if button1:
        st.write("Representa o número de alavancas disponíveis para o agente puxar. Em outras palavras, é o número de opções que o agente tem para escolher a cada vez que ele precisa tomar uma decisão.")
    maxReward = st.number_input("Máximo de Recompensa", value=1, min_value=1, max_value=10)
    button2 = st.button('Saber mais 🔎', key='button2') 
    if button2:
        st.write("Representa o valor máximo de recompensa que o agente pode receber ao puxar uma determinada alavanca. Em outras palavras, é o valor máximo que o agente pode receber por escolher a melhor opção.")
    T = st.number_input("Número de Interações", value=100, min_value=1, max_value=10000)
    button3 = st.button('Saber mais 🔎', key='button3')
    if button3:
        st.write("Representa o número de vezes que o agente pode puxar as alavancas. Quanto maior o número de interações, maior a oportunidade para o agente aprender e aprimorar sua estratégia de escolha.")
with co4:    
    gamma = st.slider("Taxa de Exploração/Explotação", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    button4 = st.button('Saber mais 🔎', key='button4')
    if button4:
        st.write("Representa a proporção entre a escolha de uma alavanca com base no valor que ela oferece (exploração) e a escolha de uma alavanca com base em sua frequência de escolha até o momento (explotação). Uma taxa de exploração alta significa que o agente está disposto a experimentar opções diferentes, enquanto uma taxa de exploração baixa significa que ele está se concentrando nas opções que lhe deram as melhores recompensas até o momento.")    
    adversarial_prob=A = st.slider("Chance de Haver Adversários", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    button5 = st.button('Saber mais 🔎', key='button5')
    if button5:
        st.write("Representa a possibilidade de que o próprio ambiente do MAB possa ser enganoso ou instável, fornecendo feedbacks incorretos ou variáveis")


co1, co2, co3, co4, co5 = st.columns([6, 8, 3, 8, 6]) 
coln1, coln2, coln3 = st.columns([1, 3, 1,])

with co3:
    if st.button('Executar'):
        with coln2:
            with st.spinner('Realizando os calculos...'):
                time.sleep(2)
                mab = MabAdversarial(numArms=numArms, maxReward=maxReward, gamma=gamma, T=T, adversarial_prob=adversarial_prob)
                mean_rewards, arm_counts, allRewards, regret, fakeRewards, results = mab.run()
                mean_rewards2, arm_counts2, allRewards2, regret2, results2 = mab.run_no_adversarial()
            st.success('Concluído!')
            
            rewards1, rewards2 = zip(*fakeRewards)
            data = {'Braço': [f'Braço {i+1}' for i in range(len(arm_counts))],
                    'Seleções': arm_counts,
                    'Recompensas Médias': mean_rewards,
                    'Recompensas Min Estimadas': rewards1,
                    'Recompensas Max Esperada': rewards2}
            df = pd.DataFrame(data)
            st.write(df)

            data = {'Braço': [f'Braço {i+1}' for i in range(len(arm_counts2))],
                    'Seleções': arm_counts2,
                    'Recompensas Médias': mean_rewards2,
                    'Recompensas Min Estimadas': rewards1,
                    'Recompensas Max Esperada': rewards2}
            df2 = pd.DataFrame(data)
            st.write(df2)


            total_selecoes = sum(arm_counts)
            porcentagens = [count / total_selecoes * 100 for count in arm_counts]
            labels = [f'Braço {i+1}' for i in range(len(porcentagens))]
            fig = go.Figure(data=[go.Pie(labels=labels, values=porcentagens, hole=0.3)])
            fig.update_layout(title='Porcentagem de seleções por braço')
            st.plotly_chart(fig)

            total_selecoes = sum(arm_counts2)
            porcentagens = [count / total_selecoes * 100 for count in arm_counts2]
            labels = [f'Braço {i+1}' for i in range(len(porcentagens))]
            fig = go.Figure(data=[go.Pie(labels=labels, values=porcentagens, hole=0.3)])
            fig.update_layout(title='Porcentagem de seleções por braço sem Adversarial')
            st.plotly_chart(fig)

            df = pd.DataFrame(results, columns=['Interação', 'Braço escolhido', 'Recompensa'])
            recompensas_braços = {}
            for braço in range(numArms):
                recompensas = df.loc[df['Braço escolhido'] == braço, 'Recompensa']
                recompensas_braços[braço] = recompensas

            fig = go.Figure()
            for braço, recompensas in recompensas_braços.items():
                fig.add_trace(go.Scatter(x=recompensas.index, y=recompensas, mode='lines', name=f'Braço {braço+1}'))
            fig.update_layout(title='Grafico de Recompensas', xaxis_title='Interação', yaxis_title='Recompensa')
            st.plotly_chart(fig)

            df = pd.DataFrame(results2, columns=['Interação', 'Braço escolhido', 'Recompensa'])
            recompensas_braços = {}
            for braço in range(numArms):
                recompensas = df.loc[df['Braço escolhido'] == braço, 'Recompensa']
                recompensas_braços[braço] = recompensas

            fig = go.Figure()
            for braço, recompensas in recompensas_braços.items():
                fig.add_trace(go.Scatter(x=recompensas.index, y=recompensas, mode='lines', name=f'Braço {braço+1}'))
            fig.update_layout(title='Grafico de Recompensas sem o Adversarial', xaxis_title='Interação', yaxis_title='Recompensa')
            st.plotly_chart(fig)





