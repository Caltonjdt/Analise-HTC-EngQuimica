import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# =============================================================================
# MODELAGEM DE RENDIMENTO DE HYDROCHAR (HTC)
# Engenharia Química - UFU
# =============================================================================

# 1. Dados Experimentais (Exemplo: Biomassa residual vs Temperatura)
# Temperatura em graus Celsius (°C) e Rendimento Mássico em porcentagem (%)
dados = {
    'temperatura': [180, 200, 220, 240, 260],
    'rendimento': [75.2, 68.5, 60.1, 52.4, 45.0]
}

# Criando o DataFrame
df = pd.DataFrame(dados)

# 2. Configurando o Modelo de Machine Learning (Regressão Linear Simples)
X = df[['temperatura']] # Variável independente
y = df['rendimento']    # Variável dependente

modelo = LinearRegression()
modelo.fit(X, y)

# 3. Realizando uma predição para uma temperatura intermediária
temp_alvo = 210
temp_teste = [[temp_alvo]]
predicao = modelo.predict(temp_teste)

print(f"--- Resultados da Modelagem ---")
print(f"Predição de rendimento estimado para {temp_alvo}°C: {predicao[0]:.2f}%\n")

# 4. Gerando o Gráfico de Dispersão e a Linha de Tendência
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='#1f77b4', s=100, label='Dados Experimentais (Bancada)')
plt.plot(X, modelo.predict(X), color='#d62728', linestyle='--', linewidth=2, label='Modelo de Predição')

# Formatando o gráfico para padrão de artigo científico
plt.title('Efeito da Temperatura no Rendimento de Hydrochar via HTC', fontsize=14)
plt.xlabel('Temperatura de Carbonização (°C)', fontsize=12)
plt.ylabel('Rendimento de Hydrochar (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Salvar o gráfico (opcional) ou apenas exibir
# plt.savefig('grafico_htc.png', dpi=300)
plt.show()
