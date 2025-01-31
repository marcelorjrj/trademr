import ccxt
import time
import threading
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from tkinter import Tk, Label, Entry, Button, Text, END, messagebox, Frame, OptionMenu, StringVar, LabelFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignorar avisos de métricas indefinidas
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Configurações
CORRETORA = 'binance'  # Corretora para coletar dados
TIMEFRAME = '1m'  # Timeframe (1 minuto)
LIMITE_DADOS = 100  # Quantidade de dados históricos (ajustado para melhor visualização)
STOP_LOSS = 0.02  # 2% de stop-loss
TAKE_PROFIT = 0.03  # 3% de take-profit

# Moedas disponíveis
MOEDAS_DISPONIVEIS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'MELANIA/USDT']  # Adicione mais moedas conforme necessário

# Variáveis globais
bot_rodando = False
auto_refresh_rodando = False
modelo = None
historico_sinais = []


# Função para coletar dados históricos
def coletar_dados(moeda):
    try:
        corretora = getattr(ccxt, CORRETORA)()
        ohlcv = corretora.fetch_ohlcv(moeda, TIMEFRAME, limit=LIMITE_DADOS)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Erro ao coletar dados: {str(e)}")


# Função para calcular indicadores técnicos
def calcular_indicadores(df):
    # Médias móveis
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    # RSI (Índice de Força Relativa)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Convergência/Divergência de Médias Móveis)
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA20'] + 2 * df['close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA20'] - 2 * df['close'].rolling(window=20).std()

    df.dropna(inplace=True)
    return df


# Função para treinar o modelo de IA
def treinar_modelo():
    global modelo
    moeda = moeda_selecionada.get()
    df = coletar_dados(moeda)
    df = calcular_indicadores(df)
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    print("Distribuição das classes:")
    print(df['Target'].value_counts())  # Verificar a distribuição das classes
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    modelo = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced'))
    ])
    modelo.fit(X_train, y_train)
    # Validação do modelo
    y_pred = modelo.predict(X_test)
    log(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
    log(classification_report(y_test, y_pred))


# Função para gerar o sinal de compra/venda
def gerar_sinal():
    global modelo
    if modelo is None:
        return "Aguardando...", "gray", 0.0
    moeda = moeda_selecionada.get()
    df = coletar_dados(moeda)
    df = calcular_indicadores(df)
    X = df.iloc[[-1]]  # Usa apenas os dados mais recentes
    proba = modelo.predict_proba(X)[0][1]
    if proba > 0.65:  # Limiar ajustado para 0.65
        return "COMPRAR", "green", proba
    elif proba < 0.35:  # Limiar ajustado para 0.35
        return "VENDER", "red", proba
    else:
        return "NEUTRO", "gray", proba


# Função para atualizar o sinal de compra/venda
def atualizar_sinal():
    while bot_rodando:
        try:
            texto, cor, proba = gerar_sinal()
            label_sinal.config(text=f"{texto} ({proba:.2%})", fg=cor)
            historico_sinais.append((texto, proba))
            atualizar_grafico()
            root.update()
            time.sleep(60)  # Atualiza a cada 60 segundos
        except Exception as e:
            log(f"Erro na atualização: {str(e)}")


# Função para atualizar o gráfico de candlesticks
def atualizar_grafico():
    moeda = moeda_selecionada.get()
    df = coletar_dados(moeda)
    ax.clear()
    mpf.plot(df, type='candle', style='charles', ax=ax, volume=False)
    canvas.draw()


# Função para iniciar/parar o bot
def iniciar_parar_bot():
    global bot_rodando
    if not bot_rodando:
        treinar_modelo()
        bot_rodando = True
        threading.Thread(target=atualizar_sinal).start()
        botao_iniciar.config(text="Parar Bot", bg="red")
    else:
        bot_rodando = False
        botao_iniciar.config(text="Iniciar Bot", bg="green")


# Função para atualizar o sinal manualmente
def atualizar_sinal_manual():
    texto, cor, proba = gerar_sinal()
    label_sinal.config(text=f"{texto} ({proba:.2%})", fg=cor)
    historico_sinais.append((texto, proba))
    atualizar_grafico()
    root.update()


# Função para ativar/desativar o auto refresh
def alternar_auto_refresh():
    global auto_refresh_rodando
    if not auto_refresh_rodando:
        auto_refresh_rodando = True
        botao_auto_refresh.config(text="Desativar Auto Refresh", bg="red")
        threading.Thread(target=auto_refresh).start()
    else:
        auto_refresh_rodando = False
        botao_auto_refresh.config(text="Ativar Auto Refresh", bg="green")


# Função para auto refresh
def auto_refresh():
    while auto_refresh_rodando:
        try:
            atualizar_sinal_manual()
            time.sleep(30)  # Atualiza a cada 30 segundos
        except Exception as e:
            log(f"Erro no auto refresh: {str(e)}")


# Função para exibir logs
def log(mensagem):
    texto_log.insert(END, f"{mensagem}\n")
    texto_log.see(END)


# Criando interface gráfica
root = Tk()
root.title("Bot de Trading com IA")
root.configure(bg="#2c3e50")

# Variável de controle para seleção de moeda (criada após a janela principal)
moeda_selecionada = StringVar(value=MOEDAS_DISPONIVEIS[0])  # Moeda padrão

# Frame para o gráfico
frame_grafico = Frame(root, bg="#2c3e50")
frame_grafico.pack(pady=20)

fig, ax = plt.subplots(figsize=(10, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
canvas.get_tk_widget().pack()

# Label para o sinal
label_sinal = Label(root, text="Aguardando...", font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="gray")
label_sinal.pack(pady=20)

# Botão para selecionar moeda
label_moeda = Label(root, text="Selecione a Moeda:", font=("Helvetica", 12), bg="#2c3e50", fg="white")
label_moeda.pack(pady=5)

menu_moedas = OptionMenu(root, moeda_selecionada, *MOEDAS_DISPONIVEIS)
menu_moedas.config(font=("Helvetica", 12), bg="#34495e", fg="white")
menu_moedas.pack(pady=10)

# Botões
botao_iniciar = Button(root, text="Iniciar Bot", command=iniciar_parar_bot, font=("Helvetica", 14), bg="green",
                       fg="white")
botao_iniciar.pack(pady=10)

botao_refresh = Button(root, text="Refresh", command=atualizar_sinal_manual, font=("Helvetica", 14), bg="blue",
                       fg="white")
botao_refresh.pack(pady=10)

botao_auto_refresh = Button(root, text="Ativar Auto Refresh", command=alternar_auto_refresh, font=("Helvetica", 14),
                            bg="green", fg="white")
botao_auto_refresh.pack(pady=10)

# Área de logs
texto_log = Text(root, height=10, width=80, bg="#34495e", fg="white")
texto_log.pack(pady=20)

root.mainloop()