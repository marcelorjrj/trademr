import ccxt
import time
import threading
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from tkinter import Tk, Label, Button, Text, END, Frame, OptionMenu, StringVar, LabelFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import UndefinedMetricWarning

# Configurações
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
CORRETORA = 'binance'
TIMEFRAME = '5m'
LIMITE_DADOS = 1000
RISCO_REWARD_RATIO = 2.0
TAKE_PROFIT = 0.02
STOP_LOSS = 0.01

MOEDAS_DISPONIVEIS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'MELANIA/USDT']

# Variáveis globais
bot_rodando = False
modelo = None
tempo_restante = 300
operacoes = []
status_led = None

# Interface Gráfica
root = Tk()
root.title("Super Bot de Trading IA Pro")
root.configure(bg="#1a1a1a")
moeda_selecionada = StringVar(value=MOEDAS_DISPONIVEIS[0])

# Componentes da Interface
frame_grafico = Frame(root, bg="#1a1a1a")
fig, ax = plt.subplots(figsize=(14, 6), facecolor="#2d2d2d")
canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
label_sinal = Label(root, text="🔄 INICIE O BOT", font=("Roboto", 24, "bold"), bg="#1a1a1a", fg="#00ff88")
label_contagem = Label(root, text="⏳ 300s", font=("Roboto", 18), bg="#1a1a1a", fg="#ffffff")
texto_log = Text(root, height=14, width=100, bg="#2d2d2d", fg="#ffffff", insertbackground="white")

# Status LED
status_frame = LabelFrame(root, text="Status", bg="#1a1a1a", fg="white")
status_led = Label(status_frame, text="●", font=("Roboto", 24), bg="#1a1a1a", fg="red")


def estilo_botao(botao, cor_normal, cor_hover):
    botao.config(font=("Roboto", 12), bg=cor_normal, fg="white", relief="flat", padx=20, pady=10, bd=0)
    botao.bind("<Enter>", lambda e: botao.config(bg=cor_hover))
    botao.bind("<Leave>", lambda e: botao.config(bg=cor_normal))


# Funções Aprimoradas
def coletar_dados(moeda):
    try:
        exchange = getattr(ccxt, CORRETORA)({'enableRateLimit': True})
        dados = exchange.fetch_ohlcv(moeda, TIMEFRAME, limit=LIMITE_DADOS)
        df = pd.DataFrame(dados, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Erro: {str(e)}")


def calcular_indicadores(df):
    # Indicadores avançados
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()

    # Ichimoku Cloud
    df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Volume Profile
    df['volume_profile'] = df['volume'] * df['close']

    # Price Action
    df['hammer'] = ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & (df['close'] > df['open'])
    df['shooting_star'] = ((df['high'] - df['close']) > 2 * (df['close'] - df['open'])) & (df['close'] < df['open'])

    # Padronização robusta
    scaler = RobustScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df.dropna()


def treinar_modelo():
    global modelo
    try:
        df = calcular_indicadores(coletar_dados(moeda_selecionada.get()))
        df['Target'] = ((df['close'].shift(-5) > (df['close'] * (1 + TAKE_PROFIT))).astype(int))  # Alvo de 5 velas

        X, y = SMOTE().fit_resample(df.drop(['Target'], axis=1), df['Target'])

        tscv = TimeSeriesSplit(n_splits=5)
        modelo = GridSearchCV(
            Pipeline([
                ('scaler', RobustScaler()),
                ('model', LGBMClassifier(boosting_type='dart', class_weight='balanced'))
            ]),
            {
                'model__num_leaves': [31, 63],
                'model__learning_rate': [0.01, 0.05],
                'model__n_estimators': [200, 300]
            },
            cv=tscv,
            scoring=make_scorer(f1_score),
            n_jobs=-1
        ).fit(X, y)

        log(f"🔥 F1-Score: {modelo.best_score_:.2%}")
        log(f"Melhores parâmetros: {modelo.best_params_}")
    except Exception as e:
        log(f"Erro no treinamento: {str(e)}")


def gerar_sinal():
    if modelo is None: return ("AGUARDE", "gray", 0.0, 0.0)

    try:
        df = calcular_indicadores(coletar_dados(moeda_selecionada.get()))
        features = df.iloc[[-1]].drop(['Target'], axis=1, errors='ignore')

        proba = modelo.predict_proba(features)[0][1]
        volatilidade = df['close'].pct_change().std()

        # Limiares dinâmicos baseados na volatilidade
        limiar_compra = np.clip(0.82 - (volatilidade * 10), 0.65, 0.90)
        limiar_venda = np.clip(0.18 + (volatilidade * 10), 0.10, 0.35)

        if proba > limiar_compra:
            return ("🚀 COMPRAR", "#00ff88", proba, volatilidade)
        elif proba < limiar_venda:
            return ("🔻 VENDER", "#ff4444", proba, volatilidade)
        return ("⏸ NEUTRO", "#aaaaaa", proba, volatilidade)
    except Exception as e:
        log(f"Erro na geração de sinal: {str(e)}")
        return ("ERRO", "gray", 0.0, 0.0)


def executar_operacao(sinal):
    global operacoes
    entrada = time.time()
    resultado = "GANHO" if np.random.rand() < 0.8 else "PERDA"  # Simulação - implementar lógica real
    operacoes.append((entrada, sinal, resultado))
    log(f"Operação: {sinal[0]} | Resultado: {resultado}")


def atualizar_contagem():
    global tempo_restante
    while bot_rodando:
        if tempo_restante > 0:
            tempo_restante -= 1
            label_contagem.config(text=f"⏳ {tempo_restante}s")
            root.update()
            time.sleep(1)
        else:
            tempo_restante = 300
            sinal = gerar_sinal()
            label_sinal.config(text=f"{sinal[0]} ({sinal[2]:.2%}) | Vlt: {sinal[3]:.2%}", fg=sinal[1])
            executar_operacao(sinal)
            atualizar_grafico()


def atualizar_status():
    status_led.config(fg="#00ff00" if bot_rodando else "#ff0000")
    root.after(500, atualizar_status)


# Função para iniciar/parar o bot
def iniciar_parar_bot():
    global bot_rodando
    if bot_rodando:
        bot_rodando = False
        log("🛑 Bot parado")
        label_sinal.config(text="🔄 INICIE O BOT", fg="gray")
    else:
        bot_rodando = True
        log("🚀 Bot iniciado")
        threading.Thread(target=atualizar_contagem).start()


# Controles da Interface
frame_controles = Frame(root, bg="#1a1a1a")
menu_moedas = OptionMenu(frame_controles, moeda_selecionada, *MOEDAS_DISPONIVEIS)
botao_iniciar = Button(frame_controles, text="INICIAR BOT", command=lambda: [treinar_modelo(), iniciar_parar_bot()])
botao_parar = Button(frame_controles, text="PARAR BOT", command=iniciar_parar_bot)

# Aplicando estilos aos botões
estilo_botao(menu_moedas, "#363636", "#4a4a4a")
estilo_botao(botao_iniciar, "#00aa00", "#00cc00")
estilo_botao(botao_parar, "#ff4444", "#ff6666")

# Layout atualizado
status_frame.pack(pady=10)
status_led.pack()
frame_grafico.pack(pady=10)
canvas.get_tk_widget().pack()
label_sinal.pack(pady=15)
label_contagem.pack()
frame_controles.pack(pady=15)
menu_moedas.pack(side="left", padx=5)
botao_iniciar.pack(side="left", padx=5)
botao_parar.pack(side="left", padx=5)
texto_log.pack(pady=15)

atualizar_status()
root.mainloop()