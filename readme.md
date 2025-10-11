# Detecção e Análise de EPIs com IA

Este projeto é uma aplicação de desktop que utiliza inteligência artificial para detectar Equipamentos de Proteção Individual (EPIs) em tempo real através de uma webcam. A aplicação identifica os EPIs, os classifica e utiliza a API do Gemini para fornecer uma análise detalhada dos materiais detectados.

## Funcionalidades

- **Detecção em Tempo Real:** Utiliza um modelo YOLOv8 para detectar EPIs e pessoas em um feed de vídeo ao vivo.
- **Classificação de Alertas:** Objetos que representam um risco (como "luva-off" ou "capacete-off") são destacados com uma caixa de cor diferente e acionam um alerta sonoro.
- **Análise com IA Generativa:** Ao clicar em um botão, a aplicação captura o frame atual, recorta os EPIs detectados e os envia para a API do Google Gemini para uma análise detalhada, que inclui tipo, cor e material aparente.
- **Interface Gráfica Simples:** Uma interface criada com `pywebview` que exibe o feed da câmera e os resultados da análise de forma clara.

## Tecnologias Utilizadas

- **Backend:** Python
- **Frontend:** HTML, Tailwind CSS, JavaScript
- **Framework da Aplicação:** `pywebview`
- **Detecção de Objetos:** `ultralytics` (YOLOv8)
- **Análise de Imagens:** Google Gemini
- **Processamento de Imagem:** `opencv-python`
- **Manipulação de Áudio:** `pygame`

## Configuração e Instalação

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### Pré-requisitos

- Python 3.8 ou superior
- Acesso a uma webcam

### Passos

1. **Clone o Repositório**

   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio
   ```

2. **Crie e Ative um Ambiente Virtual**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Instale as Dependências**

   Instale todas as bibliotecas necessárias a partir do arquivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a Chave da API**

   Este projeto requer uma chave da API do Google Gemini.

   - Crie um arquivo chamado `.env` na raiz do projeto.
   - Adicione sua chave de API ao arquivo da seguinte forma:

   ```
   GEMINI_API_KEY="SUA_CHAVE_API_AQUI"
   ```

## Como Executar

Após a instalação e configuração, inicie a aplicação executando o script `app.py`:

```bash
python app.py
```

A janela da aplicação será aberta, exibindo o feed da sua webcam.

## Como Funciona

1. **Captura de Vídeo:** A aplicação utiliza a `OpenCV` para capturar o vídeo da webcam padrão.
2. **Detecção com YOLO:** Cada frame do vídeo é processado por um modelo YOLOv8 treinado para detectar diferentes tipos de EPIs (capacetes, óculos, luvas, etc.) e pessoas.
3. **Interface e Alertas:**
   - O vídeo com as detecções é exibido na interface do `pywebview`.
   - Se um objeto de "risco" (como um EPI sendo usado incorretamente) é detectado, um som de alerta é reproduzido continuamente usando `pygame`.
4. **Análise com Gemini:**
   - Ao clicar no botão "Extrair Dados dos Materiais", o frame atual e os recortes dos EPIs detectados são enviados para a API do Gemini.
   - Um prompt estruturado instrui o modelo a analisar cada recorte e extrair atributos como tipo, cor e material.
5. **Exibição dos Resultados:** A resposta do Gemini é formatada e exibida na área "Atributos Extraídos" da interface.
