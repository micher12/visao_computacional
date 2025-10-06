import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import google.generativeai as genai  # Importação padronizada
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


# --- CONFIGURAÇÃO INICIAL ---

# 1. Carregar o modelo YOLO
YOLO_MODEL_PATH = 'runs/detect/yolo12n_finalv3/weights/best.pt'
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    messagebox.showerror("Erro", f"Erro ao carregar o modelo YOLO: {e}\n\nVerifique o caminho: '{YOLO_MODEL_PATH}'")
    exit()

# 2. Configurar a API do Google Gemini
try:
    # Sua variável de ambiente está correta: GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente GEMINI_API_KEY não foi encontrada.")
    
    # Maneira correta de configurar a API
    genai.configure(api_key=api_key)
    
    # Instanciar o modelo desejado (Gemini 1.5 Flash)
    # Usar '-latest' é uma boa prática para sempre pegar a versão mais recente.
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

except Exception as e:
    messagebox.showerror("Erro de API", f"Erro ao configurar a API do Gemini: {e}\n\nCertifique-se de que a variável de ambiente GEMINI_API_KEY está configurada corretamente em seu arquivo .env")
    exit()

# Variável global para armazenar os objetos detectados
detected_objects = []

# --- FUNÇÕES LÓGICAS ---

def process_and_detect_image(image_path):
    """
    Carrega uma imagem, executa a detecção de objetos YOLO e desenha os resultados.
    """
    global detected_objects
    detected_objects = [] # Limpa detecções anteriores

    frame = cv2.imread(image_path)
    if frame is None:
        messagebox.showerror("Erro", f"Não foi possível ler a imagem em: {image_path}")
        return None, None

    # Executar a detecção do YOLO
    results = model(frame)

    # Processar os resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Desenhar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Recortar a Região de Interesse (ROI)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # Armazenar o ROI e o nome da classe para uso posterior
                detected_objects.append({'roi': roi, 'name': class_name})

                # Processamento de contorno (opcional, como no seu código original)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_roi:
                    contour_offset = contour + (x1, y1)
                    cv2.drawContours(frame, [contour_offset], -1, (0, 255, 0), 2)

    return frame, detected_objects

def fetch_attributes_from_gemini():
    """
    Pega o primeiro ROI detectado, envia para o Gemini e exibe os atributos.
    """
    if not detected_objects:
        messagebox.showwarning("Aviso", "Nenhum objeto foi detectado para buscar atributos.")
        return

    # Pega o primeiro objeto detectado para análise
    first_object = detected_objects[0]
    roi_image = first_object['roi']
    object_name = first_object['name']

    status_label.config(text=f"Buscando atributos para '{object_name}'...")
    root.update_idletasks() # Força a atualização da interface

    # Converter a imagem do OpenCV (BGR) para RGB e depois para PIL
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("nome",roi_rgb)
    pil_image = Image.fromarray(roi_rgb)

    prompt = [
        "Identifique os principais materiais ou atributos visuais deste objeto.",
        "Responda de forma concisa, com os atributos separados por vírgula.",
        "Exemplo: Madeira, Ferro",
        pil_image
    ]

    try:
        # --- CHAMADA CORRIGIDA PARA A API ---
        # O método generate_content é chamado diretamente no objeto do modelo
        response = gemini_model.generate_content(prompt)
        attributes = response.text
        
        # Atualiza a label de resultado
        result_label.config(text=f"Objeto: {object_name}\nAtributos: {attributes}")
        status_label.config(text="Atributos recebidos com sucesso!")
        
    except Exception as e:
        messagebox.showerror("Erro na API", f"Falha ao se comunicar com a API do Gemini: {e}")
        status_label.config(text="Erro ao buscar atributos.")


# --- FUNÇÕES DA INTERFACE (GUI) ---
# (Nenhuma alteração necessária aqui)
def select_image():
    """
    Abre uma caixa de diálogo para o usuário selecionar uma imagem e inicia o processamento.
    """
    path = filedialog.askopenfilename(
        filetypes=[("Arquivos de Imagem", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if not path:
        return

    # Limpa resultados anteriores
    result_label.config(text="")
    status_label.config(text="Processando imagem...")
    btn_get_attributes.config(state=tk.DISABLED)

    # Processa a imagem
    processed_frame, objects = process_and_detect_image(path)
    
    if processed_frame is not None:
        # Converte a imagem OpenCV (BGR) para um formato que o Tkinter possa exibir (RGB)
        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Redimensiona a imagem para caber na tela, mantendo a proporção
        img_pil.thumbnail((800, 600))
        
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Atualiza o painel de imagem
        image_label.config(image=img_tk)
        image_label.image = img_tk # Mantém uma referência para evitar que a imagem seja deletada

        if objects:
            status_label.config(text=f"Detectado: {objects[0]['name']}. Pronto para buscar atributos.")
            btn_get_attributes.config(state=tk.NORMAL) # Ativa o botão
        else:
            status_label.config(text="Nenhum objeto do modelo foi detectado na imagem.")

# --- CRIAÇÃO DA INTERFACE GRÁFICA (GUI) ---
# (Nenhuma alteração necessária aqui)
root = tk.Tk()
root.title("Detector de Objetos e Atributos com YOLO e Gemini")
root.geometry("800x700")

# Frame para os botões e textos
top_frame = Frame(root, pady=10)
top_frame.pack(side=tk.TOP, fill=tk.X)

btn_load = Button(top_frame, text="Carregar Imagem", command=select_image)
btn_load.pack(side=tk.LEFT, padx=10)

btn_get_attributes = Button(top_frame, text="Buscar Atributos", command=fetch_attributes_from_gemini, state=tk.DISABLED)
btn_get_attributes.pack(side=tk.LEFT, padx=10)

result_label = Label(top_frame, text="Carregue uma imagem para começar.", wraplength=500, justify=tk.LEFT)
result_label.pack(side=tk.LEFT, padx=20)

# Label para exibir a imagem
image_label = Label(root)
image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Barra de status
status_label = Label(root, text="Pronto.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# Iniciar o loop principal da interface
root.mainloop()