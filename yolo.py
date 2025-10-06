import cv2
import os
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import io
import time ### AJUSTE: Importa a biblioteca de tempo
from dotenv import load_dotenv

load_dotenv()


# --- CONFIGURAÇÕES ---

# 1. Configure sua chave de API do Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 2. Caminho para o seu modelo YOLO treinado
YOLO_MODEL_PATH = "runs/detect/yolo12n_final/weights/best.pt"
# YOLO_MODEL_PATH = "yolo.pt"

# 3. Limiar de confiança para detecção
CONFIDENCE_THRESHOLD = 0.5

# 4. Intervalo em segundos entre as chamadas da API para o mesmo objeto
API_CALL_INTERVAL = 5 # segundos

# Dicionário para armazenar atributos (cache) e evitar chamadas repetidas
object_attributes_cache = {}
# ### AJUSTE: Dicionário para controlar o tempo da última chamada da API por classe ###
last_api_call_time = {}

# --- FUNÇÃO PARA ANÁLISE COM GEMINI ---

def get_object_attributes_with_gemini(image_crop, class_name):
    """
    Envia um recorte da imagem de um objeto detectado para a API do Gemini
    e solicita a análise de seus atributos.
    O resultado é armazenado no cache.
    """
    print(f"-> Chamando API do Gemini para a classe: {class_name}...")

    # Converte o array numpy do OpenCV (BGR) para uma imagem que o Gemini entende (RGB)
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    gemini_image_input = {
        'mime_type': 'image/jpeg',
        'data': img_byte_arr
    }

    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"Analise a imagem deste objeto. Descreva seus principais atributos em uma linha, como: cor predominante, material provável (ex: metal, plástico, madeira), e se possível, a marca. Objeto: {class_name}."

    try:
        response = model.generate_content([prompt, gemini_image_input])
        attributes = response.text.strip().replace('\n', ' ')
        
        # ### AJUSTE: Armazena o resultado no cache ###
        object_attributes_cache[class_name] = attributes
        print(f"<- Resposta da API recebida e cache atualizado: {attributes}")
    except Exception as e:
        print(f"Erro ao chamar a API do Gemini: {e}")
        # Armazena uma mensagem de erro no cache para não tentar novamente a cada frame
        object_attributes_cache[class_name] = "Erro na analise."

# --- FUNÇÃO PRINCIPAL ---

def main():
    try:
        model = YOLO(YOLO_MODEL_PATH)

    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO em '{YOLO_MODEL_PATH}'. Verifique o caminho.")
        print(f"Erro: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- PRÉ-PROCESSAMENTO E DETECÇÃO (sem alterações) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced_gray = clahe.apply(gray)
        canny_edges = cv2.Canny(contrast_enhanced_gray, 100, 200)
        masked_image = cv2.bitwise_and(frame, frame, mask=canny_edges)
        
        results = model(masked_image, verbose=False)

        for result in results:
            for box in result.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    # --- ### AJUSTE: LÓGICA DE CONTROLE DE TEMPO ### ---
                    current_time = time.time()
                    
                    # Verifica se já passou o tempo definido desde a última chamada para ESTA classe
                    last_call = last_api_call_time.get(class_name, 0)
                    if current_time - last_call > API_CALL_INTERVAL:
                        # Atualiza o tempo da última chamada IMEDIATAMENTE
                        last_api_call_time[class_name] = current_time
                        
                        # Recorta o objeto do frame ORIGINAL para enviar ao Gemini
                        detected_object_crop = frame[y1:y2, x1:x2]
                        
                        # Chama a função que vai requisitar a API e atualizar o cache.
                        # A função em si não retorna nada, apenas atualiza o cache global.
                        # get_object_attributes_with_gemini(detected_object_crop, class_name)
                    
                    # --- Desenha as informações na tela ---
                    
                    # Pega o atributo mais recente do cache para exibir na tela.
                    # Se ainda não houver nada, mostra "Analisando...".
                    attributes_text = object_attributes_cache.get(class_name, "Analisando...")
                    
                    # Desenha o bounding box e os textos
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({box.conf[0]:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, attributes_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Mascara", masked_image)
        cv2.imshow("Deteccao em Tempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()