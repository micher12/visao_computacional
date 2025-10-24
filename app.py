import cv2
import base64
import pygame
import threading
import time
import os
import io
import numpy as np

from ultralytics import YOLO
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import webview


load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


if not GEMINI_API_KEY:
    raise ValueError("A chave GEMINI_API_KEY não foi encontrada no arquivo .env")
genai.configure(api_key=GEMINI_API_KEY)


gemini_model = genai.GenerativeModel('gemini-2.5-flash')


yolo_model = YOLO('best12n_v4.pt')


# Inicia o Pygame, tocar som de alerta
pygame.mixer.init()
warning_sound = "alert.wav"

WARNING_CLASSES = {1, 2, 5}
FOCUS_CLASSS = {4, 0}

class Api:
    def __init__(self):
        self.latest_frame = None
        self.latest_boxes = []
        self.is_running = True

    def start_video_stream(self, window):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao abrir a fonte de vídeo.")
            self.is_running = False
            return

        som_tocando = False

        while self.is_running:
            success, frame = cap.read()
            if not success:
                print("Fim do fluxo de vídeo.")
                break
        
            # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # l, a, b = cv2.split(lab)
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Aumente clipLimit para mais contraste se necessário
            # l_clahe = clahe.apply(l)
            # lab_enhanced = cv2.merge((l_clahe, a, b))
            # frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Kernel menor para menos blur (era 5x5)
            # frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

            # # Sharpening leve para realçar bordas
            # kernel = np.array([[-1, -1, -1],
            #                 [-1, 9, -1],
            #                 [-1, -1, -1]])  # Kernel de sharpening básico
            # frame = cv2.filter2D(frame, -1, kernel * 0.5)  # Multiplique por 0.3-0.7 para intensidade

            self.latest_frame = frame.copy()
            
            frame_tem_warning = False   
            
            # Realiza a detecção YOLO
            results = yolo_model(frame, stream=True, verbose=False)
            
            current_boxes = []
            for r in results:
                boxes = r.boxes
                current_boxes.extend(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = yolo_model.names[cls]

                    if cls not in FOCUS_CLASSS and cls not in WARNING_CLASSES: # Ignora classes irrelevantes.
                        continue
                    
                    if(cls == 1 and conf < 0.5):
                        continue

                    if cls in WARNING_CLASSES:
                        frame_tem_warning = True
                        cor_caixa = (0, 0, 255) # Vermelho
                    else:
                        cor_caixa = (0, 255, 0) # Verde

                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_caixa, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_caixa, 2)
            
            # Armazena as últimas caixas detectadas
            self.latest_boxes = current_boxes

            # Lógica do som de alerta
            if frame_tem_warning and not som_tocando:
                pygame.mixer.music.load(warning_sound)
                pygame.mixer.music.play(-1)
                som_tocando = True
            elif not frame_tem_warning and som_tocando:
                pygame.mixer.music.stop()
                som_tocando = False

            # Converte o frame para Base64 e envia para o frontend
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Chama a função JS para atualizar o vídeo
            if window:
                window.evaluate_js(f'updateVideoFeed("{frame_b64}")')

            time.sleep(0.05) # Controla o framerate para não sobrecarregar

        cap.release()
        pygame.mixer.quit()

    def extract_data(self):
        if self.latest_frame is None or not self.latest_boxes:
            return "Nenhum objeto detectado no frame atual para análise."

        try:
            content_parts = []
            
            # Converte o frame completo para o formato PIL e adiciona à lista como contexto.
            full_image_pil = Image.fromarray(cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB))
            content_parts.append(full_image_pil)


            detected_classes = []
            for box in self.latest_boxes:

                cls = int(box.cls[0])
                class_name = yolo_model.names[cls]
                
                if class_name.lower() == 'person':
                    continue

                detected_classes.append(class_name)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi_cv = self.latest_frame[y1:y2, x1:x2]

                # Garante que o recorte não está vazio antes de converter.
                if roi_cv.size == 0:
                    continue

                # Converte o recorte de OpenCV (BGR) para PIL (RGB) e adiciona à lista.
                roi_pil = Image.fromarray(cv2.cvtColor(roi_cv, cv2.COLOR_BGR2RGB))
                content_parts.append(roi_pil)
            
            # Se após o filtro (ex: só havia 'Person'), não sobrar objetos, retorna.
            if not detected_classes:
                return "Nenhum EPI detectado para análise (apenas pessoas foram identificadas)."

            # 3. Construir o prompt dinâmico e otimizado.
            #    Utilizamos a versão melhorada que criamos anteriormente.
            prompt = f"""
            **Sua Missão:** Você é um sistema de IA especialista em Segurança do Trabalho, programado para realizar auditorias visuais de conformidade de Equipamentos de Proteção Individual (EPIs).

            **Contexto das Imagens:**
            - A primeira imagem que você recebeu é o frame de vídeo completo, para fornecer contexto.
            - As imagens seguintes são os recortes (ROIs) dos objetos que você deve analisar. A ordem dos recortes corresponde à ordem na lista de classes abaixo.

            **Objetos Detectados para Análise:**
            - {', '.join(detected_classes)}

            **Sua Tarefa:**
            Analise **exclusivamente** as imagens de recorte (ROIs). Para CADA EPI identificado nos recortes, catalogue os seguintes atributos:
            - **Tipo:** Seja específico (ex: Capacete de aba frontal, Óculos de ampla visão).
            - **Cor:** A cor principal do equipamento.
            - **Material Aparente:** O material que você deduz visualmente (ex: Plástico rígido, Policarbonato).

            **Formato da Resposta:**
            Use o nome da classe detectada como título. Exemplo:
            **Capacete:**
            - Tipo: Capacete de proteção com aba frontal
            - Cor: Branco
            - Material Aparente: Plástico rígido

            **Regras Críticas:**
            1. Se um atributo não for claramente visível no recorte, use "Não identificado".
            2. Se um dos recortes não for um EPI, pode mencioná-lo brevemente e seguir para o próximo.
            """
            
            # Colocar o prompt no inicio da lista
            content_parts.insert(0, prompt)

            response = gemini_model.generate_content(content_parts)
            
            return response.text.strip()

        except Exception as e:
            print(f"Erro ao contatar a API do Gemini ou processar as imagens: {e}")
            return "Ocorreu um erro durante a análise dos atributos."


if __name__ == '__main__':
    api = Api()
    window = webview.create_window('Detecção e Análise de Materiais', 'index.html', js_api=api, width=1200, height=800)

    webview.start(lambda: threading.Thread(target=api.start_video_stream, args=(window,)).start())
    
    api.is_running = False