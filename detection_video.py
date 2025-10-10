import cv2
from ultralytics import YOLO
import pygame

# --- CONFIGURAÇÃO INICIAL ---
try:
    # Carrega o modelo YOLO pré-treinado
    model = YOLO('runs/detect/yolo11n_detection_v2/weights/best.pt')
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    exit()


pygame.mixer.init()
warning_sound = "alert.wav"


WARNING_CLASSES = {5, 7, 8, 9, 10}


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a fonte de vídeo.")
    exit()


som_tocando = False


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Fim do fluxo de vídeo ou erro na leitura.")
        break


    frame_tem_warning = False
    
    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if(cls == 6):
                continue

            if cls in WARNING_CLASSES:
                frame_tem_warning = True
                cor_caixa = (0, 0, 255)
            else:
                cor_caixa = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_caixa, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_caixa, 2)
            

    if frame_tem_warning and not som_tocando:
        print("ALERTA DETECTADO — Tocando som...")
        pygame.mixer.music.load(warning_sound)
        pygame.mixer.music.play(-1) 
        som_tocando = True
    elif not frame_tem_warning and som_tocando:
        print("Nenhum alerta — Parando som.")
        pygame.mixer.music.stop()
        som_tocando = False


    cv2.imshow("Detecção em Vídeo", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Encerrando...")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()