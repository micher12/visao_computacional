import cv2
from ultralytics import YOLO

# --- CONFIGURAÇÃO ---
# Carregar seu modelo YOLO treinado
# Certifique-se de que o caminho para o seu arquivo .pt está correto
try:
    model = YOLO('runs/detect/yolo11n_detection_v2/weights/best.pt')
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    print("Verifique se o caminho 'runs/detect/yolo12n_final/weights/best.pt' está correto.")
    exit()

# Inicializar a webcam
# O número 0 geralmente se refere à webcam padrão do seu computador.
# Se tiver múltiplas câmeras, pode ser 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

# --- LOOP PRINCIPAL ---
while True:
    # Capturar um único frame da webcam
    success, frame = cap.read()
    if not success:
        print("Falha ao capturar o frame. Fim do stream?")
        break

    # 1. EXECUTAR A DETECÇÃO DO YOLO NO FRAME ATUAL
    results = model(frame)

    # Processar os resultados da detecção
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Obter as coordenadas do bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Obter classe e confiança
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # --- DESENHAR INFORMAÇÕES DO YOLO NO FRAME ---
            # Desenhar o bounding box (retângulo azul)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Escrever o nome da classe e a confiança
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 2. ENCONTRAR E DESENHAR CONTORNOS DENTRO DO BOUNDING BOX
            # Recortar a Região de Interesse (ROI) onde o objeto foi detectado
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue

            # Aplicar pré-processamento apenas no ROI
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            
            # Binarização com método de Otsu para encontrar o threshold automaticamente
            # É mais robusto que um valor fixo como '60' para condições de luz variáveis
            _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Encontrar contornos no ROI processado
            contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Desenhar os contornos no frame original, ajustando a posição
            for contour in contours_roi:
                # O contorno encontrado está relativo ao ROI (ex: no ponto 10,15 do recorte).
                # Precisamos somar as coordenadas do início do ROI (x1, y1) para
                # posicioná-lo corretamente no frame completo.
                contour_offset = contour + (x1, y1)
                cv2.drawContours(frame, [contour_offset], -1, (0, 255, 0), 2) # Contorno verde

    # 3. EXIBIR O FRAME PROCESSADO
    cv2.imshow("Detecção em Tempo Real - Pressione 'q' para sair", frame)

    # Condição de parada: Pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- LIMPEZA ---
# Liberar a captura da webcam e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()