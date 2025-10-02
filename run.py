import cv2
import numpy as np
from ultralytics import YOLO 

screenRead = cv2.VideoCapture(0)

while(True):
    _, img = screenRead.read()

    largura = 500

    altura = int(img.shape[0] * largura / img.shape[1])

    imgColorida = cv2.resize(img, (largura, altura))

    model = YOLO('runs/detect/yolov12n_engenharia_civil4/weights/best.pt')

    results = model(imgColorida, conf=0.4)

    result = results[0]

    for box in result.boxes:
        # Obtém as coordenadas da caixa (x1, y1, x2, y2)
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        
        # Obtém a classe do objeto e a confiança
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Obtém o nome da classe em inglês
        class_name_en = model.names[class_id]
        

        # Prepara o texto do rótulo
        label = f"{class_name_en} {confidence:.2f}"
        
        # Desenha o retângulo (bounding box) ao redor do objeto
        # (x1, y1) é o canto superior esquerdo e (x2, y2) é o inferior direito
        # (0, 255, 0) é a cor (verde) e 2 é a espessura da linha
        cv2.rectangle(imgColorida, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Coloca o texto do rótulo acima da caixa
        # cv2.putText(imagem, texto, posição, fonte, tamanho, cor, espessura)
        cv2.putText(imgColorida, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Resultado Final com Contornos", imgColorida)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cv2.destroyAllWindows()
screenRead.release()