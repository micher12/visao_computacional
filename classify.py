import cv2

from ultralytics import YOLO

imagem = cv2.imread("furadeira2.png")

model = YOLO('runs/detect/yolov12n_engenharia_civil4/weights/best.pt')

results = model(imagem, conf=0.4)

result = results[0]

# Itera sobre cada objeto detectado na imagem
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
    cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Coloca o texto do rótulo acima da caixa
    # cv2.putText(imagem, texto, posição, fonte, tamanho, cor, espessura)
    cv2.putText(imagem, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    

# Exibe a imagem com as detecções
cv2.imshow('Detector de Objetos em Obra Civil', imagem)

cv2.waitKey(0)