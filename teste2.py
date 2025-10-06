import cv2
import numpy as np
import matplotlib.pyplot as plt

def encontrar_e_desenhar_contornos(image_path):
    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar um blur para suavizar e remover ruídos
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Binarização (thresholding) para separar o objeto do fundo
    # A escolha do threshold pode precisar de ajuste dependendo das suas imagens
    # cv2.THRESH_BINARY_INV inverte a imagem se os objetos forem escuros e o fundo claro
    _, binary = cv2.threshold(blurred, 60, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    # Detecção de bordas Canny (opcional, mas bom para contornos nítidos)
    edges = cv2.Canny(binary, 50, 150)
    cv2.imshow("Teste2",binary)
    cv2.imshow("Teste",edges)

    # Encontrar contornos na imagem binária/de bordas
    # cv2.RETR_EXTERNAL: Recupera apenas os contornos externos.
    # cv2.CHAIN_APPROX_SIMPLE: Comprime segmentos horizontais, verticais e diagonais.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos na imagem original (para fins de visualização)
    image_with_contours = img.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2) # Desenha em verde

    # Exibir a imagem com os contornos (opcional)
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title(f"{len(contours)} objetos encontrados")
    plt.axis('off')
    plt.show()

    return contours, img

# Exemplo de uso:
# contours_detectados, imagem_original = encontrar_e_desenhar_contornos("furadeira2.png")


from ultralytics import YOLO
import cv2
import numpy as np

# Carregar seu modelo YOLO treinado
# Substitua 'caminho/para/seu/modelo.pt' pelo caminho do seu modelo YOLOv8
model = YOLO('runs/detect/yolo12n_finalv2/weights/best.pt')

def detectar_e_desenhar_yolo_e_contornos(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return [] # Retorna lista vazia se não carregar

    # Executar a detecção do YOLO
    results = model(img) # results é uma lista de objetos Results (um por imagem)

    detected_objects_info = []
    
    # Processar os resultados
    for r in results:
        boxes = r.boxes # Bounding boxes no formato Ultralytics
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Coordenadas do bounding box
            conf = float(box.conf[0]) # Confiança da detecção
            cls = int(box.cls[0]) # ID da classe
            class_name = model.names[cls] # Nome da classe (ex: 'chave_fenda', 'martelo')

            # Desenhar o bounding box (retângulo)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Retângulo azul

            # Adicionar texto com classe e confiança
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            detected_objects_info.append({
                "class": class_name,
                "confidence": conf,
                "bounding_box": (x1, y1, x2, y2),
                "cropped_image_b64": None # Será preenchido para o Gemini
            })
            
    # Salvar ou exibir a imagem resultante
    cv2.imwrite("resultado_yolo_contornos.png", img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detecção YOLO + Contornos")
    plt.axis('off')
    plt.show()

    return detected_objects_info

# Exemplo de uso:
detectados = detectar_e_desenhar_yolo_e_contornos("furadeira.png")
