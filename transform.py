import os
import cv2
import shutil
import yaml
from tqdm import tqdm

# --- CONFIGURAÇÕES ---

# 1. Caminho para a pasta principal do seu dataset original
BASE_DATASET_PATH = 'datasets'

# 2. Nome da pasta onde o novo dataset processado será salvo
OUTPUT_DATASET_PATH = 'datasets_processed2'

# Parâmetros do pré-processamento (devem ser os mesmos do seu script em tempo real)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---

def process_and_save_image(input_path, output_path):
    """
    Carrega uma imagem, aplica o pipeline de pré-processamento e a salva no destino.
    """
    # Carrega a imagem original
    img = cv2.imread(input_path)
    if img is None:
        print(f"Aviso: Não foi possível carregar a imagem {input_path}")
        return

    # 1. Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplica CLAHE para melhorar o contraste
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    contrast_enhanced_gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(contrast_enhanced_gray, (5, 5), 0)
    
    # 3. Aplica o detector de bordas Canny
    canny_edges = cv2.Canny(blur, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    
    # 4. Cria a imagem mascarada (masked_image)
    masked_image = cv2.bitwise_and(img, img, mask=canny_edges)
    
    # 5. Salva a imagem processada
    cv2.imwrite(output_path, masked_image)

# --- SCRIPT PRINCIPAL ---

def main():
    print(f"Iniciando o pré-processamento do dataset em '{BASE_DATASET_PATH}'...")
    print(f"O novo dataset será salvo em '{OUTPUT_DATASET_PATH}'.")

    # Cria a pasta principal de saída, se não existir
    os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

    # Subpastas a serem processadas (train, valid, test)
    splits = [d for d in os.listdir(BASE_DATASET_PATH) if os.path.isdir(os.path.join(BASE_DATASET_PATH, d))]
    
    for split in splits:
        print(f"\nProcessando a partição: {split}")
        
        input_images_path = os.path.join(BASE_DATASET_PATH, split, 'images')
        input_labels_path = os.path.join(BASE_DATASET_PATH, split, 'labels')
        
        output_images_path = os.path.join(OUTPUT_DATASET_PATH, split, 'images')
        output_labels_path = os.path.join(OUTPUT_DATASET_PATH, split, 'labels')
        
        # Cria as pastas de saída para imagens e labels
        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_labels_path, exist_ok=True)
        
        # --- 1. Processa e salva as IMAGENS ---
        if os.path.exists(input_images_path):
            image_files = [f for f in os.listdir(input_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in tqdm(image_files, desc=f"  Processando imagens de '{split}'"):
                input_img_path = os.path.join(input_images_path, filename)
                output_img_path = os.path.join(output_images_path, filename)
                process_and_save_image(input_img_path, output_img_path)
        
        # --- 2. Copia os LABELS (não precisam de processamento) ---
        if os.path.exists(input_labels_path):
            print(f"  Copiando labels de '{split}'...")
            # Usamos shutil.copytree para copiar todo o conteúdo da pasta
            # Para isso, removemos a pasta de destino se ela já existir
            if os.path.exists(output_labels_path):
                shutil.rmtree(output_labels_path)
            shutil.copytree(input_labels_path, output_labels_path)

    # --- 3. Atualiza o arquivo data.yaml ---
    original_yaml_path = os.path.join(BASE_DATASET_PATH, 'data.yaml')
    if os.path.exists(original_yaml_path):
        print("\nAtualizando o arquivo 'data.yaml'...")
        with open(original_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Atualiza os caminhos para apontar para a nova pasta do dataset processado
        # O YOLOv8 espera caminhos relativos à pasta do .yaml, então ajustamos
        data_config['path'] = f'../{OUTPUT_DATASET_PATH}' # Caminho para a raiz do dataset
        data_config['train'] = 'train/images'
        data_config['val'] = 'valid/images' # 'val' é o padrão, mas pode ser 'valid'
        if 'test' in data_config:
             data_config['test'] = 'test/images'

        new_yaml_path = os.path.join(OUTPUT_DATASET_PATH, 'data.yaml')
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        print(f"Novo 'data.yaml' salvo em: {new_yaml_path}")
        
    print("\nProcessamento concluído com sucesso!")

if __name__ == '__main__':
    main()