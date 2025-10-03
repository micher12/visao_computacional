from google import genai 
from google.genai import types 
from PIL import Image, ImageDraw, ImageFont
import json
import os
from dotenv import load_dotenv



# Carrega a variável de ambiente (agora GOOGLE_API_KEY) do arquivo .env
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def parse_json(json_output: str):
    if "```json" in json_output:
        json_output = json_output.split("```json")[1].split("```")[0]
    return json_output

def detectar_e_desenhar_objetos(image_path: str, output_path: str = "resultado_com_caixas.png"):
    """
    Detecta objetos na imagem, desenha uma caixa delimitadora ao redor de cada um
    e escreve o nome do objeto.
    """
    im = Image.open(image_path).convert("RGBA")
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

    prompt = """
    Forneça as máscaras de segmentação para os itens de madeira e vidro.
    Produza uma lista JSON de máscaras de segmentação onde cada entrada contém a caixa delimitadora 2D na chave "box_2d", a máscara de segmentação na chave "mask" e
    o rótulo de texto na chave "label". Use rótulos descritivos.
    """

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) 
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, im], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
        config=config
    ) 

    items = json.loads(parse_json(response.text))

    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.truetype("arial.ttf", size=25)
    except IOError:
        font = ImageFont.load_default()

    for item in items:
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * im.size[1])
        x0 = int(box[1] / 1000 * im.size[0])
        y1 = int(box[2] / 1000 * im.size[1])
        x1 = int(box[3] / 1000 * im.size[0])

        if y0 >= y1 or x0 >= x1:
            continue

        label = item['label']
        cor_caixa = "lime"

        draw.rectangle([x0, y0, x1, y1], outline=cor_caixa, width=4)

        posicao_texto = (x0 + 5, y0 - 30 if y0 > 30 else y0 + 5)
        
        text_bbox = draw.textbbox(posicao_texto, label, font=font)
        draw.rectangle(text_bbox, fill=cor_caixa)
        
        draw.text(posicao_texto, label, fill="black", font=font)
        
        print(f"Desenhando caixa e rótulo para: {label}")

    im = im.convert("RGB")
    im.save(output_path)
    print(f"✅ Imagem final salva em: {output_path}")
    im.show()

if __name__ == "__main__":
    detectar_e_desenhar_objetos("furadeira.png")