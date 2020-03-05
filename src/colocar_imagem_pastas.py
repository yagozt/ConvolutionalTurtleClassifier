import glob
import os
import shutil
print(glob.glob("*surprised*.jpg"))

caminho_atual = os.getcwd()

caminhos = ["angry", "contemptuous", "disgusted",
            "fearful", "happy", "neutral", "sad", "surprised"]

for pasta in caminhos:
    lista_imagens = glob.glob("*"+pasta+"*.jpg")

    for image_name in lista_imagens:
        shutil.move(caminho_atual + "\\" + image_name,
                    caminho_atual + "\\"+pasta+"\\" + image_name)
