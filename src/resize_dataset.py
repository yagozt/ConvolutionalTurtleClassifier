from PIL import Image
import os
import glob

caminhos = ["angry", "contemptuous", "disgusted",
            "fearful", "happy", "neutral", "sad", "surprised"]

caminho_atual = os.getcwd()

print(os.getcwd())


for caminho in caminhos:
    os.chdir(caminho_atual)
    try:
        path = "resized_dataset2/" + caminho
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

    os.chdir("resized_dataset/" + caminho)

    lista_imagens = glob.glob("*")

    for images in lista_imagens:
        if(os.getcwd() == "C:\\Users\\yago\\Documents\\Yago\\TCC\\ConvolutionalTurtleClassifier"):
            os.chdir("resized_dataset/" + caminho)
        image = Image.open(images)
        os.chdir(caminho_atual)
        image.thumbnail((image.width / 2, image.height / 2))

        image.save(caminho_atual + "\\resized_dataset2\\" +
                   caminho + "\\" + images)

    print("Finalizado folder " + caminho)

print("Finalizado alteração do tamanho das imagens.")
# image_resized = image.copy()

# image_resized.thumbnail((image.width / 2, image.height / 2))

# print(image_resized.height)
# print(image_resized.width)
