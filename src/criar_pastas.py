import os

path = os.getcwd()
print("The current working directory is %s" % path)


caminhos = ["angry", "contemptuous", "disgusted",
            "fearful", "happy", "neutral", "sad", "surprised"]


for pasta in caminhos:
    try:
        os.mkdir(pasta)
    except OSError:
        print("Creation of the directory %s failed" % pasta)
    else:
        print("Successfully created the directory %s " % pasta)
