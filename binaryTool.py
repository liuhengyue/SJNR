from Tkinter import *
from tkMessageBox import *
import PIL.Image
from PIL import  ImageTk
import glob
import os
import shutil
import re

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    # alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key.split('_')[1])]
    alphanum_key = lambda key: (int(re.split('([0-9]+)', key.split('_')[1])[1]), int(re.split('([0-9]+)', key.split('_')[0])[3]))
    return sorted(l, key = alphanum_key)

dataset_to_label = sorted_nicely([img for img in glob.glob("./roi2/*png")])
left = len(dataset_to_label)
output_folder = "./with_ball/"
with_ball = "./with_ball/"
without_ball = "./without_ball/"
current_index = 0
first_image = dataset_to_label[current_index]
try:
    PIL.Image.open(first_image)
except IOError:
    os.remove(first_image)
    current_index+=1
    left-=1
    first_image = dataset_to_label[current_index]


root = Tk()
root.title('Label Tool')

Image = ImageTk.PhotoImage(PIL.Image.open(first_image))
ImageLabel = Label(root, image=Image)
ImageLabel.image = Image
ImageLabel.pack()
T = Text(root, height=2, width=30)
T.pack()
T.insert(END, first_image + '\n')
T.insert(END, left)




def callbackNO(event=None):

    #save current image label to yes
    global current_index
    global left
    print(dataset_to_label[current_index])
    path = dataset_to_label[current_index]
    head, tail = os.path.split(path)
    name = tail.split(".")[0]
    extension = tail.split(".")[1]
    # copyfile(path, output_path)
    shutil.move(dataset_to_label[current_index],without_ball + tail)
    #change to next picture
    current_index = current_index + 1
    path = dataset_to_label[current_index]
    try:
        PIL.Image.open(path)
    except IOError:
        os.remove(path)
        current_index+=1
        left-=1
        path = dataset_to_label[current_index]
    updated_picture = ImageTk.PhotoImage(PIL.Image.open(path))
    ImageLabel.configure(image = updated_picture)
    ImageLabel.image = updated_picture
    T.delete('0.0',END)
    T.insert(END, path + '\n')
    left-=1
    T.insert(END, left)


def callbackYES(event=None):
    #save current image label to yes
    global current_index
    global left
    print(dataset_to_label[current_index])
    path = dataset_to_label[current_index]
    head, tail = os.path.split(path)
    name = tail.split(".")[0]
    extension = tail.split(".")[1]
    # copyfile(path, output_path)
    shutil.move(dataset_to_label[current_index],with_ball + tail)

    #change to next picture
    current_index = current_index + 1
    path = dataset_to_label[current_index]
    try:
        PIL.Image.open(path)
    except IOError:
        os.remove(path)
        current_index+=1
        left-=1
        path = dataset_to_label[current_index]
    updated_picture = ImageTk.PhotoImage(PIL.Image.open(path))
    ImageLabel.configure(image = updated_picture)
    ImageLabel.image = updated_picture
    T.delete('0.0',END)
    T.insert(END, path + '\n')
    left-=1
    T.insert(END, left)


root.bind('3', callbackYES)
root.bind('4', callbackNO)
Button(text='With Ball (3)', command=callbackYES).pack(fill=X)
Button(text='Without Ball (4)', command=callbackNO).pack(fill=X)

root.mainloop()
