from tkinter import *
from PIL import Image, ImageDraw
from keras.models import load_model
import numpy as np
import io

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    width = 600
    height = 600


    def __init__(self):
        self.model = load_model("mlp_model.h5")
        print("使用單層神經網路")
        self.shape = (1, 784)
        self.root = Tk()

        self.color_button = Button(self.root, text='換網路', command=self.choose_NN)
        self.color_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text='預測', command=self.use_predictor)
        self.eraser_button.grid(row=0, column=1)

        self.choose_size_button = Scale(self.root, from_=30, to=150, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=2)

        self.c = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.image1 = Image.new("L", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image1)
        self.c.grid(row=1, columnspan=3)
        self.item = None

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        # self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind("<Button-3>", self.right_click)

    def choose_NN(self):
        if self.shape == (1, 784):
            print("使用捲積神經網路")
            self.model = load_model("cnn_model.h5")
            self.shape = (1, 28, 28, 1)
        else:
            print("使用單層神經網路")
            self.model = load_model("mlp_model.h5")
            self.shape = (1, 784)

    def use_predictor(self):
        filename = "my_drawing.jpg"
        # self.image1.save(filename)
        ps = self.c.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        gray = img.convert('L')
        bw = gray.point(lambda x: 255 if x < 128 else 0)
        bw.save(filename)
        array = np.array(bw.resize((28, 28))).reshape(self.shape)
        ans = self.model.predict(array)[0]
        print(ans.tolist().index(1))

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            new = event.x, event.y
            self.item = self.c.create_line(self.old_x, self.old_y, *new,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def right_click(self, event):
        self.c.delete("all")


if __name__ == '__main__':
    Paint()
