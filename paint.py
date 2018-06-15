from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image, ImageDraw
from keras.models import load_model
import numpy as np


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    width = 600
    height = 600
    model = load_model("cnn_model.h5")

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='預測', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=25, to=50, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.image1 = Image.new("L", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image1)
        self.c.grid(row=1, columnspan=5)
        self.item = None

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind("<Button-3>", self.rightclick)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        filename = "my_drawing.jpg"
        self.image1.save(filename)
        array = np.array(self.image1.resize((28, 28))).reshape(1, 784)
        # print(img.shape)
        ans = self.model.predict(array)[0]
        print(ans.tolist().index(1))
        # print(self.image1.resize((28, 28)))


    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.item = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], width=self.line_width, fill=255)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def rightclick(self, event):
        self.c.delete("all")
        self.image1 = Image.new("L", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image1)


if __name__ == '__main__':
    Paint()
