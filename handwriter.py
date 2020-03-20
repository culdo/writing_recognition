#!/usr/bin/env python3
import io
from tkinter import *

import numpy as np
from PIL import Image, ImageDraw

from solve_captchas_with_model import solve_handwriting

scale = 12

class Paint(object):

    width = 750
    height = 250

    def __init__(self):
        self.shape = (1, 784)
        self.root = Tk()
        self.root.title("手寫GUI：->單層網路")

        # self.eraser_button = Button(self.root, text='預測', command=self.use_predictor, font=("Courier", scale))
        # self.eraser_button.grid(row=0, column=1)
        self.implot = None

        self.stringvar = StringVar()
        self.label = Label(self.root, textvariable=self.stringvar, font=("Courier", 20))
        self.label.grid(row=0, column=1)
        self.stringvar.set("預測戳滾輪、筆粗細->")

        self.choose_size_button = Scale(self.root, from_=0, to=50, orient=HORIZONTAL, font=("Courier", scale), length=150, width=20)
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
        self.choose_size_button.set(20)
        self.line_width = self.choose_size_button.get()

        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind("<Button-3>", self.right_click)
        self.c.bind("<Button-2>", self.left_click)
        self.c.bind("<Button-4>", self._on_mousewheel)
        self.c.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        count = self.choose_size_button.get()
        if event.num == 5 or event.delta == -120:
            self.choose_size_button.set(count - 1)
        if event.num == 4 or event.delta == 120:
            self.choose_size_button.set(count + 1)

    def use_predictor(self):
        filename = "my_drawing.jpg"
        # self.image1.save(filename)
        ps = self.c.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save(filename)
        ans = solve_handwriting(np.array(img))
        self.stringvar.set(ans)

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'black'
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
        self.label.config(font=("Courier", 30))
        self.stringvar.set("我是答案")

    def left_click(self, event):
        self.use_predictor()


if __name__ == '__main__':
    Paint()
