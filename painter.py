#!/usr/bin/env python3
import io
import json
from tkinter import *

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus if gpus]

scale = 12
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)
config["model"]["CNN"] = load_model(config["model"]["CNN"])
config["model"]["MLP"] = load_model(config["model"]["MLP"])
# Initialize model
_ = config["model"]["CNN"].predict(np.zeros(config["shape"]["CNN"]))[0]
_ = config["model"]["MLP"].predict(np.zeros(config["shape"]["MLP"]))[0]


class Paint(object):
    def __init__(self, lang="en", width=600, height=600):
        self.lang = lang
        self.width = width
        self.height = height

        self.root = Tk()
        self.NN_button = Button(self.root, command=self.choose_NN, font=("Courier", scale))
        self._apply_nn("MLP")

        self.NN_button.grid(row=0, column=0)

        # self.eraser_button = Button(self.root, text='預測', command=self.use_predictor, font=("Courier", scale))
        # self.eraser_button.grid(row=0, column=1)

        self.stringvar = StringVar()
        if lang == "zh":
            self.label = Label(self.root, textvariable=self.stringvar, font=("Courier", 20))
        elif lang == "en":
            self.label = Label(self.root, textvariable=self.stringvar, font=("Courier", 12))
        self.label.grid(row=0, column=1)
        self.stringvar.set(config["stringvar"][self.lang])

        self.choose_size_button = Scale(self.root, from_=30, to=150, orient=HORIZONTAL, font=("Courier", scale),
                                        length=150, width=20)
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
        self.choose_size_button.set(60)
        self.line_width = self.choose_size_button.get()

        self.c.bind('<B1-Motion>', self._paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind("<Button-3>", self._right_click)
        self.c.bind("<Button-2>", self._left_click)
        self.c.bind("<Button-4>", self._on_mousewheel)
        self.c.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        count = self.choose_size_button.get()
        if event.num == 5 or event.delta == -120:
            self.choose_size_button.set(count - 1)
        if event.num == 4 or event.delta == 120:
            self.choose_size_button.set(count + 1)

    def choose_NN(self):
        if self.mode == "MLP":
            self._apply_nn("CNN")
        else:
            self._apply_nn("MLP")

        self._use_predictor()

    def _apply_nn(self, mode):
        if mode == "CNN":
            option_to = "MLP"
        else:
            option_to = "CNN"
        self.mode = mode

        print(config["use_what"][self.lang] % config["l8n"][mode])
        self.model = config["model"][mode]
        self.shape = config["shape"][mode]
        self.NN_button.config(text=config["switch"][self.lang] % config["l8n"][option_to][self.lang])
        self.root.title(config["title"][self.lang] % config["l8n"][mode][self.lang])

    def _use_predictor(self):
        filename = "my_drawing.jpg"
        # self.image1.save(filename)
        ps = self.c.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        gray = img.convert('L')
        bw = gray.point(lambda x: 1 if x < 128 else 0)
        bw.save(filename)
        array = np.array(bw.resize((28, 28))).reshape(self.shape)
        # array = array.astype("float32")
        # array /= 255
        ans = self.model.predict(array)[0]
        print(np.around(ans, decimals=1))
        self.stringvar.set(np.argmax(ans))

    def _paint(self, event):
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

    def _right_click(self, event):
        self.c.delete("all")
        self.label.config(font=("Courier", 30))
        self.stringvar.set(config["Answer"][self.lang])

    def _left_click(self, event):
        self._use_predictor()


if __name__ == '__main__':
    Paint()
