import io
from tkinter import *

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

scale = 12
cnn = load_model("models/cnn_model.h5")
mlp = load_model("models/mlp_model.h5")

l8n = {"MLP": "全連接", "CNN": "捲積"}
config = {"use_what":
              {"zh": "使用%s神經網路",
               "en": "Using %s"},
          "title":
              {"zh": "手寫GUI：->%s網路",
               "en": "Handwriting GUI：->%s"},
          "switch":
              {"zh": "換%s",
               "en": "use %s"},
          "stringvar":
              {"zh": "預測戳滾輪、筆粗細->",
               "en": "predict using middle button > scrolling change size"},
          "Answer":
              {"zh": "我是答案",
               "en": "Ans"},
          "shape":
              {"MLP": (1, 784),
               "CNN": (1, 28, 28, 1)},
          "model":
              {"MLP": mlp,
               "CNN": cnn}
          }


class Paint(object):
    def __init__(self, lang="zh", width=600, height=600):
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
        self.label = Label(self.root, textvariable=self.stringvar, font=("Courier", 20))
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
        self.choose_size_button.set(90)
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

    def choose_NN(self):
        if self.model == mlp:
            self._apply_nn("CNN")
        else:
            self._apply_nn("MLP")

        self.use_predictor()

    def _apply_nn(self, mode):
        if mode == "CNN":
            switch_to = "MLP"
        else:
            switch_to = "CNN"

        print(config["use_what"][self.lang] % l8n[mode])
        self.model = config["model"][mode]
        self.shape = config["shape"][mode]
        self.NN_button.config(text=config["switch"][self.lang] % l8n[switch_to])
        self.root.title(config["title"][self.lang] % l8n[mode])

    def use_predictor(self):
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
