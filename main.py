import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import accuracy
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import io

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # 学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        # 学習時に使ったのと同じ順伝播
        h = self.feature(x)
        h = self.fc(h)
        return h

# Streamlitアプリのタイトル
st.title('Animal Predict App')

# ファイルのアップロードウィジェット
uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["png", "jpg", "gif", "jpeg"])

# ファイルがアップロードされた場合
if uploaded_file is not None:
    # アップロードされた画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 学習済みモデルの読み込み
    model = Net()
    model.load_state_dict(torch.load('dog_cat_data.pt', map_location=torch.device('cpu')))
    model.eval()

    # 画像の前処理
    img_tensor = transform(image).unsqueeze(0)  # バッチサイズ1のテンソルに変換

    # 予測結果の計算
    with torch.no_grad():
        outputs = model(img_tensor)

    # 予測結果の表示
    class_names = ["猫", "犬"]
    _, predicted = torch.max(outputs, 1)
    prediction = class_names[predicted]

    st.write(f"予測結果: {prediction}")
