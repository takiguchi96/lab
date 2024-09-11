import numpy as np
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from bokeh.layouts import row, column
from bokeh.models import (ColumnDataSource, DataTable, TableColumn, PointDrawTool, StringFormatter, NumberFormatter, 
                          CustomJS, LabelSet, Legend, LegendItem, Button, Div)
from bokeh.plotting import figure, curdoc

# 3つの分野の単語リスト
joy_words = ["喜び", "スッキリ", "ワクワク", "和む", "大喜び", "嬉しい", "幸せ", "悦び", "愉快", "感動", "感心", "懐かしい", "楽しい", "歓喜", "気持ちいい", "満足", "爽快", "癒される", "笑い", "興奮する", "落ち着く", "高ぶる", "好き", "いつくしみ", "ラブ", "優しい", "同情"]
like_words = ["好き", "好み", "好意", "尊敬", "尊敬する", "思いやり", "恋する", "愛", "愛する", "愛情", "愛情のこもった", "愛着", "慈愛", "憧れる", "欲する", "気遣う"]
sadness_words = ["悲しみ", "かわいそう", "がっかり", "ため息", "ヘコむ", "不幸", "不幸せ", "傷心", "切ない", "哀しみ", "哀れ", "喪失感", "嘆き", "困る", "失望", "孤独", "寂しい", "屈辱", "心が痛む", "悲しい", "悲しさ", "悲哀", "情けない", "惨め", "憂い事", "憂き目", "憂鬱", "憐れ", "戸惑う", "気分が晴れない", "泣ける", "苦労", "萎える", "落胆", "虚しい", "辛い"]

# BERTのモデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 各言葉の埋め込みベクトルを取得
def get_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].numpy()

# 全単語リストと対応する色
words = joy_words + like_words + sadness_words
colors = ['red'] * len(joy_words) + ['green'] * len(like_words) + ['blue'] * len(sadness_words)

embeddings = np.array([get_embedding(word) for word in words])
d = embeddings.shape[1] # 埋め込み次元数
n = len(words) # 単語数

# t-SNEを使用して次元削減
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# データフレームにテキスト、2次元座標をまとめる
df = pd.DataFrame({'text': words, 'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'color': colors})

# データソースを準備
source = ColumnDataSource(df)

# AとBのランクと初期化
rank = 2  # 低ランク行列のランク
A = np.random.randn(d, rank)  # 埋め込み次元数 x ランク
B = np.random.randn(rank, n)  # ランク x 埋め込み次元数

# 散布図を作成
p = figure(title="感情マッピング", tools="", width=800, height=600)

# データポイントをプロット
scatter_renderer = p.circle('x', 'y', source=source, size=10, color='color', alpha=0.5)

# ラベルを追加
labels = LabelSet(x='x', y='y', text='text', level='glyph', x_offset=5, y_offset=5, source=source, text_font_size='10pt', text_color='black')
p.add_layout(labels)

# PointDrawToolを追加（データ移動用）
move_tool = PointDrawTool(renderers=[p.renderers[-1]], empty_value='black', add=False)
move_tool.description = "Move Points"
p.add_tools(move_tool)

# PointDrawToolを追加（データ追加用）
add_tool = PointDrawTool(renderers=[p.renderers[-1]], empty_value='red', add=True)
add_tool.description = "Add Points"
p.add_tools(add_tool)

# ドラッグツールをデフォルトに設定
p.toolbar.active_drag = move_tool

# 更新ボタン
update_button = Button(label="Update Features", button_type="success")

# ステータス表示用
status_div = Div(text="")

# A, Bの更新関数
def update_features():
    global A, B, embeddings_2d
    # AとBを最適化（とりあえず適当に更新）
    A -= 0.01 * np.random.randn(*A.shape)  # 適当な更新ステップ（仮）
    B -= 0.01 * np.random.randn(*B.shape)
    
    # 新しい特徴ベクトルを計算
    updated_embeddings = embeddings + (A @ B).T
    
    # t-SNEで再次元削減
    embeddings_2d = tsne.fit_transform(updated_embeddings)
    
    # データソースを更新
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    source.data = dict(df)
    
    # ステータスを更新
    status_div.text = "Features updated successfully!"

# ボタンが押されたときにupdate_featuresを呼ぶ
update_button.on_click(update_features)

# データテーブルのカラムを定義
columns = [
    TableColumn(field="x", title="X Coordinate", formatter=NumberFormatter(format="0.0")),
    TableColumn(field="y", title="Y Coordinate", formatter=NumberFormatter(format="0.0")),
    TableColumn(field="text", title="text", formatter=StringFormatter(font_style="bold"))
]

# データテーブルを作成
data_table = DataTable(source=source, columns=columns, editable=True, width=400)

# 凡例の追加
legend = Legend(items=[
    LegendItem(label="喜び", renderers=[scatter_renderer], index=0),
    LegendItem(label="好き", renderers=[scatter_renderer], index=len(joy_words)),
    LegendItem(label="悲しみ", renderers=[scatter_renderer], index=len(joy_words) + len(like_words))
], location="center")

p.add_layout(legend, 'right')

# レイアウトを作成して表示
layout = column(row(p, data_table), update_button, status_div)

# 現在のドキュメントにレイアウトを追加
curdoc().add_root(layout)
curdoc().title = "感情マッピング"