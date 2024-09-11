import numpy as np
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, DataTable, TableColumn, PointDrawTool, StringFormatter, NumberFormatter, CustomJS, LabelSet, Legend, LegendItem, Div)
from bokeh.plotting import figure, curdoc, output_notebook
from tqdm import tqdm

# BERTのモデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 各言葉の埋め込みベクトルを取得
def get_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].numpy()

# 3つの分野の単語リスト
joy_words = ["喜び", "スッキリ", "ワクワク", "和む", "大喜び", "嬉しい", "幸せ", "悦び", "愉快", "感動", "感心", "懐かしい", "楽しい", "歓喜", "気持ちいい", "満足", "爽快", "癒される", "笑い", "興奮する", "落ち着く", "高ぶる", "好き", "いつくしみ", "ラブ", "優しい", "同情"]
like_words = ["好き", "好み", "好意", "尊敬", "尊敬する", "思いやり", "恋する", "愛", "愛する", "愛情", "愛情のこもった", "愛着", "慈愛", "憧れる", "欲する", "気遣う"]
sadness_words = ["悲しみ", "かわいそう", "がっかり", "ため息", "ヘコむ", "不幸", "不幸せ", "傷心", "切ない", "哀しみ", "哀れ", "喪失感", "嘆き", "困る", "失望", "孤独", "寂しい", "屈辱", "心が痛む", "悲しい", "悲しさ", "悲哀", "情けない", "惨め", "憂い事", "憂き目", "憂鬱", "憐れ", "戸惑う", "気分が晴れない", "泣ける", "苦労", "萎える", "落胆", "虚しい", "辛い"]

# 全単語リストと対応する色
words = joy_words + like_words + sadness_words
colors = ['red'] * len(joy_words) + ['green'] * len(like_words) + ['blue'] * len(sadness_words)

# 初期の埋め込みベクトルを取得
embeddings = np.array([get_embedding(word) for word in tqdm(words)])  # tqdmで進捗を表示

# t-SNEを使用して次元削減
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# データフレームにテキスト、2次元座標をまとめる
df = pd.DataFrame({'text': words, 'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'color': colors})

# データソースを準備
source = ColumnDataSource(df)

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

# カスタムJSコールバックでデフォルトの色を設定
callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var color = data['color'];
    var new_index = color.length - 1;
    color[new_index] = 'red';
    source.change.emit();
""")
source.js_on_change('data', callback)

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

# updateボタンを追加
button = Button(label="Update", button_type="success")

# 更新状況を表示するDiv
status_div = Div(text="")

# 元の座標を保存
original_coords = embeddings_2d.copy()

# 近傍情報を取得する関数
def get_nearest_neighbors(coords, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices

# BERT埋め込みを再計算して更新するコールバック関数
def update_coordinates():
    status_div.text = "Updating..."
    curdoc().add_next_tick_callback(update_embeddings)

def update_embeddings():
    # 現在の座標を取得
    current_coords = np.column_stack((source.data['x'], source.data['y']))

    # 動かされたデータ点の検出
    moved_points = np.any(current_coords != original_coords, axis=1)

    if np.any(moved_points):
        status_div.text = "Calculating new embeddings..."
        
        # 新しい埋め込みベクトルを取得
        new_embeddings = np.array([get_embedding(word) for word in tqdm(df['text'])])

        # 近傍情報を取得
        distances, indices = get_nearest_neighbors(new_embeddings)

        # モデルを学習
        X_train = []
        y_train = []

        for i in range(len(new_embeddings)):
            if moved_points[i]:
                for j in indices[i]:
                    X_train.append(np.concatenate((new_embeddings[i], new_embeddings[j])))
                    y_train.append(current_coords[i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # ランダムフォレストを使用
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 新しい座標を予測
        X_predict = []
        for i in range(len(new_embeddings)):
            for j in indices[i]:
                X_predict.append(np.concatenate((new_embeddings[i], new_embeddings[j])))

        X_predict = np.array(X_predict)

        # デバッグ用出力
        print("X_predict shape:", X_predict.shape)
        print("Expected new_coords shape:", (len(new_embeddings), 2))

        new_coords = model.predict(X_predict).reshape(len(new_embeddings), 2)

        # データフレームの更新
        df['x'] = new_coords[:, 0]
        df['y'] = new_coords[:, 1]
        source.data = ColumnDataSource.from_df(df)

        status_div.text = "Update complete!"
    else:
        status_div.text = "No points were moved. Update not necessary."

button.on_click(update_coordinates)

# レイアウトを作成して表示
layout = column(button, status_div, row(p, data_table))

# 現在のドキュメントにレイアウトを追加
curdoc().add_root(layout)
curdoc().title = "感情マッピング"
