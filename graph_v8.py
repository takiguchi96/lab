# coding:utf-8
# 商品をテキストボックスで検索できるようにする

import pandas as pd
from bokeh.models import ColumnDataSource, OpenURL, TapTool
from bokeh.plotting import figure
from bokeh.events import ButtonClick
from bokeh.models import CustomJS, HoverTool, ColumnDataSource, ImageURL, Tabs, TabPanel, Switch, Paragraph, CrosshairTool, Span
from bokeh.models.widgets import Button, Select, RadioGroup, TextInput, TableColumn, DataTable
from bokeh.layouts import Column, Row
from bokeh.io import curdoc
from bokeh.plotting import figure

def main():
    # csvファイルからデータフレームを読み込む
    df_image_t = pd.read_csv("t-sne.csv")
    df_image_p = pd.read_csv("pca.csv")
    df_image_u = pd.read_csv("umap.csv")

    # プロットの色を格納する配列
    p_color = ["skyblue"] * 3000
    df_image_p['colors'] = p_color
    df_image_t['colors'] = p_color
    df_image_u['colors'] = p_color

    # どのデータを使うか判別する関数
    def data_type(method):
        if method == 0:
            df = df_image_p
        elif method == 1:
            df = df_image_t
        elif method == 2:
            df = df_image_u
        return df

    # display_buttonをクリックしたときの処理
    def display_button_callback(event):
        # ソースデータを格納する辞書型リスト
        dict = {}

        # 可視化手法によって使用するデータフレームを変える
        df = data_type(method_radio_group.active)

        # ソースデータに登録する情報の設定
        for column in df.columns:
            dict[column] = df[column].values

        # Dictionaryをソースと紐づける
        source.data = dict

        # プロットし直し
        p.circle(x="x",y="y", size=30, alpha=0.7, fill_color="colors", line_color="colors")
        p_img.image_url(url="image_url", x="x", y="y", source=img_source, w=60, w_units="screen", h=60, h_units="screen",anchor='center')
        p_img.circle(x='x',y='y', alpha=0, source=img_source, size=25)

        # hovertoolの設定(左側のグラフ用)
        hover = HoverTool(tooltips="""
        <div>
            <div>
                <img
                    src="@imageurl" height="auto" alt"@imageurl" width="auto"
                    margin: 0px 15px 15px 0px; "border="2">
                </img>
            </div>
        </div>
        <div style="width: 300px;">
            <div>
            <span style="width: 10px; font-weight: bold;">
            @itemName</span>
            </div>
            <div>
            <span style="width: 10px; ">値段: @itemPrice円</span>
            </div>
        </div>
        """)

        # hovertoolの設定(右側のグラフ用)
        hover_img = HoverTool(tooltips="""
        <div>
            <div>
            <span style="width: 10px; font-weight: bold;">
            @item_name</span>
            </div>
            <div>
            <span style="width: 10px; ">値段: @item_price円</span>
            </div>
        </div>
        """)

        # hoverツールを追加
        p.add_tools(hover)
        p_img.add_tools(hover_img)

    # テキスト検索が行われた際の処理
    def search_button_callback(event):
        # ソースデータに使用する辞書型リスト
        dict = {}
        # 使用するデータフレームを削減法によって変える
        df = data_type(method_radio_group.active)

        # ソースデータに登録する情報の設定
        for column in df.columns:
            dict[column] = df[column].values

        # テキストボックスに入力されている文字列
        text = itemname_input.value
        itemname_list = df['itemName'].tolist()

        colors = ["a"] * 3000

        # 文字列が入力されていた場合
        if text != None:
            for i in range(3000):
                # 文字列を含む場合は赤色に
                if text in itemname_list[i]:
                    colors[i] = "darkorange"
                # 文字列を含まないものは青色に
                else:
                    colors[i] = "skyblue"
        # 空欄の場合は青色に
        else:
            colors = ["blue"] * 3000
        
        dict['colors'] = colors

        # Dictionaryをソースと紐づける
        source.data = dict

        # プロットし直し
        p.circle(x="x",y="y", size=30, alpha=0.7, fill_color="colors", line_color="colors")
        p_img.image_url(url="image_url", x="x", y="y", source=img_source, w=80, w_units="screen", h=80, h_units="screen",anchor='center')
        p_img.circle(x='x',y='y', alpha=0, source=img_source, size=25)       

    # データソースの初期設定
    source = ColumnDataSource(data=dict(length=[], width=[]))
    source.data = {"x": [], "y": [], "search":[]}

    # 可視化手法選択ラジオボタン
    par = Paragraph(text=" 可視化手法を選択")
    method_radio_group = RadioGroup(labels=["PCA", "t-SNE", "umap"], active=0)

    # 商品名検索用テキストボックス(日本語入力するとバグるため保留) 
    itemname_input = TextInput(value="商品名を入力", title="検索")

    # 解析実行ボタン
    display_button = Button(label="表示", button_type="success")
    display_button.on_event(ButtonClick, display_button_callback)

    # 検索ボタン
    search_button = Button(label="検索", button_type="success")
    search_button.on_event(ButtonClick, search_button_callback)

    # マウスオーバーの設定
    width = Span(dimension="width", line_dash="dashed", line_width=2)
    height = Span(dimension="height", line_dash="dashed", line_width=2)    

    # グラフ初期設定
    p = figure(tools="pan,lasso_select,box_select,box_zoom,wheel_zoom,zoom_in,tap,reset",
               title="Analyze Result", width=800, height=900)
    p.circle(x="x", y="y", source=source, size=15, alpha=0.3, fill_color="colors", line_color="colors")
    p.add_tools(CrosshairTool(overlay=[width, height]))

    img_source=ColumnDataSource(data=dict(x=[],y=[],image_url=[],item_url=[],item_name=[],item_price=[]))
    p_img = figure(tools="pan,box_zoom,wheel_zoom,zoom_in,tap,reset",
               title="Analyze Result(Image)", width=800, height=900)
    p_img.image_url(url="imageurl", x="x", y="y", source=img_source, w=80, w_units="screen", h=80, h_units="screen",anchor='center')
    p_img.circle(x='x',y='y', alpha=0, source=img_source)
    p_img.add_tools(CrosshairTool(overlay=[width, height]))

    source.selected.js_on_change('indices', CustomJS(args=dict(source1=source, source2=img_source), code="""
        const inds = cb_obj.indices
        const data = source1.data
        const x = Array.from(inds, (i) => data.x[i])
        const y = Array.from(inds, (i) => data.y[i])
        const image_url = Array.from(inds, (i) => data.imageurl[i])
        const item_url = Array.from(inds, (i) => data.itemUrl[i])
        const item_name = Array.from(inds, (i) => data.itemName[i])
        const item_price = Array.from(inds, (i) => data.itemPrice[i])
        source2.data = { x, y , image_url, item_url, item_name, item_price}
    """)
    )
    
    # 点をクリックすると商品ページへジャンプする
    taptool1 = p.select(type=TapTool)
    taptool1.callback = OpenURL(url="@itemUrl")
    taptool2 = p_img.select(type=TapTool)
    taptool2.callback = OpenURL(url="@item_url")

    operation_area = Column(par ,method_radio_group, display_button, itemname_input, search_button)   
    graph_area = Row(p, p_img)

    layout = Row(children=[operation_area, graph_area])
    curdoc().add_root(layout)

main()