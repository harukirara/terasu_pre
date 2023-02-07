import streamlit as st
from keras.models import load_model
import random
from sklearn import preprocessing
import numpy as np
import pandas as pd
from PIL import Image
import pickle


st.title("テラスタイプ予測AI")

#タイプ相性のファイルの読み込み
df_type=pd.read_csv("./data/typeaisyo_seme.csv",index_col=0)

with st.form(key='profile form'):
    #タイプ1の選択
    type_1=st.selectbox("タイプ1:",
    ['ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'こおり', 'かくとう', 'どく', 'じめん',
    'ひこう', 'エスパー', 'むし', 'いわ', 'ゴースト', 'ドラゴン', 'あく', 'はがね', 'フェアリー'])
    
    #タイプ2の選択
    type_2=st.selectbox("タイプ2:",
    ['なし','ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'こおり', 'かくとう', 'どく', 'じめん',
    'ひこう', 'エスパー', 'むし', 'いわ', 'ゴースト', 'ドラゴン', 'あく', 'はがね', 'フェアリー'])

    if type_2=="なし":
        pokemon_data=df_type[type_1]*1
    else:
        pokemon_data=df_type[type_1]*df_type[type_2]
    
    pokemon_data=pd.DataFrame(pokemon_data)
    pokemon_data=pokemon_data.T
    
    #タイプ和の追加
    pokemon_data["タイプ和"]=0
    for j in pokemon_data.columns[0:-1]:
        pokemon_data["タイプ和"]+=pokemon_data.loc[:,j]
    pokemon_data["タイプ和"]=pokemon_data["タイプ和"]*random.uniform(1.00,1.05)

    col1,col2=st.columns(2)

    #種族値の入力
    with col1:
        Hp=st.text_input("HP")
        Attack=st.text_input("攻撃")
        Block=st.text_input("防御")
    
    with col2:
        Contact=st.text_input("特攻")
        Diffencet=st.text_input("特防")
        Speed=st.text_input("素早")

    #ボタン
    submit_btn=st.form_submit_button("予測")
    cancel_btn=st.form_submit_button("リセット")

    if len(Hp)!=0 or len(Attack)!=0 or len(Block)!=0 or len(Contact)!=0 or len(Diffencet)!=0 or len(Speed)!=0:
        pokemon_data["HP"]=Hp
        pokemon_data["攻撃"]=Attack
        pokemon_data["防御"]=Block
        pokemon_data["特攻"]=Contact
        pokemon_data["特防"]=Diffencet
        pokemon_data["素早"]=Speed

        #文字列の列を数値に変換する際に例外処理の判定
        flag=True
        try:
            pokemon_data=pokemon_data.astype("float")
        except ValueError:
            flag=False
        
        if flag:
            pokemon_data=pokemon_data.reindex(columns=['HP', '攻撃', '防御', '特攻', '特防', '素早','ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'こおり', 'かくとう', 'どく', 'じめん',
            'ひこう', 'エスパー', 'むし', 'いわ', 'ゴースト', 'ドラゴン', 'あく', 'はがね', 'フェアリー','タイプ和'])

            #標準化の読み込み
            with open('./pkl/scaler.pkl', mode='rb') as fp:
                scaler=pickle.load(fp)
            
            #モデルへの入力
            input=scaler.transform(pokemon_data)

            #モデルでの予測
            model = load_model('./model/model.h5')
            y_pred = model.predict(input)
            y_pred_max = np.argmax(y_pred, axis=1)

            #ラベルエンコーダの読み込み
            with open('./pkl/label.pkl', mode='rb') as fp:
                le=pickle.load(fp)
            predict=le.inverse_transform(y_pred_max)

        #テラスタイプの表示
        if submit_btn:
            if flag:
                st.write(f'テラスタイプ: {predict[0]}')
            else:
                st.write(f'種族値の入力は数値にしてください')


    


