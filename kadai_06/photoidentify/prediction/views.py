#Djangoが提供するrender関数をインポートする
#render関数は、特定のテンプレートとデータをもとにHTMLを作成する
from django.shortcuts import render

#forms.pyで作成したImageUploadFormクラスをインポートする
#これで、views.pyでImageUploadFormクラスを利用できる
from .forms import ImageUploadForm

#ランダムな結果を生成するための関数
#import random　は判定結果をランダムで表示させるロジックのため削除する

#Djangoのプロジェクト設定情報を取り扱うモジュールをインポート
from django.conf import settings

#Kerasのload_model関数をインポート。この関数で予測モデルを読み込む
from tensorflow.keras.models import load_model

#Kerasのload_img関数とimg_to_array関数をインポート
#アップロードされた画像を予測モデルに入力できる形式へ変換するために使用する
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


#主にデータの取り扱いを担うioモジュールのBytesIOをインポート
#アップロードされた画像ファイルを予測モデルに適した形式へ変換するために使用する
from io import BytesIO

#OS関連の操作をするosモジュールをインポート
#予測モデルのファイルパスを生成するために使用する
import os

#requestを受け取って処理を行うpredict関数を定義する
#引数は、リクエストに関する情報を受け取れるようにrequestを設定
#requestはDjangoがビュー関数に自動で渡してくれるリクエストオブジェクト。
def predict(request):

    #HTTPメソッドによって、条件を分岐させている
    #GETメソッドは、ブラウザで特定のURLにアクセスするときに使う
    #POSTメソッドは、フォームを通じてデータ送信するときに使う
    #参考；HTTPメソッドは他に、PUT、PATCH、DELETEがよく使われる

    if request.method =='GET':
        #GETリクエストによるアクセス時の処理を記述
        form = ImageUploadForm()
        #新たな画像アップロードフォームを生成しform変数に代入する
        #引数に何も指定しない場合は、空の状態を生成できる


        return render(request, 'home.html', {'form': form})
        #HTMLテンプレートを生成・表示させるための処理
        #render関数は、指定されたテンプレート（ここではhome.html）を用いて
        #HTMLを生成し、request元にレスポンスする役割を果たす
        #home.htmlは後で作成する

        #３つの引数の説明
        #request:request情報を含むオブジェクト。HTMLテンプレートが利用できるようになる
        #home.html：HTMLテンプレートのファイル名
        #{'form': form}：連想配列。formキーに、form変数(画像アップロードのフォーム)を値として設定
        #これで、home.htmlテンプレート内でform変数を活用して画像アップロードフォームを表示できる
        #以上でGETリクエストによるアクセス時の処理の実装は完了


    if request.method =='POST':
        #POSTリクエストによるアクセス時の処理を記述

        #POSTリクエストで送信されたデータを引数として、ImageUploadFormクラスのインスタンスを作成し、form変数に代入する
        #request.POSTはファイル以外の送信されたデータ、requestFILESは送信されたファイルを意味する
        #request.POSTには、csrfmiddlewaretoken(CSRF)と呼ばれるデータが含まれており、セキュリティ攻撃を防ぐために活用される。
        form = ImageUploadForm(request.POST, request.FILES)

        #送信されたフォームのデータが適切か（必要なフィールがすべて含まれているか、画像が破損していないかなど）をチェックする
        if form.is_valid():
            
            #フォームから送信された画像データを取得して　img_file変数に代入する
            #form.cleaned_data は送信されたデータの連想配列
            #'image'キーに対応するデータは、アップロードされた画像に当たる
            img_file = form.cleaned_data['image']

             #アップロードフォームから取得した画像データを、画像ファイルのように扱えるよう変換するためのコード
            img_file = BytesIO(img_file.read())
            
            img = load_img(img_file, target_size=(224,224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,224,224,3))
            img_array = preprocess_input(img_array)  # VGG16用に前処理
            
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            result = model.predict(img_array)

            preds = model.predict(img_array)

            # decode_predictionsを使って、上位5つのラベルと確率を取得
            decoded_preds = decode_predictions(preds, top=5)[0]

            # 予測結果を処理して表示
            prediction_list = []
            for label, class_name, prob in decoded_preds:
                prediction_list.append(f"{class_name}: {prob * 100:.2f}%")
            
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form':form, 'prediction': prediction_list, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
        

