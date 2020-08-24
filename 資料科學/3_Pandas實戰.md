#
```
[1]Pandas兩大資料型態: Series與DataFrame
[2]Pandas檔案讀寫
[3]Pandas 的運算
```

```
Python資料分析 第二版
Python for Data Analysis, 2nd Edition
作者： Wes McKinney  
譯者： 張靜雯
歐萊禮出版社   出版日期：2018/10/03

https://github.com/wesm/pydata-book
```
# [1]Pandas兩大資料型態: Series與DataFrame
```

```
### DataFrame屬性與方法
```
pandas 有一些好用的屬性與方法可以快速暸解一個 DataFrame 的外觀與內容：

df.shape：這個 DataFrame 有幾列有幾欄
df.columns：這個 DataFrame 的變數資訊
df.index：這個 DataFrame 的列索引資訊
df.info()：關於 DataFrame 的詳細資訊
df.describe()：關於 DataFrame 各數值變數的描述統計
```
# [2]Pandas檔案讀寫的函數

## 抓取遠方的CSV檔案到Google Colab
```
!curl -O https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv

!ls

births = pd.read_csv('births.csv')

births.head()

```
## 讀入 csv 文字檔
```
import pandas as pd

# 讀入 csv 文字檔
csv_file = "https://storage.googleapis.com/learn_pd_like_tidyverse/gapminder.csv"

gapminder = pd.read_csv(csv_file)
print(type(gapminder))
gapminder.head()
```

## 讀入 excel 試算表
```
xlsx_file = "https://storage.googleapis.com/learn_pd_like_tidyverse/gapminder.xlsx"
gapminder = pd.read_excel(xlsx_file)
print(type(gapminder))

gapminder.head()
gapminder.tail(5)

gapminder.info()
gapminder.describe()
```
## 上傳檔案到Google Colab再執行
```
from google.colab import files
uploaded = files.upload()
```
## 用 Python 抓取 Ubike 開放資料
```
https://jerrynest.io/python-get-ubike-opendata/

https://colab.research.google.com/drive/1jQprW8RIsA_SEFpBm6POF8DgZ1XZ9YH6#scrollTo=A8qmjnFmGyNv
```
```
Ubike 資料說明

每筆資料有以下 14 個欄位，其中有部分是不會變動的資料，如 sno、sna、sarea 等等，待會放資料庫時會獨立出來放

sno：站點代號
 sna：場站名稱(中文)
 tot：場站總停車格
 sbi：場站目前車輛數量
 sarea：場站區域(中文)
 mday：資料更新時間
 lat：緯度
 lng：經度
 ar：地(中文)
 sareaen：場站區域(英文)
 snaen：場站名稱(英文)
 aren：地址(英文)
 bemp：空位數量
 act：全站禁用狀態
```
```
import requests
import json

url = "http://data.taipei/youbike"
data = requests.get(url).json()

print(data)

for key, value in data["retVal"].items():
  sno = value["sno"]
  sna = value["sna"]
  print("NO.", sno, sna)
```
# ＤataFrame的運算
```

```
