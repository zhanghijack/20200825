#
```
[1]Pandas兩大資料型態: Series與DataFrame
[2]Pandas檔案讀寫的函數


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
# [2]Pandas檔案讀寫的函數

## 抓取遠方的CSV檔案到Google Colab
```
!curl -O https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv

!ls

births = pd.read_csv('births.csv')

births.head()

```
