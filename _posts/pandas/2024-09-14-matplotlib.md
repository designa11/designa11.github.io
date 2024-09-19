---
layout: post
title: matplotlib
categories: Matplotlib
tags: [Matplotlib]
excerpt_image: https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/plt.png?raw=true
---

```python
import matplotlib.pyplot as plt
import seaborn as sns # 색살 설정이 더 자유롭고 다채로운 시각화 도구
import numpy as np
import pandas as pd
```

#### Line Plot(선그래프)

```python
x = np.arange(1,6)
x
np.random.seed(3)
y = np.random.randint(1,11,size = 5)
y
```

```python
plt.plot(x,y)
plt.show
```
![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/1.png?raw=true)



- score 데이터 선 그래프로 시각화해보기

```python
# df으로 불러와서 score 변수에 담아주기
score= pd.read_csv('./data/score.csv', encoding = 'euc-kr')
score
# 1반 성적 데이터에 접근해서 선그래프 그리기
x = ['python','db','java','cr','Web']
y = score['1반'] #1반 컬럼 접근 
# y = score.loc[:,'1반']
# y = score.iloc[:,1]
y2 = score['2반']
plt.plot(x,y,label = '1class')
# plt.show # plt.show (이미지 자체를 출력해줌 ( 안쓰면 뭔 글자랑 같이나옴))
plt.plot(x,y2, label = '2class')
plt.legend()
plt.show()
plt.plot(x,y)
plt.show()
```
![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/2.png?raw=true)
![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/3.png?raw=true)



- style option 

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/1.png?raw=true)

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/2.png?raw=true)

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/3.png?raw=true)

```python
plt.figure(figsize = (5,4)) #(x크기,y크기)
plt.plot(x,y,marker = "*",ms = 20, mec = 'r', ls = "-.")
# plt.show # plt.show (이미지 자체를 출력해줌 ( 안쓰면 뭔 글자랑 같이나옴))
plt.show()
```
![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/4.png?raw=true)



```python
score= pd.read_csv('./data/score.csv', encoding = 'euc-kr')
score
# 1반 성적 데이터에 접근해서 선그래프 그리기
x = ['python','db','java','cr','Web']
y = score['1반'] #1반 컬럼 접근 
# y = score.loc[:,'1반']
# y = score.iloc[:,1]
y2 = score['2반']
y3 = score['3반']
y4 = score['4반']
plt.plot(x,y,x,y2,x,y3,x,y4, marker = "o", label = '1class')
# plt.plot(x,y2, marker = "o",label = '2class')
# plt.plot(x,y3, marker = "o",label = '3class')
# plt.plot(x,y4, marker = "o",label = '4class')
plt.legend()

plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/5.png?raw=true)


```python
plt.plot(x,y4,x,y3,x,y2,x,y, marker = "o",label = '4class')
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/6.png?raw=true)

- 한국어 글꼴 인식하게 설정

```python
from matplotlib import font_manager as fm
font_path = 'C:\Windows\Fonts\HMFMPYUN.TTF'
font_name = fm.FontProperties(fname = font_path).get_name()
print(font_name)
plt.rc('font',family = font_name)
```

```
Pyunji R

```

```python
subject = ['py', 'db', 'java', 'cr', 'web'] # x 축 이름들

plt.figure(figsize=(5,4))
def draw_score(col_name):
    sc = score[col_name] # y축 값들 (1차원)
    plt.plot(subject,sc, marker="o", label=col_name)

for i in score.columns[1:]:
    draw_score(i)

plt.legend(loc = "best")
plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/7.png?raw=true)


- 장래 인구 변동 데이터를 이용하여 시각화 (출처 : 국가 통계포털 사이트 제공)

```python
# 데이터 불러와서 data 변수에 담기 (인코딩, 불러오면서 인덱스 값을 "인구변동요인별" 로 설정해주기)
# 2. 데이터 살펴보기 (전체 정보, 컬럼명, 인덱스 개수, 데이터 크기, 결측치 여부, 데이터 타입)
# 3. 출생아수와 사망자수 선그래프 시각화 해보기 (x축 = 연도 , y축 = 인구수)
# 4. 스타일 욥션도 지정해보기
data = pd.read_csv('./data/장래_인구변동_KOSIS.csv', encoding = 'euc-kr', index_col = "인구변동요인별")
data
```

```python
#Q2.
data.info()
data.columns
len(list(data.columns))
data.isnull()
data.dtypes
```

```
<class 'pandas.core.frame.DataFrame'>
Index: 3 entries, 인구(천명) to 사망자수(천명)
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   2020    3 non-null      int64
 1   2030    3 non-null      int64
 2   2040    3 non-null      int64
 3   2050    3 non-null      int64
 4   2060    3 non-null      int64
 5   2070    3 non-null      int64
dtypes: int64(6)
memory usage: 168.0+ bytes

```

```python
#Q3.
list(data.index)
list(data.columns)

# for i in data.index:
#     for j in data.columns:
#         # plt.plot(data.loc[i])
#         print(data.loc[i,[j]])
#         # print(data.loc[[i],[j]])

def con(cs):
    a = data[cs]
    print(a)

con('2020')
```

```
인구변동요인별
인구(천명)      51836
출생아수(천명)      275
사망자수(천명)      308
Name: 2020, dtype: int64

```

```python
#ans Q3.
x = data.columns
birth = data.loc['출생아수(천명)'] # data.iloc[1]
death = data.loc['사망자수(천명)']

plt.plot(x,birth,marker = '*',label = 'birth')
plt.plot(x,death,marker = 'o' ,label = 'death')
plt.legend()
plt.grid(axis = 'both')
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/8.png?raw=true)



![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/4.png?raw=true)

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/5.png?raw=true)

```python
# from matplotlib import font_manager as fm
# font_path = 'C:\Windows\Fonts\HMFMPYUN.TTF'
# font_name = fm.FontProperties(fname = font_path).get_name()
# print(font_name)
# plt.rc('font',family = font_name)


x = range(2020,2071,10)
birth = data.loc['출생아수(천명)'] # data.iloc[1]
death = data.loc['사망자수(천명)']

plt.plot(x,birth,marker = '*',label = '출생아수(천명)')
plt.plot(x,death,marker = 'o' ,label = '사망자수(천명)')
plt.legend()
plt.xlabel("연도")
plt.ylabel("사람수")
plt.title("국가통계 요인별 장래 인구 동향")
plt.grid(axis = 'both')
plt.xticks(range(2020,2071,5))
plt.yticks(range(150,751,50))
plt.savefig('./data/국가통계_요인별_장래인구동향.png', dpi = 300, bbox_inches = 'tight') # dpi = 해상도/ 이름이 같으면 계속 덮어쓰기
#bbox_inches =  그래프 여백 설정
# plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/9.png?raw=true)



```python
x = range(2020,2071,10) # x = range(2020,2071,10)
birth = data.loc["출생아수(천명)"] # data.iloc[1]
death = data.loc["사망자수(천명)"] # data.iloc[2]

# 시각화
plt.plot(x, birth, marker="*", label='birth')
plt.plot(x, death, marker="o", label='death')
plt.legend()
plt.title("국가통계 요인별 장래 인구 동향")
plt.xlabel("년도"); plt.ylabel("사람수")
plt.grid()
plt.xticks(range(2020,2071,5)); plt.yticks(range(150,751,50))
plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/10.png?raw=true)


#### 기온 데이터 실습
- 데이터 불러오기

```python
#skiprows 첫번째부터 설정한 행까지 행 스
tdata = pd.read_csv('./data/기온데이터(19992023).csv', encoding = 'euc-kr', skiprows = range(0,7), delimiter = ",")
#delimiter = ""
#크기확인 :shape
tdata
tdata.shape
#정보 확인 : info()
tdata.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8767 entries, 0 to 8766
Data columns (total 5 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   날짜       8767 non-null   object 
 1   지점       8767 non-null   int64  
 2   평균기온(℃)  8750 non-null   float64
 3   최저기온(℃)  8764 non-null   float64
 4   최고기온(℃)  8764 non-null   float64
dtypes: float64(3), int64(1), object(1)
memory usage: 342.6+ KB

```

```python
#1. 전체 평균기온 시각화해보기
tdata.head(3)
# 평균 기온 접근
# plt.plot(x,y) 사실 y값만 넣어도 그려짐
# y축에 -기호 출력
plt.rcParams['axes.unicode_minus']=False
plt.plot(tdata['평균기온(℃)'])
plt.grid()

tdata['평균기온(℃)'].describe()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/11.png?raw=true)


```python
# 2. 내 생일 날짜의 평균 기온 시각화해보기
plt.plot(tdata['평균기온(℃)'][tdata['날짜'].str[6:] == '04-24'])
plt.show()

```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/12.png?raw=true)


```python
#2. Ans
tdata['일자'] = tdata['날짜'].str[6:]
y = tdata[tdata['일자']== '04-24'].loc[:,'평균기온(℃)']
plt.plot(y)
plt.show
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/13.png?raw=true)


#### histogram(히스토그램)
- 수치데이터를 범위로 설정하여 빈도수 값을 표현하는 그래프

```python
np.random.seed(3)
a = np.random.randint(1,256,size = 100)
a.size
a
```

```python
plt.hist(a, bins = 200) #막대그래프                  bins  = 구간의 개수를 설정
plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/14.png?raw=true)


![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/6.png?raw=true)

```python
ntdata = pd.read_csv('./data/기온데이터(19992023).csv', encoding = 'euc-kr', skiprows = range(0,7), delimiter = ",")
ntdata
```

```python
a = input("월 입력 (01, 02 ..., 11, 12) >>")
month = ntdata['날짜'].str[6:8]
plt.hist(ntdata['최고기온(℃)'][month == a], bins = 100, alpha = 0.5)
plt.hist(ntdata['최저기온(℃)'][month == a], bins = 100, alpha = 0.5)

plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/15.png?raw=true)


#### Bar Plot(범주데이터 시각화)
- 카테고리의 개수(빈도)를 셀 때 사용하는 그래프

```python
np.random.seed(6)
bar_x = np.arange(1,4)
bar_y = np.random.randint(50,100, size = 3) #value_counts() 의 결과
#시리즈로 변경
s = pd.Series(bar_y, index = bar_x)
s

# bar
plt.bar(s.index,s, color = ['gray', 'black','pink'])
plt.grid() # 눈금자 앞에 오는 건 바꿀 수가 없음.
for i in range(len(bar_x)):
    #좌표를 만들어내는 값으로 인덱스 활용
    plt.text(bar_x[i], bar_y[i]-30, f'{bar_y[i]}',fontdict = {'color':'white', 'size' : 30})
plt.show()

```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/16.png?raw=true)



```python
eco = pd.read_csv('./data/시도_성별_경제활동인구_총괄_KOSIS_2022.csv', encoding= 'euc-kr')
# 데이터 정보
eco.info()
#결측치 여부 판단
eco.head()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 34 entries, 0 to 33
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   행정구역(시도)      34 non-null     object 
 1   성별            34 non-null     object 
 2   15세이상인구 (천명)  34 non-null     int64  
 3   경제활동인구 (천명)   34 non-null     int64  
 4   취업자 (천명)      34 non-null     int64  
 5   실업자 (천명)      34 non-null     int64  
 6   비경제활동인구 (천명)  34 non-null     int64  
 7   경제활동참가율(％)    34 non-null     float64
 8   고용률(%)        24 non-null     float64
 9   실업률(%)        28 non-null     float64
dtypes: float64(3), int64(5), object(2)
memory usage: 2.8+ KB

```

```python
#고용률 전처리 - 채우기
# 고용률 = 취업자/ 15세이상인구
# 배열 연산 가능
# 소수점은 1자리수만 남겨두기

```

```python
eco['고용률(%)'].fillna((eco['취업자 (천명)']/eco['15세이상인구 (천명)']*100).round(1),inplace= True)
eco.head()
```

```python
#실업률 전처리 - 채우기
# 실업률 = (실업자/경제활동인구)*100
# 배열 연산 간으!
# 소수점은 1자리수만 남겨두기1

```

```python
eco['실업률(%)'].fillna((eco['실업자 (천명)']/eco['경제활동인구 (천명)']*100).round(1),inplace=True)


eco.head()
```

```python
# 분석1) 시도별 고용율 평균 구해보기
# 분석2) 분석1 결과를 막대그래프 시각화 해보기
```

```python
# 분석 1
avg = eco.groupby(by='행정구역(시도)',as_index=False).agg(평균 = ('고용률(%)','mean'))
#as_index = 인덱스로 설정 x 그룹바이안에
avg
```

```python

plt.figure(figsize=(10,7))
plt.bar(avg.index,avg['평균'])
plt.ylim(55,80)
plt.xticks(rotation = 45)
plt.xlabel('행정구역(시도)')
plt.ylabel('고용률 평균(%)')
plt.title('시도별 고용룰 평균 시각화')
plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/17.png?raw=true)


```python
import seaborn as sns
plt.figure(figsize=(10,5))
sns.barplot(data=avg, x = '행정구역(시도)', y = '평균')
plt.ylim(55,80)
plt.xticks(rotation = 45)
# plt.xlabel('행정구역(시도)')
# plt.ylabel('고용률 평균(%)')
plt.title('시도별 고용룰 평균 시각화')
plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/18.png?raw=true)


#### Scatter, Pie

```python
#산점도 그래프
np.random.seed(2)
x = np.arange(1,4)
y = np.random.randint(1,200,size = 3)

# plt.scatter(x,y, s= y, c = ['r','g','b'])
plt.scatter(x,y, s= y, c =range(3),cmap='flag')
plt.colorbar()
plt.show()

#사고 위치 데이터를 기반으로 점(데이터 포인트)을 찍어본다.
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/19.png?raw=true)


```python
#pie plot
y2 = np.random.rand(3)
for i in range(3):
    print(f'{y2[i]/y2.sum()*100:2f}')

```

```
39.112067
30.976025
29.911908

```

```python
plt.pie(y2,labels=x, autopct='%.2f%%', colors= ['r','g','b'], explode=[1,0,0])
plt.show()
```

![image.png](https://github.com/designa11/designa11.github.io/blob/master/assets/images/plt/babo/20.png?raw=true)
