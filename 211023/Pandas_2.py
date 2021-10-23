import pandas as pd

data_dict = {'Name': ['john', 'Sabre', 'Kim', 'Sato', 'Lee', 'Smith', 'David'],
             'Country': ['USA', 'France', 'Korea', 'Japan', 'Korea', 'USA', 'USA'],
             'Age': [31, 33, 28, 40, 36, 55, 48],
             'Job': ['Student', 'Lawyer', 'Developer', 'Chef', 'Professor', 'CEO', 'Banker']
             }

df = pd.DataFrame(data_dict)


# 데이터프레임에서 열 단위 데이터 추출은 대괄호 안에 열 이름을 사용함
"""
df_job = df['Job']
print(df_job)
df_job = df[['Job']]
print(df_job)
"""

"""
df_country_job = df[['Country', 'Job']]
print(df_country_job)

cols = ['Country', 'Job']
df_country_job = df[cols]
print(df_country_job)
"""

# df.loc[인덱스], df.iloc[행번호]를 사용하여 행 단위로 데이터를 가져옴
"""
df_1st_row = df.loc[[0]]
print(df_1st_row)

df_1st_4th_row = df.loc[[0, 3]]
print(df_1st_4th_row)

df_slice = df.loc[0:3] # 0~3번째 행
print(df_slice)

print(df.loc[df['Country']=='USA'])
print(df.loc[df['Age']>30])
print(df.loc[df['Age']>30, ['Job']])
print(df.loc[(df['Age']>30) & (df['Job']=='Chef')])

print(df.loc[[1], :])
print(df.loc[[1, 3], ['Name', 'Job']])
print(df.loc[ :, :])
"""

# df.drop()을 통해 데이터프레임 행과 열 삭제 (axis=0 행, axis=1 열)
"""
#행 삭제
df = df.drop(1, axis=0)
print(df)
df = df.drop([3,5], axis=0)
print(df)
df = df.reset_index()
print(df)
df.drop(1, axis=0, inplace=True) #inplace=True면 데이터프레임에 바로 저장
print(df)
"""
"""
#열 삭제
df = df.drop('Age', axis=1)
print(df)
df.drop('Age', axis=1, inplace=True)
print(df)
"""

# 데이터프레임 행추가 df.append(dict_new_data, ignore_index=True)

# 데이터프레임 합치기 pd.concat()
