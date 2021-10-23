import pandas as pd

data_dict = {'Name': ['john', 'Sabre', 'Kim', 'Sato', 'Lee', 'Smith', 'David'],
             'Country': ['USA', 'France', 'Korea', 'Japan', 'Korea', 'USA', 'USA'],
             'Age': [31, 33, 28, 40, 36, 55, 48],
             'Job': ['Student', 'Lawyer', 'Developer', 'Chef', 'Professor', 'CEO', 'Banker']
             }

df = pd.DataFrame(data_dict)

# print(df)

# print(df.head()) #앞 5개 숫자 넣으면 그만큼
# print(df.tail()) #끝 5개
# print(df.describe())
# df.info()

# df.to_csv('dataframe_with_header_index.csv', header=True, index=True)
# df.to_csv('dataframe_without_header_index.csv', header=False, index=False)
# df.to_csv('dataframe_header_none.csv', header=None)
# df = pd.read_csv('dataframe_without_header_index.csv') #첫 행을 헤더로 인식
df = pd.read_csv('dataframe_without_header_index.csv', header=None)
print(df)
