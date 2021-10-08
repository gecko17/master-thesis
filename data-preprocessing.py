import pandas as pd
import os

from hdbcli import dbapi
from langdetect import detect
from dotenv import load_dotenv
from transformers import BertTokenizer

from modules.datasets import SentenceClusterDataset, DocSplitDataset

load_dotenv()

connection = dbapi.connect(
    address='localhost',
    port=30015,
    user=os.getenv('hana_user'),
    password=os.getenv('hana_password'),
    currentSchema=os.getenv('hana_nc_schema'),
    connectTimeout=5000,
    communicationTimeout=60000
)

cursor = connection.cursor()
cursor.setfetchsize(100000)

# selecting data and building pandas DataFrame
stmt = 'SELECT "CONTENT", "LOCALE" \
        FROM "SAP_NEWS_CENTER_ARTICLES" \
        WHERE "LOCALE" = ? AND "CONTENT" is not null AND "DESCRIPTION" NOT LIKE ? AND "URL" NOT LIKE ? \
        LIMIT 5000 OFFSET ?'

df_result = None
for i in range(0, 20):
    offset = i * 5000
    print(f'selecting with offset {offset}')
    cursor.execute(
        stmt,
        ('de_DE',
         '%Video mit Transkript%',
         '%welt%plus%',
         offset))

    cols = [
        col[1][0]
        for col in enumerate(cursor.description)
    ]
    result_rows = cursor.fetchall()

    if cursor.rowcount > 0:
        # Construct the data frame
        if df_result is None:
            df_result = pd.DataFrame.from_records(result_rows, columns=cols)
        else:
            df_result = df_result.append(
                pd.DataFrame.from_records(
                    result_rows, columns=cols))


df = df_result.dropna()


def detect_language(x):
    try:
        return detect(x)
    except BaseException:
        return None


df['lang_detected'] = df['CONTENT'].apply(lambda x: detect_language(x))
df = df[df['lang_detected'] == 'de'][['CONTENT']]

df1 = df.iloc[:15000]
df2 = df.iloc[:30000]

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
sent_cluster_set = SentenceClusterDataset(tokenizer=tokenizer, df=df)
doc_split_set = DocSplitDataset(tokenizer=tokenizer, df=df)
