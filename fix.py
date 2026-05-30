import pandas as pd

# путь к исходному файлу
input_path = "INDPRO.csv"

# читаем файл как ОДИН столбец
df_raw = pd.read_csv(input_path, header=None)

# разбиваем строку по запятой
df_split = df_raw[0].str.split(",", expand=True)

# если в первой строке заголовки
if df_split.iloc[0, 0] == "observation_date":
    df_split.columns = ["date", "value"]
    df_split = df_split.iloc[1:]
else:
    df_split.columns = ["date", "value"]

# приводим типы
df_split["date"] = pd.to_datetime(df_split["date"])
df_split["value"] = pd.to_numeric(df_split["value"], errors="coerce")

# сортировка и сброс индекса
df_split = df_split.sort_values("date").reset_index(drop=True)

# сохраняем исправленный файл
output_path = "INDPRO_clean.csv"
df_split.to_csv(output_path, index=False)

print("Файл успешно исправлен и сохранён как:", output_path)
print(df_split.head())
