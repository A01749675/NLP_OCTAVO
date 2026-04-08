from charset_normalizer import from_path

result = from_path("data_train.xlsx").best()

if result is None:
    print("No text encoding detected. This is likely a binary file, such as .xlsx.")
else:
    print(result.encoding)