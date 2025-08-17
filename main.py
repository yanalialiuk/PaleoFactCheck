from data_layer import build_dataset
from fact_check import fact_check

if __name__ == "__main__":
    # print("Строим базу данных Chroma...")
    # build_dataset()  

    print("Проверка факта...")
    query = "Костные рыбы составляют 95% современной ихтиофауны. "
    result = fact_check(query)

    print("Результат:")
    print(result)
