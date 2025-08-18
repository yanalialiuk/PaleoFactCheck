from data_builder import build_dataset
from fact_check import fact_check

if __name__ == "__main__":

    # build_dataset()  
    
    print("Проверка факта...")
    query = "Анкилозавр был хищником"
    result = fact_check(query)

    print("Результат:")
    print(result)