from data_processing.data_builder import build_dataset
from fact_check import fact_check

if __name__ == "__main__":

    # build_dataset()  
    
    print("Проверка факта...")
    query = "Характерной особенностью анкилозавров  являются костные образования на туловище."
    result = fact_check(query)

    print("Результат:")
    print(result)