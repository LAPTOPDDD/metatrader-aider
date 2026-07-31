import os
import time

def run_evolution():
    while True:
        print("Запуск цикла оптимизации...")
        os.system("python quantum_lab.py")
        
        try:
            with open('last_result.txt', 'r') as f:
                content = f.read()
                print(f"Анализ результата: {content}")
                
                # Парсинг результата
                parts = content.split(',')
                sharpe = float(parts[0].split(':')[1].strip())
                
                if sharpe >= 1.5:
                    print("ЦЕЛЬ ДОСТИГНУТА! Sharpe Ratio >= 1.5")
                    break
                else:
                    print(f"Текущий Sharpe Ratio {sharpe}. Это ниже цели 1.5. Запуск автоматической доработки...")
                    # Формируем команду для Aider
                    cmd = f'aider --msg "Текущий Sharpe Ratio {sharpe}. Это плохо. Проанализируй strategy_logic.py, добавь новые казуальные фичи, улучши таргет и исправь код, чтобы побить 1.5. После правок ничего не спрашивай." --yes'
                    os.system(cmd)
                    
        except Exception as e:
            print(f"Ошибка при чтении результата: {e}")
            time.sleep(5)

if __name__ == '__main__':
    run_evolution()
